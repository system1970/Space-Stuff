import pytest
import json
import os
from unittest.mock import mock_open, patch, MagicMock

# Ensure the logger is mocked or managed if it's used at the module level in rag_core
# For simplicity here, we assume it's not causing issues during import or can be handled by pytest-mock's auto-mocking features for global loggers.

# Import functions from rag_core
# To make this work, ensure AstroQueryGPT is in PYTHONPATH or use relative imports if tests are run as a module.
# For now, let's assume direct import works due to pytest path handling or PYTHONPATH setup.
from AstroQueryGPT import rag_core, config

# Initialize schema for tests that might need it (even if mocked)
@pytest.fixture(autouse=True)
def manage_global_schema(mocker):
    """Ensure SDSS_SCHEMA_GLOBAL is managed for tests."""
    mocker.patch.object(rag_core, 'SDSS_SCHEMA_GLOBAL', [])
    # If initialize_rag_schema is called, mock its behavior too
    mocker.patch('AstroQueryGPT.rag_core.initialize_rag_schema', MagicMock())


@pytest.fixture
def mock_llm_client(mocker):
    """Fixture to mock the llm_client used in rag_core."""
    client = MagicMock()
    mocker.patch('AstroQueryGPT.rag_core.llm_client', client)
    return client

# Tests for load_sdss_schema
def test_load_sdss_schema_success(mocker):
    """Test successful loading of SDSS schema."""
    mock_data = [{"name": "Table1", "columns": ["colA", "colB"]}]
    mock_file_content = json.dumps(mock_data)
    
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("builtins.open", mock_open(read_data=mock_file_content))
    mocker.patch("json.load", return_value=mock_data) # Also mock json.load directly
    
    schema = rag_core.load_sdss_schema("dummy_path.json")
    assert schema == mock_data
    json.load.assert_called_once()

def test_load_sdss_schema_file_not_found(mocker, caplog):
    """Test schema loading when file does not exist."""
    mocker.patch("os.path.exists", return_value=False)
    
    schema = rag_core.load_sdss_schema("non_existent_path.json")
    assert schema == []
    assert "Schema file not found" in caplog.text

def test_load_sdss_schema_corrupted_json(mocker, caplog):
    """Test schema loading with corrupted JSON content."""
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("builtins.open", mock_open(read_data="this is not json"))
    mocker.patch("json.load", side_effect=json.JSONDecodeError("Error", "doc", 0))
    
    schema = rag_core.load_sdss_schema("corrupted_path.json")
    assert schema == []
    assert "Error loading or parsing schema" in caplog.text

# Tests for build_rag_prompt_for_sql_generation
def test_build_rag_prompt_empty_schema(caplog):
    """Test prompt generation with no table schemas."""
    prompt = rag_core.build_rag_prompt_for_sql_generation(
        user_query="find galaxies",
        table_schemas_with_scores=[],
        top_n_results=10
    )
    assert "No specific table schema context available" in prompt
    assert "TOP 10" in prompt
    assert "find galaxies" in prompt
    assert "No table schema context provided for RAG prompt" in caplog.text


def test_build_rag_prompt_with_schema():
    """Test prompt generation with sample table schemas."""
    sample_schemas = [
        ({"name": "PhotoObj", "description": "Photometric objects", "fields": [{"name": "ra", "type": "float", "description": "Right Ascension"}]}, 0.9),
        ({"name": "SpecObj", "description": "Spectroscopic objects", "fields": [{"name": "z", "type": "float", "description": "Redshift"}]}, 0.8)
    ]
    prompt = rag_core.build_rag_prompt_for_sql_generation(
        user_query="find high redshift galaxies",
        table_schemas_with_scores=sample_schemas,
        top_n_results=25
    )
    assert "Table Name: PhotoObj" in prompt
    assert "Description: Photometric objects" in prompt
    assert "Field: ra (Type: float, Description: Right Ascension)" in prompt
    assert "Table Name: SpecObj" in prompt
    assert "Field: z (Type: float, Description: Redshift)" in prompt
    assert "(Semantic Relevance Score to user query: 0.90)" in prompt
    assert "TOP 25" in prompt
    assert "find high redshift galaxies" in prompt

def test_build_rag_prompt_top_n_insertion():
    """Verify top_n_results is correctly inserted."""
    prompt = rag_core.build_rag_prompt_for_sql_generation("query", [], 500)
    assert "TOP 500" in prompt
    assert "SELECT TOP 500" in prompt # Check specific phrasing from template

# Tests for generate_and_correct_sql (focus on cleaning and TOP N)
@pytest.mark.parametrize("llm_response, expected_sql", [
    ("SELECT * FROM TableA", "SELECT TOP 10 * FROM TableA"),
    ("```sql\nSELECT ra, dec FROM PhotoObj\n```", "SELECT TOP 10 ra, dec FROM PhotoObj"),
    ("Some leading text SELECT id FROM SpecObj WHERE z > 1; trailing text", "SELECT TOP 10 id FROM SpecObj WHERE z > 1"),
    ("  SELECT   field1, field2   FROM   MyTable  ", "SELECT TOP 10 field1, field2   FROM   MyTable"),
    ("SELECT TOP 5 * FROM TableB", "SELECT TOP 10 * FROM TableB"), # Existing TOP N is different
    ("SELECT DISTINCT TOP 7 name FROM Stars", "SELECT DISTINCT TOP 10 name FROM Stars"), # Existing TOP N with DISTINCT
    ("SELECT DISTINCT name FROM Stars", "SELECT DISTINCT TOP 10 name FROM Stars"), # No TOP N with DISTINCT
    ("WITH cte AS (SELECT id FROM Source) SELECT id FROM cte", "WITH cte AS (SELECT id FROM Source) SELECT TOP 10 id FROM cte") # Basic CTE
])
def test_generate_and_correct_sql_cleaning_and_top_n(mock_llm_client, llm_response, expected_sql):
    """Test SQL cleaning and TOP N enforcement."""
    # Mock the config value for DEFAULT_TOP_N_RESULTS if rag_core uses it directly
    # For this test, we assume top_n_results=10 is passed, so it should be used.
    
    mock_llm_client.chat.completions.create.return_value.choices[0].message.content = llm_response
    
    # Call with minimal args for cleaning and TOP N test
    # top_n_results is passed as 10 explicitly here to match expected_sql
    cleaned_query = rag_core.generate_and_correct_sql(
        original_user_query="test query",
        rag_prompt_for_llm="dummy_prompt", # Not relevant for this specific test focus
        top_n_results=10 
    )
    assert cleaned_query == expected_sql

def test_generate_and_correct_sql_llm_returns_no_select(mock_llm_client, caplog):
    """Test when LLM returns something that isn't SQL and not in correction mode."""
    mock_llm_client.chat.completions.create.return_value.choices[0].message.content = "Sorry, I cannot generate this query."
    
    result = rag_core.generate_and_correct_sql("query", "prompt", 10)
    assert result is None
    assert "LLM response for generation doesn't appear to contain a SQL SELECT statement" in caplog.text

def test_generate_and_correct_sql_llm_fails_completely(mock_llm_client, caplog):
    """Test when the LLM client call itself fails."""
    mock_llm_client.chat.completions.create.side_effect = Exception("LLM API Error")
    
    result = rag_core.generate_and_correct_sql("query", "prompt", 10)
    assert result is None
    assert "Error calling LLM or processing its response for generation: LLM API Error" in caplog.text

def test_generate_and_correct_sql_correction_flow_top_n(mock_llm_client):
    """Test that TOP N is also enforced in correction flow."""
    llm_response = "SELECT id FROM MyTable WHERE error_condition = 1"
    expected_corrected_sql = "SELECT TOP 20 id FROM MyTable WHERE error_condition = 1" # Assume top_n_results=20 for this test

    mock_llm_client.chat.completions.create.return_value.choices[0].message.content = llm_response

    corrected_sql = rag_core.generate_and_correct_sql(
        original_user_query="some query",
        rag_prompt_for_llm="initial_prompt", # Not used in correction
        top_n_results=20,
        error_message="DB error: syntax error", # Trigger correction flow
        prior_sql="SELECT TOP 10 id FROM MyTable"
    )
    assert corrected_sql == expected_corrected_sql

# Optional: Test for explain_sql_query
def test_explain_sql_query_constructs_prompt(mock_llm_client):
    """Test that explain_sql_query constructs the correct prompt for LLM."""
    sql_to_explain = "SELECT TOP 10 ra, dec FROM PhotoObjAll WHERE clean = 1"
    mock_llm_client.chat.completions.create.return_value.choices[0].message.content = "This is an explanation."

    rag_core.explain_sql_query(sql_to_explain)

    mock_llm_client.chat.completions.create.assert_called_once()
    call_args = mock_llm_client.chat.completions.create.call_args
    messages = call_args[1]['messages'] # or call_args.kwargs['messages']
    
    assert any(msg["role"] == "system" and "explain SQL queries" in msg["content"] for msg in messages)
    assert any(msg["role"] == "user" and sql_to_explain in msg["content"] for msg in messages)

def test_explain_sql_query_llm_fails(mock_llm_client, caplog):
    """Test explain_sql_query when the LLM call fails."""
    mock_llm_client.chat.completions.create.side_effect = Exception("LLM Explainer Error")
    
    explanation = rag_core.explain_sql_query("SELECT * FROM Table")
    assert "An error occurred while trying to generate an explanation" in explanation
    assert "Error getting SQL explanation from LLM: LLM Explainer Error" in caplog.text

def test_explain_sql_query_no_client_or_query():
    """Test explain_sql_query with no client or no query."""
    assert "LLM client not available" in rag_core.explain_sql_query("SELECT 1", llm_client=None) # Temporarily override for this call
    assert "No SQL query was provided" in rag_core.explain_sql_query("")


# Fixture to load actual config values for MAX_RAG_TABLES_CONTEXT etc.
# This might be more involved if config itself has dependencies or needs mocking.
# For now, assume config values are directly usable.

@pytest.fixture
def sample_schema_data():
    # A simplified schema for testing retrieve_relevant_schema
    return [
        {"name": "PhotoObjAll", "description": "All photometric objects from SDSS.", "fields": [{"name": "ra"}, {"name": "dec"}]},
        {"name": "SpecObjAll", "description": "All spectroscopic objects with redshift.", "fields": [{"name": "z"}, {"name": "plate"}]},
        {"name": "Galaxy", "description": "Information about galaxies.", "fields": [{"name": "objID"}, {"name": "petroRad_r"}]},
    ]

@pytest.fixture
def mock_retriever_model(mocker):
    """Mocks the SentenceTransformer model."""
    mock_model = MagicMock(spec=SentenceTransformer)
    # Mock the encode method to return deterministic embeddings
    # The actual embedding values don't matter as much as their relative similarities for some tests.
    # For more precise tests, specific embedding vectors would be needed.
    def mock_encode(texts, convert_to_tensor, show_progress_bar):
        if "photometric" in texts[0].lower(): return np.array([[0.1, 0.2, 0.7]]) # Galaxy-like
        if "spectroscopic" in texts[0].lower(): return np.array([[0.7, 0.2, 0.1]]) # SpecObj-like
        if "galaxies" in texts[0].lower(): return np.array([[0.1, 0.1, 0.8]]) # User query for galaxies
        return np.random.rand(len(texts), 3) # Fallback random embeddings
        
    mock_model.encode.side_effect = mock_encode
    mocker.patch('AstroQueryGPT.rag_core.get_retriever_model', return_value=mock_model)
    return mock_model

def test_retrieve_relevant_schema_basic(mocker, sample_schema_data, mock_retriever_model, caplog):
    """Test basic schema retrieval functionality."""
    mocker.patch.object(rag_core, 'SDSS_SCHEMA_GLOBAL', sample_schema_data)
    
    # Mock util.cos_sim to control similarity scores
    # This is important to make the test deterministic
    def mock_cos_sim(query_emb, corpus_emb):
        # Example: query for "galaxies" should be most similar to "Galaxy" table
        # query_emb for "galaxies" is [[0.1, 0.1, 0.8]]
        # corpus_emb for PhotoObjAll might be [[0.1, 0.2, 0.7]] -> sim for Photo (high)
        # corpus_emb for SpecObjAll might be [[0.7, 0.2, 0.1]] -> sim for Spec (low)
        # corpus_emb for Galaxy might be built from "Table Name: Galaxy. Description: Information about galaxies..."
        # For simplicity, let's assume the mock_encode handles this and we just need to return plausible scores
        # based on the mocked embeddings.
        # A more direct way is to mock cos_sim to return specific scores:
        # If query for "galaxies", make it similar to "Galaxy"
        # This example will use a more complex mock_cos_sim based on the mocked embeddings.
        
        # A simpler mock for cos_sim:
        # If the query was "galaxies", the query_embedding is [[0.1, 0.1, 0.8]]
        # Corpus embeddings are created based on table descriptions by mock_encode
        # PhotoObjAll: [[0.1, 0.2, 0.7]]
        # SpecObjAll: [[0.7, 0.2, 0.1]]
        # Galaxy: Let's assume its description also leads to something like [[0.1, 0.1, 0.8]] via mock_encode
        # For this test, let's directly mock util.cos_sim to return predefined scores.
        
        # This mock needs to align with how `corpus` is built in `retrieve_relevant_schema`
        # Order: PhotoObjAll, SpecObjAll, Galaxy
        # If user_query is "galaxies"
        # Similarity with PhotoObjAll, SpecObjAll, Galaxy
        mock_similarity_scores = np.array([[0.7, 0.3, 0.9]]) # Higher score for Galaxy
        return mock_similarity_scores

    mocker.patch('sentence_transformers.util.cos_sim', side_effect=mock_cos_sim)

    user_query = "find galaxies"
    results = rag_core.retrieve_relevant_schema(user_query, top_k=1, min_score_threshold=0.5)
    
    assert len(results) == 1
    assert results[0][0]['name'] == "Galaxy"
    assert results[0][1] == pytest.approx(0.9)

def test_retrieve_relevant_schema_no_good_match_fallback(mocker, sample_schema_data, mock_retriever_model, caplog):
    """Test fallback to best match if no schema meets threshold."""
    mocker.patch.object(rag_core, 'SDSS_SCHEMA_GLOBAL', sample_schema_data)
    mocker.patch('sentence_transformers.util.cos_sim', return_value=np.array([[0.4, 0.3, 0.2]])) # All below threshold 0.5

    results = rag_core.retrieve_relevant_schema("very obscure query", min_score_threshold=0.5, top_k=1)
    
    assert len(results) == 1
    assert results[0][0]['name'] == "PhotoObjAll" # Falls back to the first one (highest score 0.4)
    assert results[0][1] == pytest.approx(0.4)
    assert "No tables met RAG threshold 0.5. Falling back to best match" in caplog.text

def test_retrieve_relevant_schema_no_schema_loaded(mocker, caplog):
    """Test retrieval when schema is empty or fails to load."""
    mocker.patch.object(rag_core, 'SDSS_SCHEMA_GLOBAL', [])
    # Mock initialize_rag_schema to also fail or return empty
    mocker.patch('AstroQueryGPT.rag_core.load_sdss_schema', return_value=[])
    
    # Explicitly call initialize_rag_schema to simulate failure path if desired,
    # or rely on retrieve_relevant_schema's internal call.
    # We need to ensure the RuntimeError from initialize_rag_schema is handled or tested if it propagates.
    # For this test, let's assume initialize_rag_schema is called internally and might raise error or log.
    
    # If initialize_rag_schema raises RuntimeError, retrieve_relevant_schema should catch or test for it.
    # The current retrieve_relevant_schema tries to initialize, and if that fails (empty schema), it logs and returns [].
    
    results = rag_core.retrieve_relevant_schema("any query")
    assert results == []
    assert "Schema list is empty" in caplog.text or "Failed to initialize RAG schema" in caplog.text # Depending on exact flow

def test_retrieve_relevant_schema_no_retriever_model(mocker, sample_schema_data, caplog):
    """Test retrieval when the retriever model fails to load."""
    mocker.patch.object(rag_core, 'SDSS_SCHEMA_GLOBAL', sample_schema_data)
    mocker.patch('AstroQueryGPT.rag_core.get_retriever_model', return_value=None) # Simulate model loading failure
    
    results = rag_core.retrieve_relevant_schema("any query")
    assert results == []
    assert "Retriever model is not available" in caplog.text

def test_initialize_rag_schema_success(mocker):
    """Test successful RAG schema initialization."""
    mock_schema_data = [{"name": "TestTable"}]
    mocker.patch('AstroQueryGPT.rag_core.load_sdss_schema', return_value=mock_schema_data)
    rag_core.SDSS_SCHEMA_GLOBAL = [] # Reset before test
    
    rag_core.initialize_rag_schema()
    assert rag_core.SDSS_SCHEMA_GLOBAL == mock_schema_data

def test_initialize_rag_schema_load_fails(mocker, caplog):
    """Test RAG schema initialization when loading fails."""
    mocker.patch('AstroQueryGPT.rag_core.load_sdss_schema', return_value=[])
    rag_core.SDSS_SCHEMA_GLOBAL = [] # Reset
    
    with pytest.raises(RuntimeError, match="RAG Core: SDSS Schema could not be loaded."):
        rag_core.initialize_rag_schema()
    assert "SDSS Schema could not be loaded" in caplog.text

def test_get_retriever_model_loads_once(mocker):
    """Test that the retriever model is loaded only once."""
    mock_st_constructor = MagicMock(return_value=MagicMock(spec=SentenceTransformer))
    mocker.patch('sentence_transformers.SentenceTransformer', mock_st_constructor)
    
    rag_core._retriever_model = None # Ensure it's not loaded
    
    model1 = rag_core.get_retriever_model()
    model2 = rag_core.get_retriever_model()
    
    assert model1 is not None
    assert model1 is model2
    mock_st_constructor.assert_called_once()

def test_get_retriever_model_loading_exception(mocker, caplog):
    """Test handling of exception during model loading."""
    mocker.patch('sentence_transformers.SentenceTransformer', side_effect=Exception("Model load failed"))
    rag_core._retriever_model = None # Reset
    
    model = rag_core.get_retriever_model()
    assert model is None
    assert "Error loading sentence transformer model" in caplog.text
    assert "Model load failed" in caplog.text
    assert "RAG features will be impaired" in caplog.text
