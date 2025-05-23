"""
Core components for the Retrieval Augmented Generation (RAG) system.

This module includes functionalities for:
- Loading and managing the SDSS schema.
- Retrieving relevant schema parts based on user queries using semantic search.
- Constructing prompts for an LLM to generate SQL queries.
- Generating and correcting SQL queries using an LLM.
- Explaining SQL queries using an LLM.
"""
import json
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util

import config # Import shared configurations
from initialize_client import llm_client # Import the initialized LLM client

logger = logging.getLogger(__name__)

# --- Schema Loading ---
SDSS_SCHEMA_GLOBAL: List[Dict[str, Any]] = []
"""Global variable to store the loaded SDSS schema. Initialized by `initialize_rag_schema`."""

# Default path for the schema file, can be overridden by config.SCHEMA_FILE_PATH
DEFAULT_SCHEMA_MODEL = "all-MiniLM-L6-v2"
"""Default SentenceTransformer model for schema embedding and retrieval."""

MAX_FIELDS_PER_TABLE_IN_CORPUS = 30
"""Maximum number of fields from a single table to include in the semantic search corpus text."""

MAX_FIELDS_PER_TABLE_IN_PROMPT = 20
"""Maximum number of fields from a single table to include in the LLM prompt context."""


def load_sdss_schema(file_path: str = config.SCHEMA_FILE_PATH) -> List[Dict[str, Any]]:
    """
    Loads the SDSS schema from a JSON file.

    Args:
        file_path: Path to the JSON schema file. Defaults to `config.SCHEMA_FILE_PATH`.

    Returns:
        A list of dictionaries, where each dictionary represents a table schema.
        Returns an empty list if the file is not found or an error occurs during parsing.
    """
    if not os.path.exists(file_path):
        logger.error(f"Schema file not found at {file_path}")
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            schema_data = json.load(f)
            logger.info(f"Successfully loaded schema from {file_path}, {len(schema_data)} tables.")
            return schema_data
    except Exception as e:
        logger.error(f"Error loading or parsing schema from {file_path}: {e}", exc_info=True)
        return []

def initialize_rag_schema():
    """
    Initializes the global SDSS schema by loading it from the file.

    Raises:
        RuntimeError: If the SDSS schema cannot be loaded, as it's critical for RAG.
    """
    global SDSS_SCHEMA_GLOBAL
    if not SDSS_SCHEMA_GLOBAL: # Load only if not already loaded
        logger.info("Initializing RAG schema...")
        SDSS_SCHEMA_GLOBAL = load_sdss_schema()
        if not SDSS_SCHEMA_GLOBAL:
            logger.critical("RAG Core: SDSS Schema could not be loaded. Application may not function correctly.")
            # This error should be handled by the calling application (e.g., Streamlit UI)
            raise RuntimeError("RAG Core: SDSS Schema could not be loaded.")
        logger.info("RAG schema initialized successfully.")

# --- Semantic Retriever ---
def _embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """Helper function to embed a list of texts using the provided SentenceTransformer model."""
    if not texts:
        return np.array([])
    logger.debug(f"Embedding {len(texts)} texts...")
    embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
    logger.debug("Texts embedded successfully.")
    return embeddings

_retriever_model: Optional[SentenceTransformer] = None
def get_retriever_model() -> Optional[SentenceTransformer]:
    """
    Loads and returns the SentenceTransformer model for RAG.

    Uses a global variable to cache the loaded model for efficiency.

    Returns:
        The loaded SentenceTransformer model, or None if loading fails.
    """
    global _retriever_model
    if _retriever_model is None:
        logger.info(f"Loading sentence transformer model '{DEFAULT_SCHEMA_MODEL}' for RAG...")
        try:
            _retriever_model = SentenceTransformer(DEFAULT_SCHEMA_MODEL)
            logger.info("Sentence transformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading sentence transformer model '{DEFAULT_SCHEMA_MODEL}': {e}. RAG features will be impaired.", exc_info=True)
            # Returning None will cause downstream functions to fail if they don't handle it,
            # which should be caught by the application's error handling.
            return None
    return _retriever_model

def retrieve_relevant_schema(
    user_query: str,
    min_score_threshold: float = config.MIN_SEMANTIC_SCORE_THRESHOLD,
    top_k: int = config.MAX_RAG_TABLES_CONTEXT
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Retrieves relevant table schemas from the SDSS schema based on semantic similarity to the user query.

    Args:
        user_query: The user's natural language query.
        min_score_threshold: Minimum cosine similarity score for a table to be considered relevant.
        top_k: The maximum number of relevant tables to return.

    Returns:
        A list of tuples, where each tuple contains a table schema (dict) and its similarity score (float).
        Returns an empty list if the schema is not loaded or no relevant tables are found.
    """
    if not SDSS_SCHEMA_GLOBAL:
        logger.warning("SDSS_SCHEMA_GLOBAL is not initialized. Attempting to initialize...")
        try:
            initialize_rag_schema() # Ensure schema is loaded
        except RuntimeError: # If initialization fails
             logger.error("Failed to initialize RAG schema during retrieve_relevant_schema call.")
             return []
             
    if not SDSS_SCHEMA_GLOBAL: # Check again after attempt
        logger.error("RAG Core: Schema list is empty even after initialization attempt. Cannot retrieve relevant table.")
        return []

    model = get_retriever_model()
    if model is None:
        logger.error("Retriever model is not available. Cannot retrieve relevant schema.")
        return []

    # Prepare corpus of table and field descriptions for semantic search
    corpus = []
    table_refs = [] # To map corpus entries back to original schema dicts
    logger.debug(f"Building RAG corpus from {len(SDSS_SCHEMA_GLOBAL)} tables...")
    for table in SDSS_SCHEMA_GLOBAL:
        table_name = table.get('name', 'UnnamedTable')
        table_desc = table.get('description', 'No description.')
        table_text = f"Table Name: {table_name}. Description: {table_desc}"
        
        field_texts = []
        for f_idx, f in enumerate(table.get('fields', [])):
            if f_idx >= MAX_FIELDS_PER_TABLE_IN_CORPUS:
                field_texts.append(f"... (and {len(table.get('fields', [])) - MAX_FIELDS_PER_TABLE_IN_CORPUS} more fields)")
                break
            field_texts.append(
                f"Field: {f.get('name', 'N/A')} (Type: {f.get('type', 'N/A')}) Description: {f.get('description', 'N/A')}"
            )
        full_text = table_text + ". Fields: " + "; ".join(field_texts)
        corpus.append(full_text)
        table_refs.append(table)
    logger.debug(f"RAG corpus built with {len(corpus)} entries.")

    if not corpus:
        logger.warning("RAG corpus is empty. No schema information to search.")
        return []

    corpus_embeddings = _embed_texts(corpus, model)
    query_embedding = _embed_texts([user_query], model)

    if query_embedding.size == 0 or corpus_embeddings.size == 0:
        logger.error("Failed to generate embeddings for query or corpus.")
        return []
        
    similarities = util.cos_sim(query_embedding, corpus_embeddings)[0].numpy()

    num_candidates = min(top_k, len(similarities))
    top_indices_sorted = np.array([], dtype=int)

    if num_candidates > 0 and len(similarities) > 0:
        # Get indices of top_k highest scores, then reverse to descending order
        top_indices_sorted = np.argsort(similarities)[-num_candidates:][::-1] 
    
    results = []
    for i in top_indices_sorted:
        score = float(similarities[i])
        if score >= min_score_threshold:
            results.append((table_refs[i], score))
            logger.debug(f"Found relevant table '{table_refs[i]['name']}' with score {score:.2f}")
    
    # If no tables meet the threshold, fall back to the single best match if configured to do so (or if desired)
    # Current implementation (from previous step) falls back to best match if results is empty.
    if not results and corpus and len(similarities) > 0:
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        logger.info(
            f"No tables met RAG threshold {min_score_threshold}. "
            f"Falling back to best match: '{table_refs[best_idx]['name']}' (Score: {best_score:.2f})."
        )
        results = [(table_refs[best_idx], best_score)]
        
    if not results:
        logger.warning(f"No relevant schema found for query: '{user_query[:100]}...'")

    return results

# --- RAG Prompt Construction ---
def build_rag_prompt_for_sql_generation(
    user_query: str,
    table_schemas_with_scores: List[Tuple[Dict[str, Any], float]],
    top_n_results: int = config.DEFAULT_TOP_N_RESULTS
) -> str:
    """
    Constructs the prompt for the LLM to generate an SQL query.

    Args:
        user_query: The user's natural language query.
        table_schemas_with_scores: A list of relevant table schemas and their similarity scores.
        top_n_results: The number of results (TOP N) to include in the SQL query.

    Returns:
        A string representing the formatted prompt for the LLM.
    """
    logger.debug(f"Building RAG prompt for user query: '{user_query[:100]}...'")
    context_blocks = []
    if not table_schemas_with_scores:
        logger.warning("No table schema context provided for RAG prompt. Using general knowledge.")
        context_blocks.append(
            "No specific table schema context available. "
            "Please use general SDSS knowledge, focusing on common tables like "
            "PhotoObjAll or SpecObjAll if appropriate."
        )
    else:
        for table_schema, score in table_schemas_with_scores:
            field_lines = []
            num_fields = len(table_schema.get("fields", []))
            for f_idx, f in enumerate(table_schema.get("fields", [])):
                if f_idx < MAX_FIELDS_PER_TABLE_IN_PROMPT:
                     field_info = (
                         f"- {f.get('name', 'N/A')} "
                         f"(Type: {f.get('type', 'N/A')}, "
                         f"Description: {f.get('description', 'N/A')})"
                     )
                     field_lines.append(field_info)
                elif f_idx == MAX_FIELDS_PER_TABLE_IN_PROMPT:
                    field_lines.append(f"- ... (and {num_fields - MAX_FIELDS_PER_TABLE_IN_PROMPT} more fields)")
                    break
            fields_str = "\n".join(field_lines) if field_lines else "No detailed field information available for this table."
            context_blocks.append(
                f"Table Name: {table_schema.get('name', 'UnnamedTable')}\n"
                f"Table Description: {table_schema.get('description', 'No description.')}\n"
                f"Fields:\n{fields_str}\n"
                f"(Semantic Relevance Score to user query: {score:.2f})"
            )
    full_context = "\n\n---\n\n".join(context_blocks)
    
    # This is the core prompt structure for the LLM.
    # It guides the LLM on how to behave and what kind of output is expected.
    prompt = f"""You are an expert SDSS SQL query writer. Your task is to generate a single, valid, executable SQL query for the SDSS SkyServer (Transact-SQL dialect) based on the provided table schema(s) and the user's request.

Strictly follow these rules:
1. Analyze the user's request and the provided table schema(s) carefully.
2. **Prioritize using the table(s) and field(s) from the provided context that are most relevant to the user's request.**
3. **Only use table names and field names explicitly listed in the provided schema context.** Do not invent tables or fields. If crucial information seems missing from the context for a complete query, construct the best possible query using ONLY the provided information. Avoid making assumptions about fields not listed.
4. Construct a single, executable SQL query.
5. **Include `TOP {top_n_results}` in the SELECT clause.** For example: `SELECT TOP {top_n_results} ra, dec FROM PhotoObjAll`. This is crucial.
6. If the user asks for specific columns, SELECT those. Otherwise, if the request is general, you can use `SELECT TOP {top_n_results} *`.
7. Pay close attention to field types for correct WHERE clause conditions (e.g., strings in quotes, numeric types not in quotes).
8. Return ONLY the SQL query. No explanations, comments, or markdown formatting (like ```sql).

Provided Table Schema Context:
{full_context}

User's Request: "{user_query}"

SQL Query:"""
    logger.debug(f"Generated RAG prompt. Length: {len(prompt)}")
    return prompt.strip()

# --- Unified SQL Generation & Correction ---
def generate_and_correct_sql(
    original_user_query: str,
    rag_prompt_for_llm: str,
    top_n_results: int = config.DEFAULT_TOP_N_RESULTS,
    error_message: Optional[str] = None,
    prior_sql: Optional[str] = None,
    data_verification_failed: bool = False,
    failed_data_sample: Optional[str] = None
) -> Optional[str]:
    """
    Generates an SQL query using the LLM, or corrects a previous one based on errors.

    Args:
        original_user_query: The initial user query.
        rag_prompt_for_llm: The RAG-constructed prompt for initial SQL generation.
        top_n_results: The number of results (TOP N) to include in the SQL query.
        error_message: Database error from a previous execution attempt.
        prior_sql: The previously executed SQL query that failed.
        data_verification_failed: Flag indicating if previous data structure verification failed.
        failed_data_sample: A string sample of the data that failed verification.

    Returns:
        A string containing the generated or corrected SQL query, or None if generation fails.
    """
    if not llm_client:
        logger.error("RAG Core: LLM client not initialized. Cannot generate/correct SQL.")
        return None

    messages = []
    # Determine the model to use from config, stripping potential provider prefixes if needed by the client.
    model_to_call = config.LLM_PROVIDER_MODEL.split('/')[-1] if '/' in config.LLM_PROVIDER_MODEL else config.LLM_PROVIDER_MODEL
    
    current_mode = "generation"
    if error_message or data_verification_failed:
        current_mode = "correction"
        logger.info(f"Entering SQL correction mode. Error: '{error_message}', Data issue: {data_verification_failed}")
        system_correction_prompt = (
            f"You are an expert SDSS SQL query writer. Your task is to correct a previously generated SQL query "
            f"based on the provided database error message or data structure issue. "
            f"Ensure the corrected query strictly adheres to the SDSS SkyServer (Transact-SQL dialect) syntax. "
            f"The corrected query MUST include `TOP {top_n_results}` in the SELECT clause. "
            f"Return ONLY the corrected SQL query, with no additional text, comments, or markdown."
        )
        messages.append({"role": "system", "content": system_correction_prompt})
        messages.append({"role": "user", "content": f"Original user request to inform the correction: \"{original_user_query}\""})
        if prior_sql:
            messages.append({"role": "user", "content": f"The following SQL query needs correction:\n```sql\n{prior_sql}\n```"})
        if error_message:
            messages.append({"role": "user", "content": f"The database returned this error:\n{error_message}"})
        if data_verification_failed:
            messages.append({"role": "user", "content": "The data returned by the previous query was not well-structured, seemed empty, or like a single long string instead of tabular data."})
            if failed_data_sample:
                messages.append({"role": "user", "content": f"Here's a sample of the problematic data (first few rows/chars):\n{failed_data_sample}"})
        messages.append({"role": "user", "content": "Please provide a corrected SQL query that addresses these issues."})
    else:
        logger.info("Entering SQL generation mode.")
        # For initial generation, the user role directly contains the RAG prompt.
        messages.append({"role": "user", "content": rag_prompt_for_llm})

    logger.debug(f"LLM API call messages for {current_mode}: {messages}")

    try:
        response = llm_client.chat.completions.create(
            model=model_to_call, messages=messages, temperature=0.1, max_tokens=400 # Temperature is low for more deterministic SQL
        )
        raw_sql_query = response.choices[0].message.content.strip()
        logger.info(f"LLM ({current_mode}) raw response: '{raw_sql_query[:300]}...'")
        
        # --- SQL Cleaning Logic ---
        # This aims to extract the core SQL query if the LLM includes markdown or other text.
        cleaned_sql = raw_sql_query
        # Try to extract from ```sql ... ``` blocks first.
        sql_markdown_match = re.search(r"```(?:sql)?\s*(SELECT .*?)\s*```", cleaned_sql, re.IGNORECASE | re.DOTALL)
        if sql_markdown_match:
            cleaned_sql = sql_markdown_match.group(1).strip()
            logger.debug(f"Extracted SQL from markdown block: '{cleaned_sql[:200]}...'")
        else:
            # If not in a markdown block, try to find the "SELECT" statement,
            # allowing for some leading/trailing non-SQL text if it's not a correction flow.
            # In correction flow, we expect only SQL.
            select_direct_match = re.search(r"(SELECT\s+.*?)(?:;|$)", cleaned_sql, re.IGNORECASE | re.DOTALL)
            if select_direct_match:
                cleaned_sql = select_direct_match.group(1).strip()
                if cleaned_sql != raw_sql_query: # Log if changes were made
                     logger.debug(f"Extracted SQL by direct SELECT search: '{cleaned_sql[:200]}...'")
            elif not (error_message or data_verification_failed): # Not in correction flow & no SELECT found
                logger.warning(f"RAG Core: LLM response for generation doesn't appear to contain a SQL SELECT statement. Response: '{raw_sql_query[:300]}...'")
                # Return None as this is likely not a valid SQL query.
                return None
            # If in correction flow and no SELECT found, it might be an LLM comment (e.g., "I cannot correct this").
            # This might be an issue, but we'll let it pass to TOP N enforcement, which might fail it.
            # Ideally, the LLM should always return SQL or a clear "cannot do" that's still SQL-like.

        # --- Safer TOP N Clause Enforcement ---
        # Pattern to find "SELECT TOP N", "SELECT DISTINCT TOP N"
        top_n_pattern_search = r"SELECT\s+(DISTINCT\s+)?TOP\s+\d+"
        # Pattern to replace existing "TOP N" or insert it.
        # It targets "SELECT" or "SELECT DISTINCT" to insert "TOP N" after it.
        select_prefix_pattern = r"SELECT(\s+DISTINCT)?"
        
        existing_top_n_match = re.search(top_n_pattern_search, cleaned_sql, re.IGNORECASE)

        if existing_top_n_match:
            # A TOP N clause exists. Check if its value is correct.
            current_top_value_match = re.search(r"TOP\s+(\d+)", existing_top_n_match.group(0), re.IGNORECASE)
            if current_top_value_match and int(current_top_value_match.group(1)) == top_n_results:
                logger.debug(f"Correct TOP {top_n_results} clause already exists in LLM output.")
            else:
                # Incorrect TOP N value. Replace it.
                logger.debug(f"Incorrect TOP clause found: '{existing_top_n_match.group(0)}'. Replacing with TOP {top_n_results}.")
                # Remove old "TOP <number>"
                cleaned_sql = re.sub(r"TOP\s+\d+\s*", "", cleaned_sql, count=1, flags=re.IGNORECASE).strip()
                # Re-insert correct "TOP N" after "SELECT" or "SELECT DISTINCT"
                cleaned_sql = re.sub(select_prefix_pattern, rf"SELECT\1 TOP {top_n_results}", cleaned_sql, count=1, flags=re.IGNORECASE).strip()
                # The strip() and regex ensure spaces are handled.
        else:
            # No TOP N clause found. Add it.
            logger.debug(f"No TOP N clause found. Adding TOP {top_n_results}.")
            # Insert "TOP N" after "SELECT" or "SELECT DISTINCT"
            match_select_prefix = re.match(select_prefix_pattern, cleaned_sql, re.IGNORECASE)
            if match_select_prefix:
                # This is the most common case.
                prefix = match_select_prefix.group(0) # "SELECT" or "SELECT DISTINCT"
                rest_of_query = cleaned_sql[len(prefix):].strip()
                cleaned_sql = f"{prefix} TOP {top_n_results} {rest_of_query}"
            elif cleaned_sql.upper().startswith("WITH"): # Handle Common Table Expressions (CTEs)
                # For CTEs, TOP N should be in the final SELECT statement.
                # This is a simplified assumption; complex CTEs might need more nuanced handling.
                # We'll assume the LLM produces a CTE where TOP N can be added to the final SELECT.
                # A more robust solution might involve parsing the SQL structure.
                logger.warning("Query starts with 'WITH' (CTE). Attempting to add TOP N to the final SELECT. This might be fragile.")
                # This is a heuristic: find the last SELECT and try to inject TOP N.
                # This could fail for complex CTEs with multiple SELECTs in the final part.
                last_select_match = list(re.finditer(r"SELECT", cleaned_sql, re.IGNORECASE))
                if last_select_match:
                    last_select_pos = last_select_match[-1].start()
                    # Check if it's "SELECT DISTINCT"
                    is_distinct_match = re.match(r"SELECT\s+DISTINCT", cleaned_sql[last_select_pos:], re.IGNORECASE)
                    if is_distinct_match:
                        insert_pos = last_select_pos + len(is_distinct_match.group(0))
                        cleaned_sql = cleaned_sql[:insert_pos] + f" TOP {top_n_results} " + cleaned_sql[insert_pos:].strip()
                    else: # Just "SELECT"
                        insert_pos = last_select_pos + len("SELECT")
                        cleaned_sql = cleaned_sql[:insert_pos] + f" TOP {top_n_results} " + cleaned_sql[insert_pos:].strip()
                else: # No SELECT found in CTE (highly unlikely for valid SQL)
                    logger.error("CTE detected, but no SELECT found to add TOP N clause. Query may be invalid.")
                    return None # Or return as is, if we prefer to let the DB catch it.
            else:
                # If the query doesn't start with SELECT or WITH (e.g., it's just a comment or invalid),
                # then this query is likely malformed.
                logger.warning(f"LLM output after cleaning ('{cleaned_sql[:100]}...') does not start with SELECT or WITH. Cannot reliably enforce TOP N. Returning as is or None.")
                # Depending on strictness, either return None or the cleaned_sql and let the DB handle it.
                # Given the prompt's emphasis on a single SELECT, this indicates a problem.
                if not cleaned_sql.upper().startswith("SELECT"): # If it's not even a SELECT, it's bad.
                     return None


        logger.info(f"Final cleaned SQL ({current_mode}): '{cleaned_sql.strip()[:300]}...'")
        return cleaned_sql.strip() if cleaned_sql else None

    except Exception as e:
        logger.error(f"RAG Core: Error calling LLM or processing its response for {current_mode}: {e}", exc_info=True)
        return None

# --- SQL Explanation ---
def explain_sql_query(sql_query: str) -> Optional[str]:
    """
    Uses the LLM to generate a natural language explanation of an SQL query.

    Args:
        sql_query: The SQL query to explain.

    Returns:
        A string containing the explanation, or a default message if explanation fails or is unavailable.
    """
    if not llm_client:
        logger.warning("LLM client not available. Cannot explain SQL query.")
        return "LLM client not available, so I cannot provide an explanation for the SQL query."
    if not sql_query:
        logger.warning("No SQL query provided to explain.")
        return "No SQL query was provided to explain."
    
    logger.info(f"Requesting LLM explanation for SQL: '{sql_query[:200]}...'")
    model_to_call = config.LLM_PROVIDER_MODEL.split('/')[-1] if '/' in config.LLM_PROVIDER_MODEL else config.LLM_PROVIDER_MODEL
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Your task is to explain SQL queries related to the SDSS (Sloan Digital Sky Survey) astronomical database in simple, clear terms. Focus on what data the query retrieves and why it might be useful for an astronomer or student. Avoid jargon where possible, or explain it if necessary."},
        {"role": "user", "content": f"Please explain this SDSS SQL query in a way that's easy to understand:\n```sql\n{sql_query}\n```"}
    ]
    
    try:
        response = llm_client.chat.completions.create(
            model=model_to_call, messages=messages, temperature=0.3, max_tokens=300 # Slightly higher temp for more descriptive explanation
        )
        explanation = response.choices[0].message.content.strip()
        logger.info(f"LLM explanation received: '{explanation[:100]}...'")
        return explanation
    except Exception as e:
        logger.error(f"RAG Core: Error getting SQL explanation from LLM: {e}", exc_info=True)
        return "An error occurred while trying to generate an explanation for the SQL query."