import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util

import config # Import shared configurations
from initialize_client import llm_client # Import the initialized LLM client

# --- Schema Loading ---
SDSS_SCHEMA_GLOBAL: List[Dict[str, Any]] = []

def load_sdss_schema(file_path: str = config.SCHEMA_FILE_PATH) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        print(f"Error: Schema file not found at {file_path}")
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading or parsing schema from {file_path}: {e}")
        return []

def initialize_rag_schema():
    global SDSS_SCHEMA_GLOBAL
    if not SDSS_SCHEMA_GLOBAL:
        SDSS_SCHEMA_GLOBAL = load_sdss_schema()
        if not SDSS_SCHEMA_GLOBAL:
            # This should be handled by the calling app (e.g., Streamlit)
            raise RuntimeError("RAG Core: SDSS Schema could not be loaded.")

# --- Semantic Retriever ---
def _embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(texts, convert_to_tensor=False, show_progress_bar=False)

_retriever_model = None
def get_retriever_model():
    global _retriever_model
    if _retriever_model is None:
        print("Loading sentence transformer model for RAG...")
        _retriever_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _retriever_model

def retrieve_relevant_schema(
    user_query: str,
    min_score_threshold: float = config.MIN_SEMANTIC_SCORE_THRESHOLD,
    top_k: int = config.MAX_RAG_TABLES_CONTEXT
) -> List[Tuple[Dict[str, Any], float]]:
    if not SDSS_SCHEMA_GLOBAL:
        initialize_rag_schema() # Ensure schema is loaded
    if not SDSS_SCHEMA_GLOBAL:
        print("RAG Core: Schema list is empty. Cannot retrieve relevant table.")
        return []

    corpus = []
    table_refs = []
    for table in SDSS_SCHEMA_GLOBAL:
        table_text = f"Table Name: {table.get('name', '')}. Description: {table.get('description', '')}"
        field_texts = [
            f"Field: {f.get('name', '')} (Type: {f.get('type', '')}) Description: {f.get('description', '')}"
            for f_idx, f in enumerate(table.get('fields', [])) if f_idx < 30 # Limit fields per table in corpus
        ]
        full_text = table_text + ". Fields: " + "; ".join(field_texts)
        corpus.append(full_text)
        table_refs.append(table)

    model = get_retriever_model()
    corpus_embeddings = _embed_texts(corpus, model)
    query_embedding = _embed_texts([user_query], model)
    similarities = util.cos_sim(query_embedding, corpus_embeddings)[0].numpy()

    num_candidates = min(top_k, len(similarities))
    top_indices_sorted = np.array([], dtype=int)
    if num_candidates > 0 and len(similarities) > 0 :
        top_indices_sorted = np.argsort(similarities)[-num_candidates:][::-1]
    
    results = []
    for i in top_indices_sorted:
        if similarities[i] >= min_score_threshold:
            results.append((table_refs[i], float(similarities[i])))
    
    if not results and corpus and len(similarities)>0 :
        best_idx = int(np.argmax(similarities))
        # print(f"RAG Core: No tables met threshold {min_score_threshold}. Falling back to best (Score: {similarities[best_idx]:.2f}).")
        results = [(table_refs[best_idx], float(similarities[best_idx]))]
    return results

# --- RAG Prompt Construction ---
def build_rag_prompt_for_sql_generation(
    user_query: str,
    table_schemas_with_scores: List[Tuple[Dict[str, Any], float]],
    top_n_results: int = config.DEFAULT_TOP_N_RESULTS
) -> str:
    # ... (largely same as your previous version, ensure it uses top_n_results correctly) ...
    context_blocks = []
    if not table_schemas_with_scores:
        context_blocks.append("No specific table schema context available. Please use general SDSS knowledge, focusing on common tables like PhotoObjAll or SpecObjAll if appropriate.")
    else:
        for table_schema, score in table_schemas_with_scores:
            field_lines = []
            for f_idx, f in enumerate(table_schema.get("fields", [])):
                if f_idx < 20: 
                     field_info = f"- {f.get('name', '')} (Type: {f.get('type', '')}, Description: {f.get('description', '')})"
                     field_lines.append(field_info)
                elif f_idx == 20:
                    field_lines.append(f"- ... (and {len(table_schema.get('fields', [])) - 20} more fields)")
                    break
            fields_str = "\n".join(field_lines) if field_lines else "No detailed field information available."
            context_blocks.append(
                f"Table Name: {table_schema['name']}\n"
                f"Table Description: {table_schema['description']}\n"
                f"Fields:\n{fields_str}\n"
                f"(Relevance Score: {score:.2f})"
            )
    full_context = "\n\n---\n\n".join(context_blocks)
    prompt = f"""You are an expert SDSS SQL query writer. Your task is to generate a single, valid, executable SQL query for the SDSS SkyServer (Transact-SQL dialect) based on the provided table schema(s) and the user's request.

Strictly follow these rules:
1. Analyze the user's request and the provided table schema(s) carefully.
2. **Prioritize using the table(s) and field(s) from the provided context that are most relevant to the user's request.**
3. **Only use table names and field names explicitly listed in the provided schema context.** Do not invent tables/fields. If crucial information is missing, make the best possible query using only the provided information.
4. Construct a single, executable SQL query.
5. **Include `TOP {top_n_results}` in the SELECT clause.** Example: `SELECT TOP {top_n_results} ra, dec FROM PhotoObjAll`.
6. If the user asks for specific columns, SELECT those. Otherwise, use `SELECT TOP {top_n_results} *`.
7. Pay close attention to field types for correct WHERE clause conditions.
8. Return ONLY the SQL query. No explanations, comments, or markdown formatting (like ```sql).

Provided Table Schema Context:
{full_context}

User's Request: "{user_query}"

SQL Query:"""
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
    if not llm_client:
        print("RAG Core: LLM client not initialized.")
        return None

    messages = []
    model_to_call = config.LLM_PROVIDER_MODEL.split('/')[-1] if '/' in config.LLM_PROVIDER_MODEL else config.LLM_PROVIDER_MODEL

    if error_message or data_verification_failed:
        system_correction_prompt = (
            f"You are an expert SDSS SQL query writer. Correct a previously generated SQL query based on an error or data mismatch. "
            f"Ensure the corrected query uses `TOP {top_n_results}`. Return ONLY the corrected SQL query."
        )
        messages.append({"role": "system", "content": system_correction_prompt})
        messages.append({"role": "user", "content": f"Original user request: \"{original_user_query}\""})
        if prior_sql: messages.append({"role": "user", "content": f"Previous SQL:\n```sql\n{prior_sql}\n```"})
        if error_message: messages.append({"role": "user", "content": f"Database error:\n{error_message}"})
        if data_verification_failed:
            messages.append({"role": "user", "content": "Data returned by previous query was not well-structured or seemed like gibberish."})
            if failed_data_sample: messages.append({"role": "user", "content": f"Sample of problematic data:\n{failed_data_sample}"})
        messages.append({"role": "user", "content": "Please provide a corrected SQL query."})
    else:
        messages.append({"role": "user", "content": rag_prompt_for_llm})

    try:
        response = llm_client.chat.completions.create(
            model=model_to_call, messages=messages, temperature=0.1, max_tokens=400
        )
        raw_sql_query = response.choices[0].message.content.strip()
        
        cleaned_sql = raw_sql_query
        match = re.search(r"^(?:```(?:sql)?)?\s*(SELECT .*)\s*(?:```)?$", cleaned_sql, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if match: cleaned_sql = match.group(1).strip()
        
        if not cleaned_sql.upper().startswith("SELECT"):
            select_match = re.search(r"(SELECT\s+TOP\s+\d+\s+.*?(?:;)?)$", cleaned_sql, re.IGNORECASE | re.DOTALL) # Simplified regex
            if select_match: cleaned_sql = select_match.group(1).strip()
            else:
                if not (error_message or data_verification_failed):
                    print(f"RAG Core: LLM response doesn't look like SQL: {cleaned_sql}")
                    return None
        
        # Ensure TOP N
        cleaned_sql = re.sub(rf"TOP\s+\d+\s+", "", cleaned_sql, count=1, flags=re.IGNORECASE).strip() # Remove existing TOP N, more robust
        if cleaned_sql.upper().startswith("SELECT DISTINCT"):
            cleaned_sql = f"SELECT DISTINCT TOP {top_n_results} " + cleaned_sql[len("SELECT DISTINCT"):].strip()
        elif cleaned_sql.upper().startswith("SELECT"):
            cleaned_sql = f"SELECT TOP {top_n_results} " + cleaned_sql[len("SELECT"):].strip()
        
        return cleaned_sql.strip() if cleaned_sql else None

    except Exception as e:
        print(f"RAG Core: Error calling LLM: {e}")
        return None

# --- SQL Explanation ---
def explain_sql_query(sql_query: str) -> Optional[str]:
    if not llm_client or not sql_query:
        return "LLM client not available or no SQL to explain."
    
    model_to_call = config.LLM_PROVIDER_MODEL.split('/')[-1] if '/' in config.LLM_PROVIDER_MODEL else config.LLM_PROVIDER_MODEL
    messages = [
        {"role": "system", "content": "You are a helpful assistant who explains SQL queries related to the SDSS astronomical database in simple terms. Focus on what data the query retrieves and why."},
        {"role": "user", "content": f"Please explain this SDSS SQL query:\n```sql\n{sql_query}\n```"}
    ]
    try:
        response = llm_client.chat.completions.create(
            model=model_to_call, messages=messages, temperature=0.3, max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"RAG Core: Error getting SQL explanation: {e}")
        return "Could not generate an explanation for the SQL query."