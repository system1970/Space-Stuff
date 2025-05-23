"""
Streamlit web application for the SDSS Agentic RAG SQL Generator.

This application allows users to ask questions about SDSS data.
An agent then uses Retrieval Augmented Generation (RAG) to find relevant
SDSS schema information, generates an SQL query using an LLM,
executes the query against the SDSS database (or a simulation),
and displays the results and an explanation of the SQL.
"""
import streamlit as st
import pandas as pd
import time
import logging # Import logging module

# Configure basic logging for the application
# This should be done once, preferably at the very beginning of the app's entry point.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__) # Get a logger for this module

import config
from rag_core import (
    initialize_rag_schema,
    retrieve_relevant_schema,
    build_rag_prompt_for_sql_generation,
    generate_and_correct_sql,
    explain_sql_query
)

# Attempt to import the real database query function.
# If it fails, use a simulated version for local testing or when sdss_db.py is unavailable.
try:
    from sdss_db import query_sdss
    logger.info("Successfully imported query_sdss from sdss_db.")
except ImportError:
    logger.error("CRITICAL: `sdss_db.py` not found or `query_sdss` function is missing. Real database queries are disabled.")
    st.error(
        "CRITICAL: `sdss_db.py` not found or `query_sdss` function is missing. "
        "Real database queries are disabled. Using simulated data. "
        "Please check the application setup."
    )
    # Define a fallback simulated query_sdss function
    def query_sdss(sql: str) -> pd.DataFrame:
        """Simulated version of query_sdss for fallback."""
        logger.warning(f"SIMULATING SQL EXECUTION (fallback): {sql[:100]}...")
        st.warning(f"SIMULATING SQL EXECUTION: {sql[:100]}...") # Keep UI warning
        if "error_test" in sql.lower():
            logger.error("Simulated DB error triggered.")
            raise ValueError("Simulated DB error")
        if "empty_test" in sql.lower():
            logger.info("Simulated empty result triggered.")
            return pd.DataFrame()
        time.sleep(0.5) # Simulate network delay
        sim_data = {'sim_col1': range(3), 'sim_col2': [f'data_{i}' for i in range(3)]}
        logger.info(f"Simulated execution successful, returning data: {sim_data}")
        return pd.DataFrame(sim_data)

def verify_data_structure(df: pd.DataFrame) -> bool:
    """
    Performs basic verification of the structure of the DataFrame returned by a query.

    This helps catch cases where the LLM might generate SQL that returns
    a single long string, an error message, or an empty/malformed result.

    Args:
        df: The Pandas DataFrame to verify.

    Returns:
        True if the data structure seems valid, False otherwise.
    """
    logger.info(f"Verifying data structure. Shape: {df.shape}, Head:\n{df.head().to_string()}")
    if df.empty:
        logger.warning("Data Verification: Failed because DataFrame is empty.")
        return False
    if len(df.columns) == 0:
        logger.warning("Data Verification: Failed because DataFrame has no columns.")
        return False
    
    # Check for a common failure mode: a single cell containing a long string or error message
    if len(df) == 1 and len(df.columns) == 1:
        first_cell_value = str(df.iloc[0,0])
        # Heuristic: very long string with few spaces might be a concatenated error or unparsed data
        if len(first_cell_value) > 200 and first_cell_value.count(' ') < 5:
            logger.warning(f"Data Verification: Failed because it looks like a single long string. Preview: {first_cell_value[:100]}...")
            return False
        # Heuristic: common error phrases in the single cell
        error_phrases = ["error", "failed", "unable", "cannot", "syntax", "invalid", "incorrect"]
        if any(phrase in first_cell_value.lower() for phrase in error_phrases) and len(first_cell_value) < 150: # Arbitrary length for short errors
            logger.warning(f"Data Verification: Failed because first cell contains potential error message: {first_cell_value[:100]}...")
            return False
            
    logger.info("Data Verification: Basic structure appears valid.")
    return True

def display_rag_context(container, rag_results: list):
    """Displays the RAG context (retrieved table schemas) in the Streamlit UI."""
    container.empty() # Clear previous content if any
    with container:
        st.subheader("🧠 RAG Context Used")
        if not rag_results:
            st.caption("No RAG context was retrieved or used for this query.")
            return
        with st.expander("Show Schema Context", expanded=False):
            for t_schema, t_score in rag_results:
                st.markdown(f"**Table: {t_schema.get('name', 'N/A')}** (Score: {t_score:.2f})")
                st.caption(f"{t_schema.get('description', 'No description available.')}")

def display_query_log(container):
    """Displays the agent's run log (attempts, SQL, errors, etc.) in the Streamlit UI."""
    with container:
        if not st.session_state.get('query_log'):
            st.caption("Run log will appear here after a query is submitted.")
            return

        st.subheader("📜 Agent Run Log")
        for i, log_entry in enumerate(st.session_state.query_log):
            attempt_num = log_entry.get("attempt", i + 1)
            status = log_entry.get('status', 'Unknown')
            exp_title = f"Attempt {attempt_num}: {status}"
            # Expand the last entry by default
            is_last_expanded = (i == len(st.session_state.query_log) - 1)
            
            with st.expander(exp_title, expanded=is_last_expanded):
                st.code(log_entry.get("sql", "N/A"), language="sql")
                if "error" in log_entry and log_entry["error"]:
                    st.error(f"Error: {log_entry['error']}")
                if "data_preview" in log_entry and log_entry["data_preview"]:
                    st.text("Data Preview (from this attempt):")
                    st.code(log_entry["data_preview"], language="text")
                if "explanation" in log_entry and log_entry["explanation"]:
                    st.info(f"Explanation (for final successful SQL): {log_entry['explanation']}")


def run_streamlit_app():
    """
    Main function to run the Streamlit application.
    Sets up the UI, handles user input, and orchestrates the agent's workflow.
    """
    st.set_page_config(layout="wide", page_title="SDSS Agentic SQL Bot")
    logger.info("Streamlit app page configured.")

    st.title("🔭 SDSS Agentic RAG SQL Generator")
    st.markdown(
        "Ask a question about SDSS data (DR16), and the agent will attempt to: \n"
        "1. Understand your question using RAG with SDSS table schemas. \n"
        "2. Generate a Transact-SQL query. \n"
        "3. Execute the query against the SDSS database. \n"
        "4. Display the results and an explanation. \n"
        "The agent may perform multiple attempts if queries fail or data seems incorrect."
    )

    try:
        initialize_rag_schema() # Critical step, ensure schema is ready for RAG
        logger.info("RAG schema initialized successfully for the app.")
    except RuntimeError as e:
        logger.critical(f"Failed to initialize RAG schema: {e}", exc_info=True)
        st.error(f"A critical error occurred during RAG schema initialization: {e}. The application might not function correctly.")
        return # Stop further execution if RAG schema fails

    # Initialize session state variables
    if 'query_log' not in st.session_state:
        st.session_state.query_log = []
        logger.debug("Initialized 'query_log' in session state.")
    
    # UI Layout
    left_column, right_column = st.columns([0.6, 0.4]) # Main layout columns

    # --- Left Column: User Input and Results ---
    with left_column:
        st.subheader("💬 Your Question & Controls")
        user_query = st.text_area(
            "Enter your astronomy question for SDSS:", height=100, key="user_query_input",
            placeholder="e.g., 'Find bright galaxies with high redshift'"
        )
        
        # Control elements in columns for better layout
        col_controls1, col_controls2 = st.columns(2)
        with col_controls1:
            top_n_results = st.number_input(
                "Number of results (TOP N)", min_value=1, max_value=config.MAX_QUERY_RESULTS_LIMIT, # Use a config for max limit
                value=config.DEFAULT_TOP_N_RESULTS, step=1, key="top_n_input",
                help=f"Specifies the maximum number of rows the SQL query should return (using TOP N). Max allowed: {config.MAX_QUERY_RESULTS_LIMIT}"
            )
        with col_controls2:
            max_retries = st.slider(
                "Max Correction Retries:", 0, 5, config.MAX_AGENT_RETRIES, key="max_retries_slider",
                help="Maximum number of times the agent will attempt to correct a failing SQL query."
            )
        
        submit_button = st.button("🚀 Generate & Execute SQL", use_container_width=True, type="primary")

    results_placeholder = st.container() # Placeholder for results, allowing RAG context to be moved
    rag_context_placeholder_right = right_column.container() # Placeholder for RAG context in right column
    log_placeholder_right = right_column.container() # Placeholder for logs in right column

    # --- Agent Processing Logic ---
    if submit_button and user_query:
        logger.info(f"Submit button clicked. User query: '{user_query}', TOP N: {top_n_results}, Max Retries: {max_retries}")
        st.session_state.query_log = [] # Reset log for new query
        
        # Initialize state for the current run
        current_sql_query: Optional[str] = None
        db_error_message: Optional[str] = None
        data_structure_ok: bool = False
        last_failed_data_sample: Optional[str] = None
        top_tables_for_rag: list = [] # Ensure it's initialized for this run

        with results_placeholder: # Processing messages will appear here
            st.subheader("⚙️ Agent Processing...")
            progress_bar = st.progress(0, text="Initializing...") # Text for progress bar
            status_text = st.empty() # For dynamic status updates

            # --- Step 1: Retrieve RAG Context ---
            status_text.info("🔍 Step 1/4: Retrieving relevant schema context (RAG)...")
            progress_bar.progress(10, text="RAG: Retrieving schema...")
            logger.debug("Attempting to retrieve relevant schema via RAG.")
            top_tables_for_rag = retrieve_relevant_schema(user_query) 
            
            if not top_tables_for_rag:
                logger.warning("RAG could not determine relevant table schema for the query.")
                status_text.error("🚫 Could not determine relevant table schema for your query using RAG. Please try rephrasing your question or check if the schema is loaded correctly.")
                st.stop() # Stop processing if no RAG context
            
            logger.info(f"RAG retrieved {len(top_tables_for_rag)} table(s) for context.")
            display_rag_context(rag_context_placeholder_right, top_tables_for_rag) # Display RAG context immediately

            # --- Step 2: Build RAG Prompt & LLM Interaction Loop ---
            rag_llm_prompt = build_rag_prompt_for_sql_generation(
                user_query, top_tables_for_rag, top_n_results
            )
            
            max_attempts = max_retries + 1
            for attempt in range(max_attempts):
                attempt_num = attempt + 1
                logger.info(f"Attempt {attempt_num}/{max_attempts} for query: '{user_query[:50]}...'")
                
                status_text.info(f"🛠️ Step 2/4: LLM Generating SQL (Attempt {attempt_num}/{max_attempts})...")
                progress_value_llm_start = 20 + int(60 * (attempt / max_attempts)) # Progress for LLM stage
                progress_bar.progress(progress_value_llm_start, text=f"LLM: Generating SQL (Attempt {attempt_num})")

                current_sql_query = generate_and_correct_sql(
                    original_user_query=user_query,
                    rag_prompt_for_llm=rag_llm_prompt, # Used only on first attempt
                    top_n_results=top_n_results,
                    error_message=db_error_message, # From previous failed attempt
                    prior_sql=st.session_state.query_log[-1]["sql"] if st.session_state.query_log and attempt > 0 else None,
                    data_verification_failed=not data_structure_ok and attempt > 0, # If prev verification failed
                    failed_data_sample=last_failed_data_sample
                )
                
                # Reset error/data states for this new attempt
                db_error_message = None
                data_structure_ok = False 
                last_failed_data_sample = None

                if not current_sql_query:
                    logger.error(f"LLM failed to generate SQL on attempt {attempt_num}.")
                    status_text.error(f"😔 Attempt {attempt_num}: LLM failed to generate SQL. Check logs for details.")
                    st.session_state.query_log.append({"attempt": attempt_num, "sql": "LLM failed to generate SQL", "status": "LLM Error", "error": "No SQL returned by LLM."})
                    if attempt < max_retries: continue # Go to next retry
                    else: break # Max retries reached for LLM generation

                log_entry_base = {"attempt": attempt_num, "sql": current_sql_query, "status": "Generated by LLM"}
                st.session_state.query_log.append(log_entry_base)
                logger.debug(f"Attempt {attempt_num} generated SQL: {current_sql_query}")
                
                # --- Step 3: Execute SQL ---
                status_text.info(f"Executing SQL (Attempt {attempt_num})...")
                progress_bar.progress(progress_value_llm_start + int(10 / max_attempts) , text=f"DB: Executing SQL (Attempt {attempt_num})")
                try:
                    df_results = query_sdss(current_sql_query)
                    st.session_state.query_log[-1]["status"] = "Executed Successfully"
                    logger.info(f"Attempt {attempt_num} SQL executed. Result shape: {df_results.shape}")
                    
                    # --- Step 4: Verify Data Structure ---
                    status_text.info(f"Verifying data structure (Attempt {attempt_num})...")
                    progress_bar.progress(progress_value_llm_start + int(20 / max_attempts), text=f"Agent: Verifying data (Attempt {attempt_num})")
                    data_structure_ok = verify_data_structure(df_results)

                    if data_structure_ok:
                        logger.info(f"Attempt {attempt_num}: Data structure verified successfully.")
                        status_text.success("✅ Query successful and data structure looks good!")
                        st.session_state.query_log[-1]["status"] = "Success & Verified"
                        st.session_state.query_log[-1]["data_preview"] = df_results.head(config.MAX_DF_PREVIEW_ROWS).to_markdown(index=False)
                        
                        with left_column: # Display results in the main (left) column area
                            st.subheader("📊 Query Results")
                            st.dataframe(df_results, height=300, use_container_width=True)
                            
                            st.subheader("📖 SQL Explanation")
                            with st.spinner("Getting SQL explanation from LLM..."):
                                explanation = explain_sql_query(current_sql_query)
                            st.info(explanation if explanation else "Could not retrieve explanation.")
                        st.session_state.query_log[-1]["explanation"] = explanation
                        
                        progress_bar.progress(100, text="Completed!")
                        st.balloons()
                        break # Successful attempt, exit loop
                    else:
                        logger.warning(f"Attempt {attempt_num}: Data structure verification failed.")
                        st.session_state.query_log[-1]["status"] = "Executed, Data Structure Issue"
                        st.session_state.query_log[-1]["error"] = "Returned data structure seems invalid, empty, or like an error message."
                        last_failed_data_sample = df_results.head(config.MAX_DF_PREVIEW_ROWS_IN_LOG).to_string(index=False,max_colwidth=50) if not df_results.empty else "DataFrame was empty."
                        st.session_state.query_log[-1]["data_preview"] = last_failed_data_sample

                        if attempt < max_retries:
                            status_text.warning(f"⚠️ Data structure verification failed for attempt {attempt_num}. The agent will try to correct the SQL. Retrying...")
                            time.sleep(1) # Brief pause for user to see message
                        else: # Max retries reached
                            status_text.error(f"🚫 Max retries reached. Data structure issue for SQL:\n```sql\n{current_sql_query}\n```")
                            logger.error("Max retries reached due to data structure issue.")
                            with left_column: # Still show the problematic data
                                st.subheader("📊 Final (Problematic) Query Results")
                                st.dataframe(df_results, height=300, use_container_width=True)
                            break # Exit loop

                except Exception as e:
                    logger.error(f"Attempt {attempt_num}: SQL execution failed: {e}", exc_info=True)
                    db_error_message = str(e)
                    st.session_state.query_log[-1]["status"] = "Execution Error"
                    st.session_state.query_log[-1]["error"] = db_error_message
                    if attempt < max_retries:
                        status_text.warning(f"Attempt {attempt_num}: SQL execution failed: {db_error_message}. The agent will try to correct it. Retrying...")
                        time.sleep(1) # Brief pause
                    else: # Max retries reached
                        status_text.error(f"🚫 Max retries reached. Final SQL execution failed for:\n```sql\n{current_sql_query}\n```\nError: {db_error_message}")
                        logger.error("Max retries reached due to SQL execution error.")
                        break # Exit loop
            
            progress_bar.progress(100, text="Processing complete.") # Ensure progress bar completes
            logger.info("Agent processing loop finished.")

    # --- Right Column: RAG Context and Agent Log ---
    # Display RAG context if not already shown (e.g., if query hasn't run yet)
    if not (submit_button and user_query) and 'top_tables_for_rag' not in locals() and not st.session_state.get('query_log'):
        with rag_context_placeholder_right:
            st.subheader("🧠 RAG Context")
            st.caption("Enter a query to see the RAG context that will be used by the LLM.")
            
    display_query_log(log_placeholder_right) # Display the log

    if submit_button and not user_query:
        logger.warning("Submit button pressed without a user query.")
        st.warning("Please enter a question before submitting.")


if __name__ == "__main__":
    logger.info("Application starting...")
    run_streamlit_app()
    logger.info("Application finished.")