import streamlit as st
import pandas as pd
import time 

import config
from rag_core import (
    initialize_rag_schema,
    retrieve_relevant_schema,
    build_rag_prompt_for_sql_generation,
    generate_and_correct_sql,
    explain_sql_query
)

try:
    from sdss_db import query_sdss
except ImportError:
    st.error("`sdss_db.py` not found or `query_sdss` is missing. SQL execution will be simulated.")
    def query_sdss(sql: str) -> pd.DataFrame:
        st.warning(f"SIMULATING SQL EXECUTION: {sql[:100]}...")
        if "error_test" in sql.lower(): raise ValueError("Simulated DB error")
        if "empty_test" in sql.lower(): return pd.DataFrame()
        time.sleep(0.5) # Simulate delay
        return pd.DataFrame({'sim_col1': range(3), 'sim_col2': [f'data_{i}' for i in range(3)]})

def verify_data_structure(df: pd.DataFrame) -> bool:
    if df.empty:
        print("Data Verification: DataFrame is empty.")
        return False 
    if len(df.columns) == 0:
        print("Data Verification: DataFrame has no columns.")
        return False
    if len(df) == 1 and len(df.columns) == 1:
        first_cell_value = str(df.iloc[0,0])
        if len(first_cell_value) > 200 and first_cell_value.count(' ') < 5:
            print("Data Verification: Looks like a single long string.")
            return False
        error_phrases = ["error", "failed", "unable", "cannot", "syntax", "invalid"]
        if any(phrase in first_cell_value.lower() for phrase in error_phrases) and len(first_cell_value) < 150:
            print(f"Data Verification: First cell contains potential error: {first_cell_value[:50]}...")
            return False
    print("Data Verification: Basic structure appears valid.")
    return True

def run_streamlit_app():
    st.set_page_config(layout="wide", page_title="SDSS Agentic SQL Bot")
    st.title("🔭 SDSS Agentic RAG SQL Generator")
    st.markdown("Ask a question about SDSS data, and the agent will generate and execute the SQL query!")

    try:
        initialize_rag_schema()
    except RuntimeError as e:
        st.error(str(e))
        return

    if 'query_log' not in st.session_state:
        st.session_state.query_log = []
    
    # Initialize top_tables_for_rag outside the conditional block
    top_tables_for_rag = None 

    left_column, right_column = st.columns([0.6, 0.4])

    with left_column:
        st.subheader("💬 Your Question & Controls")
        user_query = st.text_area(
            "Enter your astronomy question for SDSS:", height=100, key="user_query_input",
            placeholder="e.g., 'Find bright galaxies with high redshift'"
        )
        col_controls1, col_controls2 = st.columns(2)
        with col_controls1:
            top_n_results = st.number_input(
                "Number of results (TOP N)", min_value=1, max_value=1000, 
                value=config.DEFAULT_TOP_N_RESULTS, step=1, key="top_n_input"
            )
        with col_controls2:
            max_retries = st.slider(
                "Max Correction Retries:", 0, 5, config.MAX_AGENT_RETRIES, key="max_retries_slider"
            )
        submit_button = st.button("🚀 Generate & Execute SQL", use_container_width=True)

    results_placeholder = st.container()

    if submit_button and user_query:
        st.session_state.query_log = []
        current_sql_query = None
        db_error_message = None
        data_structure_ok = False
        last_failed_data_sample = None
        
        # Initialize progress value here
        current_progress_value = 0

        with results_placeholder:
            st.subheader("⚙️ Agent Processing...")
            progress_bar = st.progress(current_progress_value) # Initialize progress bar with 0
            status_text = st.empty()

            status_text.info("🔍 Step 1: Retrieving relevant schema context (RAG)...")
            current_progress_value = 10 # Set progress
            progress_bar.progress(current_progress_value)
            
            # top_tables_for_rag is defined here
            top_tables_for_rag = retrieve_relevant_schema(user_query) 
            
            if not top_tables_for_rag:
                status_text.error("🚫 Could not determine relevant table schema for your query using RAG.")
                st.stop()
            
            # Display RAG context in the right column immediately after retrieval
            with right_column:
                st.empty() # Clear previous content in right column if any
                st.subheader("🧠 RAG Context Used")
                with st.expander("Show Schema Context", expanded=False):
                    for t_schema, t_score in top_tables_for_rag:
                        st.markdown(f"**Table: {t_schema['name']}** (Score: {t_score:.2f})")
                        st.caption(f"{t_schema['description']}")
            
            rag_llm_prompt = build_rag_prompt_for_sql_generation(
                user_query, top_tables_for_rag, top_n_results
            )

            for attempt in range(max_retries + 1):
                status_text.info(f"🛠️ Step 2: LLM Generating SQL (Attempt {attempt + 1}/{max_retries + 1})...")
                # Calculate progress increment for this stage
                llm_progress_start = 20
                llm_progress_total_span = 60 # Reserve 60% for LLM and execution attempts
                progress_per_attempt_stage = llm_progress_total_span / (max_retries + 1)
                current_progress_value = int(llm_progress_start + (attempt * progress_per_attempt_stage))
                progress_bar.progress(current_progress_value)

                current_sql_query = generate_and_correct_sql(
                    original_user_query=user_query,
                    rag_prompt_for_llm=rag_llm_prompt,
                    top_n_results=top_n_results,
                    error_message=db_error_message,
                    prior_sql=st.session_state.query_log[-1]["sql"] if st.session_state.query_log and attempt > 0 else None,
                    data_verification_failed=not data_structure_ok and attempt > 0,
                    failed_data_sample=last_failed_data_sample
                )
                
                db_error_message = None
                data_structure_ok = False 
                last_failed_data_sample = None

                if not current_sql_query:
                    status_text.error(f"😔 Attempt {attempt + 1}: LLM failed to generate SQL.")
                    st.session_state.query_log.append({"attempt": attempt + 1, "sql": "LLM failed to generate SQL", "status": "LLM Error", "error": "No SQL returned"})
                    if attempt < max_retries: continue
                    else: break

                st.session_state.query_log.append({"attempt": attempt + 1, "sql": current_sql_query, "status": "Generated"})
                
                status_text.info(f"Executing SQL (Attempt {attempt + 1})...")
                current_progress_value = int(llm_progress_start + (attempt * progress_per_attempt_stage) + (progress_per_attempt_stage / 3)) # Mid-attempt progress
                progress_bar.progress(current_progress_value)
                try:
                    df_results = query_sdss(current_sql_query)
                    st.session_state.query_log[-1]["status"] = "Executed"
                    
                    status_text.info(f"Verifying data structure (Attempt {attempt + 1})...")
                    current_progress_value = int(llm_progress_start + (attempt * progress_per_attempt_stage) + (2 * progress_per_attempt_stage / 3)) # Further in attempt
                    progress_bar.progress(current_progress_value)
                    data_structure_ok = verify_data_structure(df_results)

                    if data_structure_ok:
                        status_text.success("✅ Query successful and data structure looks good!")
                        st.session_state.query_log[-1]["status"] = "Success & Verified"
                        st.session_state.query_log[-1]["data_preview"] = df_results.head(config.MAX_DF_PREVIEW_ROWS).to_markdown(index=False)
                        
                        # Display results in the main (left) column area
                        with left_column: # Ensure results are displayed under controls
                            st.subheader("📊 Query Results")
                            st.dataframe(df_results, height=300, use_container_width=True)
                            
                            st.subheader("📖 SQL Explanation")
                            with st.spinner("Getting SQL explanation from LLM..."):
                                explanation = explain_sql_query(current_sql_query)
                            st.info(explanation)
                        st.session_state.query_log[-1]["explanation"] = explanation
                        
                        current_progress_value = 100
                        progress_bar.progress(current_progress_value)
                        st.balloons()
                        break 
                    else:
                        st.session_state.query_log[-1]["status"] = "Executed, Data Structure Issue"
                        st.session_state.query_log[-1]["error"] = "Returned data structure seems invalid or empty."
                        last_failed_data_sample = df_results.head(5).to_string(index=False,max_colwidth=50) if not df_results.empty else "DataFrame was empty."
                        st.session_state.query_log[-1]["data_preview"] = last_failed_data_sample

                        if attempt < max_retries:
                            status_text.warning(f"⚠️ Data structure verification failed for attempt {attempt + 1}. Retrying...")
                            time.sleep(1) 
                        else:
                            status_text.error(f"🚫 Max retries reached. Data structure issue for SQL:\n```sql\n{current_sql_query}\n```")
                            with left_column:
                                st.subheader("📊 Final (Problematic) Query Results")
                                st.dataframe(df_results, height=300, use_container_width=True)
                            break

                except Exception as e:
                    db_error_message = str(e)
                    st.session_state.query_log[-1]["status"] = "Execution Error"
                    st.session_state.query_log[-1]["error"] = db_error_message
                    if attempt < max_retries:
                        status_text.warning(f"Attempt {attempt + 1}: SQL execution failed: {db_error_message}. Retrying...")
                        time.sleep(1)
                    else:
                        status_text.error(f"🚫 Max retries reached. Final SQL execution failed for:\n```sql\n{current_sql_query}\n```\nError: {db_error_message}")
                        break
            
            current_progress_value = 100 # Ensure progress bar completes to 100
            progress_bar.progress(current_progress_value)

    # Display query log/history in the right column
    # This section runs regardless of whether the submit button was pressed,
    # so top_tables_for_rag needs to be initialized.
    with right_column:
        # Only show "RAG Context Used" header if it wasn't shown during processing (i.e., button not pressed yet)
        if not (submit_button and user_query) and top_tables_for_rag is None:
            st.subheader("🧠 RAG Context")
            st.caption("Enter a query to see RAG context that will be used.")
        
        if st.session_state.query_log:
            st.subheader("📜 Agent Run Log")
            for i, log_entry in enumerate(st.session_state.query_log):
                attempt_num = log_entry.get("attempt", i + 1)
                exp_title = f"Attempt {attempt_num}: {log_entry.get('status', 'Unknown')}"
                is_last_expanded = (i == len(st.session_state.query_log) - 1)
                
                with st.expander(exp_title, expanded=is_last_expanded):
                    st.code(log_entry.get("sql", "N/A"), language="sql")
                    if "error" in log_entry and log_entry["error"]:
                        st.error(f"Error: {log_entry['error']}")
                    if "data_preview" in log_entry and log_entry["data_preview"]:
                        st.text("Data Preview:")
                        st.code(log_entry["data_preview"], language="text")
                    if "explanation" in log_entry and log_entry["explanation"]:
                        st.info(f"Explanation: {log_entry['explanation']}")
        elif submit_button and not user_query: # If button pressed but no query
            st.warning("Please enter a query.")


if __name__ == "__main__":
    run_streamlit_app()