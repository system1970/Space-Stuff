import os

# --- Files and Paths ---
SCHEMA_FILE_PATH = "sdss_schema_dr16.json"

# --- LLM Configuration ---
# Example: "openai/gpt-3.5-turbo" or just "gpt-3.5-turbo" if base_url implies OpenAI compatibility
LLM_PROVIDER_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-3.5-turbo") # More specific if your client setup needs it

# OPENAI_API_KEY should be set in .env or environment
# --- RAG Configuration ---
DEFAULT_TOP_N_RESULTS = 10
MIN_SEMANTIC_SCORE_THRESHOLD = 0.35
MAX_RAG_TABLES_CONTEXT = 2 # How many top tables to include in RAG prompt

# --- Agent Configuration ---
MAX_AGENT_RETRIES = 2

# --- UI Configuration ---
MAX_DF_PREVIEW_ROWS = 10