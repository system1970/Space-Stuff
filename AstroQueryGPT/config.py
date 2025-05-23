"""
Configuration settings for the AstroQueryGPT application.

This module defines constants and settings used throughout the application,
including file paths, LLM parameters, RAG behavior, agent settings,
and UI configurations. Values can be overridden by environment variables
where specified.
"""
import os
import logging

logger = logging.getLogger(__name__)

# --- Files and Paths ---
SCHEMA_FILE_PATH = "sdss_schema_dr16.json"
"""Path to the SDSS schema JSON file."""

# --- LLM Configuration ---
# The model identifier for the LLM provider.
# Example: "openai/gpt-3.5-turbo" or "gpt-3.5-turbo" if OPENAI_BASE_URL is set.
LLM_PROVIDER_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-3.5-turbo")
"""Model identifier for the LLM provider. Can be set via OPENAI_MODEL env var."""

# Note: OPENAI_API_KEY and OPENAI_BASE_URL are typically loaded from .env
# in initialize_client.py and used there.

# --- RAG Configuration ---
DEFAULT_TOP_N_RESULTS = 10
"""Default number of results to request from the database (TOP N)."""

MIN_SEMANTIC_SCORE_THRESHOLD = 0.35
"""Minimum semantic similarity score for a table schema to be considered relevant by RAG."""

MAX_RAG_TABLES_CONTEXT = 2
"""Maximum number of top-scoring table schemas to include in the RAG context prompt."""

# --- Agent Configuration ---
MAX_AGENT_RETRIES = 2
"""Maximum number of retries the agent will attempt to correct a failed SQL query."""

# --- UI Configuration ---
MAX_DF_PREVIEW_ROWS = 10
"""Maximum number of rows to display in DataFrame previews in the Streamlit UI."""

logger.info("Configuration loaded.")