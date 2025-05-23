"""
Initializes and configures the OpenAI client for the AstroQueryGPT application.

This module loads necessary environment variables (API key and base URL)
from a .env file or the environment. It then instantiates the OpenAI client,
making it available for use in other parts of the application.
"""
from openai import OpenAI
import os
import logging
from dotenv import load_dotenv

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Load environment variables from .env file if present
load_dotenv()
logger.info(".env file loaded (if present).")

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

if not API_KEY:
    logger.warning(
        "OPENAI_API_KEY not found in environment. "
        "Initializing client without an API key. LLM calls may fail or "
        "be restricted if the endpoint requires authentication."
    )
    API_KEY = None # Use Python's None object, as expected by the OpenAI library

if not BASE_URL:
    logger.warning(
        "OPENAI_BASE_URL not found in environment. "
        "Using default OpenAI API endpoint."
    )
    # When base_url is not provided, the OpenAI client defaults to the official OpenAI API.
    llm_client = OpenAI(api_key=API_KEY)
    logger.info("OpenAI client initialized with default base URL.")
else:
    llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    logger.info(f"OpenAI client initialized with custom base URL: {BASE_URL}")

# Optional: Test basic connectivity.
# This can be useful for immediate feedback during startup, but might add slight overhead.
# Consider enabling it during development or if connection issues are common.
# try:
#     logger.debug("Attempting to list models to test LLM client connectivity...")
#     llm_client.models.list()
#     logger.info("LLM client initialized and connection tested successfully.")
# except Exception as e:
#     logger.error(f"LLM client initialized, but connection test failed: {e}", exc_info=True)
#     logger.error(
#         "Ensure your API key (if required) and base URL are correct, "
#         "and the LLM endpoint is reachable."
#     )