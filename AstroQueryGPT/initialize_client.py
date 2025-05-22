# initialize_client.py
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv() # Loads environment variables from .env file

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL") 

if not API_KEY:
    print("Warning: OPENAI_API_KEY not found. LLM calls might fail or use defaults.")
    # For some open endpoints, 'None' or 'empty' might be acceptable if the library handles it.
    # However, OpenAI class usually expects a string.
    API_KEY = "None" # Placeholder if truly no key is needed and library supports this.

if not BASE_URL:
    print("Warning: OPENAI_BASE_URL not found. Using default OpenAI endpoint.")
    # Default for OpenAI client if no base_url is provided.
    llm_client = OpenAI(api_key=API_KEY)
else:
    llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Test basic connectivity if possible (optional)
# try:
#     llm_client.models.list()
#     print("LLM client initialized and connected successfully.")
# except Exception as e:
#     print(f"LLM client initialized, but connection test failed: {e}")
#     print("Ensure your API key and base URL are correct and the endpoint is reachable.")