"""
Handles interactions with the SDSS SkyServer database.

This module provides functionality to execute SQL queries against the
SDSS SkyServer, retrieve results, and perform basic data processing
such as type conversions for compatibility with downstream tools.
"""
from io import StringIO
import pandas as pd
import requests
import logging
# import time # No longer used directly, can be removed if not needed by config
# import config # No longer used directly, can be removed if not needed by config

logger = logging.getLogger(__name__)

# SDSS SkyServer Query URL
SDSS_API_URL = "https://skyserver.sdss.org/dr16/en/tools/search/x_sql.aspx"
# Timeout for the HTTP request in seconds
REQUEST_TIMEOUT = 60

def query_sdss(sql_query: str) -> pd.DataFrame:
    """
    Executes an SQL query against the SDSS SkyServer DR16.

    Args:
        sql_query: The SQL query string to execute.

    Returns:
        A Pandas DataFrame containing the query results.
        Returns an empty DataFrame if the query yields no results or if an error occurs
        that is handled by returning an empty DataFrame (e.g., empty response from server).

    Raises:
        ValueError: If SDSS returns an HTML error page or a detectable SQL error message.
        requests.exceptions.Timeout: If the query times out.
        requests.exceptions.HTTPError: For other HTTP-related errors.
        Exception: For other unexpected errors during parsing or processing.

    Notes:
        - Converts columns with "id" in their name (case-insensitive) to strings
          to prevent JavaScript number precision issues in Streamlit.
        - Detects common SDSS error messages and HTML responses.
    """
    params = {"cmd": sql_query, "format": "csv"}
    
    logger.info(f"Executing SQL against SDSS SkyServer: {sql_query[:250]}...") # Log more of the query
    
    try:
        response = requests.get(SDSS_API_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Raises HTTPError for 4xx/5xx responses

        content_type = response.headers.get("Content-Type", "").lower()
        is_html_response = "text/html" in content_type # More robust check for HTML

        # Check for HTML responses or specific error strings in the response text
        # These often indicate a problem with the SQL query or server-side issues.
        if is_html_response:
            error_preview = response.text[:500].replace('\n', ' ').strip()
            if "error near" in response.text.lower():
                logger.error(f"SDSS SQL Error (detected in HTML response, Content-Type: {content_type}): {error_preview}")
                raise ValueError(f"SDSS SQL Error (detected in HTML response): {error_preview}")
            logger.warning(f"SDSS returned an HTML page (Content-Type: {content_type}), not CSV. Preview: {error_preview}")
            raise ValueError(f"SDSS returned an HTML page (Content-Type: {content_type}), not CSV. Preview: {error_preview}")
        
        # Check for error messages even if Content-Type wasn't HTML (e.g. plain text errors from the server)
        # The "error near" check is particularly important for syntax errors.
        if "error report" in response.text or "error near" in response.text.lower():
            error_preview = response.text[:500].replace('\n', ' ').strip()
            logger.error(f"SDSS Error (detected in non-HTML response text): {error_preview}")
            raise ValueError(f"SDSS Error (detected in non-HTML response text): {error_preview}")

        # Handle empty response from SDSS server
        if not response.text.strip():
            logger.warning("SDSS returned an empty response. Returning empty DataFrame.")
            return pd.DataFrame()

        # Attempt to parse the CSV data
        # 'on_bad_lines' helps to skip rows that might be malformed,
        # 'comment=#' handles lines starting with # as comments (SDSS often includes these).
        df = pd.read_csv(StringIO(response.text), on_bad_lines='warn', comment='#') # Changed to 'warn' for bad lines
        
        # Further check if DataFrame is empty after parsing,
        # which can happen if the CSV only contained comments or a header with no data.
        if df.empty and response.text.strip() and len(response.text.strip().splitlines()) <=1 :
             logger.warning("SDSS CSV seems to contain only a header or is empty after parsing. Response text: %s", response.text[:200])
             return pd.DataFrame()

        # --- Convert columns containing "id" (case-insensitive) to string ---
        # This is crucial for compatibility with Streamlit, which can have issues
        # with large integer IDs due to JavaScript's number precision limits.
        if not df.empty:
            converted_cols = []
            for col in df.columns:
                if "id" in col.lower():
                    if df[col].dtype != 'object' and df[col].dtype != 'str':
                        if pd.api.types.is_numeric_dtype(df[col]):
                            converted_cols.append(col)
                            try:
                                df[col] = df[col].astype(str)
                            except Exception as e:
                                logger.warning(f"Failed to convert column '{col}' (dtype: {df[col].dtype}) to string: {e}", exc_info=True)
                        # else:
                            # logger.debug(f"Column '{col}' has 'id' in name but is not numeric (dtype: {df[col].dtype}). No conversion performed.")
            if converted_cols:
                logger.info(f"Converted 'id' columns to string: {', '.join(converted_cols)}")
        # --- End of ID column conversion ---
        
        logger.info(f"Successfully queried SDSS and parsed {len(df)} rows.")
        return df

    except requests.exceptions.Timeout:
        logger.error(f"Timeout error while querying SDSS for: {sql_query[:100]}...", exc_info=True)
        raise TimeoutError("SDSS query timed out after {REQUEST_TIMEOUT} seconds.") # Include timeout value
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}. Response content (preview): {response.text[:500]}", exc_info=True)
        if "error near" in response.text.lower(): # Specific check for SQL syntax errors within HTTP error context
            raise ValueError(f"SDSS SQL Syntax Error (from HTTP response): {response.text[:500].strip()}")
        raise # Re-raise the original HTTPError
    except pd.errors.EmptyDataError:
        logger.error(f"Pandas EmptyDataError: No data or columns to parse in CSV response. Raw response text preview: {response.text[:500]}", exc_info=True)
        return pd.DataFrame() # Return empty DataFrame for this specific pandas error
    except ValueError as ve: # Catch specific ValueErrors raised above
        logger.error(f"ValueError during SDSS query processing: {ve}", exc_info=True)
        raise # Re-raise the ValueError
    except Exception as e:
        # Catch-all for any other unexpected errors
        logger.error(f"Unexpected error occurred for query: {sql_query[:100]}...", exc_info=True)
        logger.debug(f"Raw response text preview (first 1000 chars) for unexpected error:\n{response.text[:1000] if 'response' in locals() else 'Response object not available.'}")
        raise