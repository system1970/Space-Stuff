from io import StringIO
import pandas as pd
import requests
import time

import config # Import config if you need any constants from it here

def query_sdss(sql_query: str) -> pd.DataFrame:
    """
    Executes an SQL query against the SDSS SkyServer DR16 and returns results as a Pandas DataFrame.
    Converts columns with "id" in their name (case-insensitive) to strings to prevent
    JavaScript number precision issues in Streamlit.
    """
    url = "https://skyserver.sdss.org/dr16/en/tools/search/x_sql.aspx"
    params = {
        "cmd": sql_query,
        "format": "csv"
    }
    
    print(f"Executing SQL against SDSS SkyServer: {sql_query[:150]}...")
    
    try:
        response = requests.get(url, params=params, timeout=60) 
        response.raise_for_status()
        
        if response.text.strip().startswith("<!DOCTYPE html>") or "Error report" in response.text or "error near" in response.text.lower():
            error_preview = response.text[:500].replace('\n', ' ').strip()
            if "error near" in error_preview.lower():
                 raise ValueError(f"SDSS SQL Error: {error_preview}")
            raise ValueError(f"SDSS returned an HTML error page, not CSV. Preview: {error_preview}")

        if not response.text.strip():
            print("Warning: SDSS returned an empty response. Returning empty DataFrame.")
            return pd.DataFrame()

        df = pd.read_csv(StringIO(response.text), on_bad_lines='skip', comment='#')
        
        if df.empty and response.text.strip() and len(response.text.strip().splitlines()) <=1 :
             print("Warning: SDSS CSV seems to contain only a header or is empty after parsing. Returning empty DataFrame.")
             return pd.DataFrame()

        # --- Convert columns containing "id" (case-insensitive) to string ---
        if not df.empty:
            for col in df.columns:
                if "id" in col.lower(): # Case-insensitive check for "id"
                    # Check if the column isn't already object/string and seems numeric before converting
                    if df[col].dtype != 'object' and df[col].dtype != 'str':
                        try:
                            # Only attempt conversion if it's likely a numeric type that could cause issues
                            # This check helps avoid trying to convert actual string columns that happen to have "id"
                            if pd.api.types.is_numeric_dtype(df[col]):
                                print(f"Converting column '{col}' to string due to 'id' in name and numeric type.")
                                # astype(str) handles NaN/None correctly by converting them to 'nan' or 'None' strings.
                                # If you want empty strings for NaNs: df[col].fillna('').astype(str)
                                df[col] = df[col].astype(str)
                        except Exception as e:
                            print(f"Warning: Could not convert column '{col}' to string: {e}")
        # --- End of ID column conversion ---
        
        return df

    except requests.exceptions.Timeout:
        print(f"Timeout error while querying SDSS for: {sql_query[:100]}...")
        raise TimeoutError("SDSS query timed out.")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response content (preview): {response.text[:500]}")
        if "error near" in response.text.lower():
            raise ValueError(f"SDSS SQL Syntax Error (from HTTP response): {response.text[:500].strip()}")
        raise
    except pd.errors.EmptyDataError:
        print("Pandas EmptyDataError: No data or columns to parse in CSV response. Returning empty DataFrame.")
        print(f"Raw response text preview: {response.text[:500]}")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Could not parse CSV or other error occurred for query: {sql_query[:100]}...")
        print(f"   Error type: {type(e).__name__}, Message: {e}")
        print(f"   Raw response text preview (first 1000 chars):\n{response.text[:1000]}")
        raise