import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import requests
from io import StringIO
from unittest.mock import MagicMock

# Import the function to test
from AstroQueryGPT import sdss_db # Assuming sdss_db.py is in AstroQueryGPT directory

# Mock config if sdss_db uses it, though the provided snippet doesn't show direct config use.
# from AstroQueryGPT import config # Example if needed

@pytest.fixture
def mock_requests_get(mocker):
    """Fixture to mock requests.get."""
    return mocker.patch("requests.get")

# Test for successful query
def test_query_sdss_success(mock_requests_get, caplog):
    """Test successful SDSS query and DataFrame processing, including ID conversion."""
    csv_data = (
        "objID,ra,dec,g,r,i,z,isGalaxy,specObjID,sourceID\n"
        "123,150.0,2.0,18.0,17.5,17.0,16.8,1,456,789\n"
        "124,150.1,2.1,19.0,18.5,18.0,17.8,0,457,790\n"
        "125,150.2,2.2,20.0,19.5,19.0,18.8,1,NULL,791\n" # Test NULL 'id' like field
        "126,150.3,2.3,21.0,20.5,20.0,19.8,1,458,non_numeric_id\n" # Test non-numeric 'id' like field
    )
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = csv_data
    mock_response.headers = {"Content-Type": "text/plain"} # Or "text/csv"
    mock_requests_get.return_value = mock_response

    expected_df_data = {
        "objID": ["123", "124", "125", "126"], # Converted to string
        "ra": [150.0, 150.1, 150.2, 150.3],
        "dec": [2.0, 2.1, 2.2, 2.3],
        "g": [18.0, 19.0, 20.0, 21.0],
        "r": [17.5, 18.5, 19.5, 20.5],
        "i": [17.0, 18.0, 19.0, 20.0],
        "z": [16.8, 17.8, 18.8, 19.8],
        "isGalaxy": [1,0,1,1], # Stays numeric
        "specObjID": ["456", "457", "nan", "458"], # Converted to string, NULL becomes 'nan'
        "sourceID": ["789","790","791","non_numeric_id"] # sourceID is numeric for some, string for one, should all become string
    }
    expected_df = pd.DataFrame(expected_df_data)
    
    # Ensure correct dtypes for non-id columns before comparison, as read_csv might infer int
    expected_df = expected_df.astype({
        'ra': float, 'dec': float, 'g': float, 'r': float, 'i': float, 'z': float,
        'isGalaxy': int # Pandas might make it int64, ensure consistency
    })


    result_df = sdss_db.query_sdss("SELECT objID,ra,dec,g,r,i,z,isGalaxy,specObjID,sourceID FROM PhotoObj")

    # Check logs for conversion messages
    assert "Converting column 'objID'" in caplog.text
    assert "Converting column 'specObjID'" in caplog.text
    assert "Converting column 'sourceID'" in caplog.text
    
    # Verify dtypes for ID columns
    assert result_df["objID"].dtype == "object" # Pandas uses 'object' for strings
    assert result_df["specObjID"].dtype == "object"
    assert result_df["sourceID"].dtype == "object"
    
    # Verify dtypes for non-ID columns (example)
    assert pd.api.types.is_numeric_dtype(result_df["ra"])
    assert pd.api.types.is_numeric_dtype(result_df["isGalaxy"])

    # Compare dataframes
    # Need to be careful about dtypes if read_csv infers differently than DataFrame constructor
    # For robust comparison, sort columns and reset index.
    expected_df = expected_df.reindex(sorted(expected_df.columns), axis=1)
    result_df = result_df.reindex(sorted(result_df.columns), axis=1)
    
    assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True), check_dtype=True)


# Test for SDSS API error (HTML response)
def test_query_sdss_api_error_html(mock_requests_get, caplog):
    """Test handling of an SDSS API error returned as HTML."""
    html_error_content = "<!DOCTYPE html><html><body>Error report: error near 'SELECTX'</body></html>"
    mock_response = MagicMock()
    mock_response.status_code = 200 # API might return 200 but with an HTML error page
    mock_response.text = html_error_content
    mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
    mock_requests_get.return_value = mock_response

    with pytest.raises(ValueError, match="SDSS SQL Error (detected in HTML response)"):
        sdss_db.query_sdss("SELECTX ra FROM PhotoObj") # Malformed query
    assert "SDSS SQL Error (detected in HTML response)" in caplog.text

def test_query_sdss_api_error_non_html_text(mock_requests_get, caplog):
    """Test handling of an SDSS API error returned as plain text."""
    text_error_content = "Error: Query failed due to error near keyword BLAH."
    mock_response = MagicMock()
    mock_response.status_code = 200 
    mock_response.text = text_error_content
    mock_response.headers = {"Content-Type": "text/plain"}
    mock_requests_get.return_value = mock_response

    with pytest.raises(ValueError, match="SDSS Error (detected in non-HTML response text)"):
        sdss_db.query_sdss("SELECT ra FROM PhotoObj WHERE BLAH")
    assert "SDSS Error (detected in non-HTML response text)" in caplog.text


# Test for HTTP error
def test_query_sdss_http_error(mock_requests_get, caplog):
    """Test handling of a generic HTTP error."""
    mock_requests_get.side_effect = requests.exceptions.HTTPError("500 Server Error")
    
    with pytest.raises(requests.exceptions.HTTPError):
        sdss_db.query_sdss("SELECT ra FROM PhotoObj")
    assert "HTTP error occurred: 500 Server Error" in caplog.text

def test_query_sdss_http_error_with_sql_message_in_body(mock_requests_get, caplog):
    """Test HTTP error that also contains 'error near' in its body."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 400 # e.g. Bad Request
    mock_response.text = "HTTP Error: Syntax error near 'FROMM'"
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Client Error", response=mock_response)
    mock_requests_get.return_value = mock_response
    
    with pytest.raises(ValueError, match="SDSS SQL Syntax Error (from HTTP response)"):
        sdss_db.query_sdss("SELECT ra FROMM PhotoObj")
    assert "HTTP error occurred" in caplog.text # General HTTP error logged
    assert "SDSS SQL Syntax Error (from HTTP response)" in caplog.text # More specific error logged before raising


# Test for timeout
def test_query_sdss_timeout(mock_requests_get, caplog):
    """Test handling of a request timeout."""
    mock_requests_get.side_effect = requests.exceptions.Timeout("Connection timed out")

    with pytest.raises(TimeoutError, match="SDSS query timed out"):
        sdss_db.query_sdss("SELECT ra FROM PhotoObj")
    assert "Timeout error while querying SDSS" in caplog.text


# Test for empty response
def test_query_sdss_empty_response_text(mock_requests_get, caplog):
    """Test handling of an empty string response from SDSS."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "" # Empty response text
    mock_response.headers = {"Content-Type": "text/plain"}
    mock_requests_get.return_value = mock_response

    result_df = sdss_db.query_sdss("SELECT ra FROM PhotoObj")
    assert result_df.empty
    assert "SDSS returned an empty response" in caplog.text

def test_query_sdss_empty_csv_after_parsing(mock_requests_get, caplog):
    """Test when CSV parsing results in an empty DataFrame (e.g. only header or comments)."""
    csv_content_only_header = "ra,dec\n" # Header but no data
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = csv_content_only_header
    mock_response.headers = {"Content-Type": "text/plain"}
    mock_requests_get.return_value = mock_response
    
    result_df = sdss_db.query_sdss("SELECT ra, dec FROM PhotoObj")
    assert result_df.empty
    assert "SDSS CSV seems to contain only a header or is empty after parsing" in caplog.text

def test_query_sdss_pandas_empty_data_error(mock_requests_get, caplog):
    """Test handling of pandas EmptyDataError if read_csv gets truly empty parsable content."""
    # This can happen if the content is just comments, or if on_bad_lines='skip' removes everything.
    # Forcing this error directly via read_csv mock is more reliable.
    mocker.patch('pandas.read_csv', side_effect=pd.errors.EmptyDataError("No columns to parse from file"))
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "#comment only" # read_csv would raise EmptyDataError
    mock_response.headers = {"Content-Type": "text/plain"}
    mock_requests_get.return_value = mock_response

    result_df = sdss_db.query_sdss("SELECT ra FROM PhotoObj")
    assert result_df.empty
    assert "Pandas EmptyDataError: No data or columns to parse" in caplog.text


def test_query_sdss_unexpected_exception(mock_requests_get, caplog):
    """Test handling of an unexpected error during processing."""
    mock_requests_get.side_effect = Exception("Some other weird error")

    with pytest.raises(Exception, match="Some other weird error"):
        sdss_db.query_sdss("SELECT ra FROM PhotoObj")
    assert "Unexpected error occurred for query" in caplog.text

# Test for ID column conversion with various scenarios
@pytest.mark.parametrize("col_name, col_data, initial_dtype, expected_dtype_after_conversion, should_convert", [
    ("objID", [1, 2, 3], "int64", "object", True),            # Numeric ID col
    ("source_id", [10, 20], "int64", "object", True),        # Numeric ID col (lowercase with underscore)
    ("customIdField", [30, 40], "float64", "object", True),  # Float ID col
    ("id", ["a", "b", "c"], "object", "object", False),      # Already string/object ID col
    ("identifier", [1,2,3], "int64", "object", True), # another variant of id
    ("value", [1, 2, 3], "int64", "int64", False),         # Numeric, no 'id' in name
    ("name", ["x", "y"], "object", "object", False),         # String, no 'id' in name
    ("photoIdentification", [1,2], "int64", "object", True) # Camel case 'id'
])
def test_id_column_conversions(mock_requests_get, caplog, col_name, col_data, initial_dtype, expected_dtype_after_conversion, should_convert):
    """Parameterized test for various ID column conversion scenarios."""
    
    # Construct CSV data string with a single column
    csv_data_str = f"{col_name}\n" + "\n".join(map(str, col_data))
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = csv_data_str
    mock_response.headers = {"Content-Type": "text/plain"}
    mock_requests_get.return_value = mock_response

    df = sdss_db.query_sdss(f"SELECT {col_name} FROM DummyTable")

    assert df[col_name].dtype == expected_dtype_after_conversion
    if should_convert:
        assert f"Converting column '{col_name}'" in caplog.text
    else:
        assert f"Converting column '{col_name}'" not in caplog.text

def test_id_conversion_failure_warning(mock_requests_get, caplog):
    """Test that a warning is logged if astype(str) fails for an ID column."""
    # This is hard to trigger reliably with standard numeric types.
    # We can mock the astype call on a specific series to simulate failure.
    
    csv_data = "objID\n123\n456"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = csv_data
    mock_response.headers = {"Content-Type": "text/plain"}
    mock_requests_get.return_value = mock_response

    # Mock pandas Series.astype to raise an error for the 'objID' column
    # This requires knowing that pd.read_csv will produce a Series that we can patch.
    
    original_series_astype = pd.Series.astype
    def mock_astype(self, dtype):
        if self.name == "objID" and dtype == str:
            raise TypeError("Simulated astype(str) failure")
        return original_series_astype(self, dtype)

    with patch.object(pd.Series, 'astype', mock_astype):
        df = sdss_db.query_sdss("SELECT objID FROM Table")
    
    assert "Failed to convert column 'objID'" in caplog.text
    assert "Simulated astype(str) failure" in caplog.text
    # The column type might remain numeric if conversion failed.
    # Depending on when the exception is caught, it might still be numeric or object if partially converted.
    # The key is that the warning was logged.
    assert df["objID"].dtype != "object" # Or check it's the original numeric type
```
