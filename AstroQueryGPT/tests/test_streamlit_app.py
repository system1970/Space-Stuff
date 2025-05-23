import pytest
import pandas as pd

# Import the function to test
# Assuming streamlit_app.py is in AstroQueryGPT directory
from AstroQueryGPT import streamlit_app 

# Mock the logger if verify_data_structure uses it directly for printing/logging
@pytest.fixture(autouse=True)
def mock_app_logger(mocker):
    # If streamlit_app.py has `logger = logging.getLogger(__name__)`
    # And verify_data_structure uses `logger.info`, `logger.warning` etc.
    # This mock will capture those calls.
    # If it uses print(), then caplog fixture (from pytest) would be used in tests.
    # The provided streamlit_app.py uses print(), so caplog will be used.
    pass


# Tests for verify_data_structure
def test_verify_data_structure_empty_dataframe(caplog):
    """Test with an empty DataFrame."""
    df = pd.DataFrame()
    assert streamlit_app.verify_data_structure(df) is False
    assert "DataFrame is empty" in caplog.text

def test_verify_data_structure_no_columns(caplog):
    """Test with a DataFrame that has rows but no columns."""
    df = pd.DataFrame(index=[0, 1, 2])
    assert streamlit_app.verify_data_structure(df) is False
    assert "DataFrame has no columns" in caplog.text

def test_verify_data_structure_single_long_string(caplog):
    """Test with a DataFrame that represents a single long string."""
    # Single cell, long string, few spaces
    data = {"col1": ["a"*250]} 
    df = pd.DataFrame(data)
    assert streamlit_app.verify_data_structure(df) is False
    assert "looks like a single long string" in caplog.text

    # Single cell, long string with some spaces (should still fail if spaces are few)
    data = {"col1": ["this is a very long string " * 20]} # count(' ') would be 5*20 = 100, length > 200
    df = pd.DataFrame(data)
    # This case might pass or fail depending on the exact heuristics.
    # The current heuristic is `count(' ') < 5` for a string > 200. This will pass.
    # To make it fail, it needs fewer spaces.
    # Let's test a clear fail:
    data_fail = {"col1": ["longstring" * 50]} # len=500, spaces=0
    df_fail = pd.DataFrame(data_fail)
    assert streamlit_app.verify_data_structure(df_fail) is False
    assert "looks like a single long string" in caplog.text


def test_verify_data_structure_single_cell_error_phrase_short(caplog):
    """Test with a DataFrame containing a short error phrase in the first cell."""
    error_phrases = ["error", "failed", "unable", "cannot", "syntax", "invalid", "incorrect"]
    for phrase in error_phrases:
        caplog.clear() # Clear logs for each phrase
        data = {"col1": [f"An {phrase} occurred during execution."]} # Length < 150
        df = pd.DataFrame(data)
        assert streamlit_app.verify_data_structure(df) is False, f"Failed for phrase: {phrase}"
        assert f"first cell contains potential error message: An {phrase}" in caplog.text or \
               f"first cell contains potential error: An {phrase}" in caplog.text # Match previous log format too

def test_verify_data_structure_single_cell_error_phrase_long_but_not_long_string_type(caplog):
    """Test error phrase in single cell that is long but doesn't trigger 'single long string'."""
    # This tests if the error phrase check is independent of the "single long string" check
    # if the string has enough spaces not to be a "long string" but contains an error.
    data = {"col1": ["This query has failed with a syntax error and this message is quite long but has spaces."]} # len < 150 is part of error phrase heuristic
    df = pd.DataFrame(data)
    # This should NOT be caught by the error phrase detector if length > 150 for that check
    # The current code is `len(first_cell_value) < 150` for error phrase check.
    # So, this long error message will NOT be caught by that specific error phrase check.
    # It will be considered a valid structure.
    assert streamlit_app.verify_data_structure(df) is True
    assert "first cell contains potential error" not in caplog.text # Should not trigger this
    assert "Basic structure appears valid" in caplog.text


def test_verify_data_structure_valid_dataframe_multiple_rows_cols(caplog):
    """Test with a typical valid DataFrame (multiple rows and columns)."""
    data = {"colA": [1, 2, 3], "colB": ["a", "b", "c"]}
    df = pd.DataFrame(data)
    assert streamlit_app.verify_data_structure(df) is True
    assert "Basic structure appears valid" in caplog.text

def test_verify_data_structure_valid_dataframe_single_row_multiple_cols(caplog):
    """Test with a valid DataFrame (single row, multiple columns)."""
    data = {"colA": [1], "colB": ["a"]}
    df = pd.DataFrame(data)
    assert streamlit_app.verify_data_structure(df) is True
    assert "Basic structure appears valid" in caplog.text

def test_verify_data_structure_valid_dataframe_multiple_rows_single_col(caplog):
    """Test with a valid DataFrame (multiple rows, single column)."""
    data = {"colA": [1, 2, 3]}
    df = pd.DataFrame(data)
    assert streamlit_app.verify_data_structure(df) is True
    assert "Basic structure appears valid" in caplog.text
    
def test_verify_data_structure_valid_dataframe_single_cell_short_non_error(caplog):
    """Test with a valid DataFrame (single cell, short, non-error content)."""
    data = {"col1": ["OK"]}
    df = pd.DataFrame(data)
    assert streamlit_app.verify_data_structure(df) is True
    assert "Basic structure appears valid" in caplog.text

def test_verify_data_structure_single_cell_long_with_sufficient_spaces(caplog):
    """Test a single cell, long string, but with enough spaces to be valid."""
    # String > 200 chars, but space count >= 5
    data = {"col1": ["This is a single cell with a reasonably long description that has plenty of spaces." * 5]}
    df = pd.DataFrame(data)
    assert streamlit_app.verify_data_structure(df) is True
    assert "looks like a single long string" not in caplog.text
    assert "Basic structure appears valid" in caplog.text

def test_verify_data_structure_one_row_one_col_numeric(caplog):
    """Test one row, one col, numeric data - should be valid."""
    df = pd.DataFrame({'value': [12345]})
    assert streamlit_app.verify_data_structure(df) is True
    assert "Basic structure appears valid" in caplog.text

def test_verify_data_structure_one_row_one_col_looks_like_error_but_long(caplog):
    """Test one row, one col, looks like an error message but is too long for the error phrase check."""
    df = pd.DataFrame({'error_message': ["This is an error message that is very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long."]}) # > 150 chars
    # This should NOT trigger the "potential error" path because of the length check (len < 150)
    # And it should NOT trigger the "single long string" if it has enough spaces (which it does)
    assert streamlit_app.verify_data_structure(df) is True
    assert "first cell contains potential error" not in caplog.text
    assert "Basic structure appears valid" in caplog.text

# Example of how you might test the Streamlit UI interaction parts (more complex, needs streamlit-testing-library or similar)
# This is beyond "basic unit tests" for now.
# def test_streamlit_app_initial_state(mock_streamlit_runtime):
#     # Requires a more sophisticated setup like streamlit_test_runner or AppTest
#     # from streamlit.testing.v1 import AppTest
#     # at = AppTest.from_file("AstroQueryGPT/streamlit_app.py").run()
#     # assert at.title[0].value == "🔭 SDSS Agentic RAG SQL Generator"
#     # assert not at.session_state.query_log
    pass
