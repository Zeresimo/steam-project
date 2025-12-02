"""
Utility functions for logging, validation, cleaning, and file handling.

This module is shared across:
- Pipeline (data fetching and saving)
- ML (training and prediction)
- EDA (plot generation)

Centralizing these helpers keeps behavior consistent across the project.
"""

import os
from datetime import datetime
import re
import numpy as np
import glob

from paths import EDA_PLOTS


def log(message, level="INFO", base_path="Pipeline/logs/", filename="error_log.txt"):
    """
    Write a timestamped message to a log file.

    Args:
        message: Text to record in the log.
        level: Message type (INFO, WARNING, ERROR).
        base_path: Folder where the log file is saved.
        filename: Log filename.

    NOTES:
    - Automatically creates the log folder if missing.
    - Logging never raises errors; it silently handles issues.
    - Used across Pipeline, ML, and EDA for debugging.
    """
    os.makedirs(base_path, exist_ok=True)
    log_path = os.path.join(base_path, filename)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {level}: {message}"
    
    with open(log_path, "a") as logfile:
        logfile.write(formatted_message + "\n")
        
def error(msg, LOG):
    """
    Log an error and return None.
    """
    log(msg, level="ERROR", **LOG)
    return None

def error2(msg, LOG):
    """
    Log an error and return (None, None).
    """
    log(msg, level="ERROR", **LOG)
    return None, None

def info(msg, LOG):
    """
    Log an informational message.
    """
    log(msg, level="INFO", **LOG)


# ==================== Validation Helpers ==================== #

def validate_selected_game(selected_game):
    """
    Validate that a selected game dict contains a numeric Steam app id.

    Args:
        selected_game: Dictionary returned from Steam search.

    Returns:
        True if valid, False otherwise.

    NOTES:
    - Ensures Pipeline never saves reviews with invalid IDs.
    - Prevents crashes later when using the ID in URLs.
    """
    if not isinstance(selected_game, dict): # Check if selected_game is a dictionary
        return False

    if 'id' not in selected_game: # Check for game ID in selected_game
        return False

    if selected_game['id'] is False: # Check for False game ID
        return False
    
    if not isinstance(selected_game['id'], (str, int)): # Check for valid game ID type
        return False
    
    if not str(selected_game['id']).isdigit(): # Check if game ID is numeric
        return False

    return True

def validate_review_author(author):
    """
    Validate the 'author' field of a Steam review.

    Args:
        author: Dict containing playtime and num_reviews.

    Returns:
        True if required fields exist and have correct types.

    NOTES:
    - Ensures author['playtime_forever'] and ['num_reviews'] exist and are numeric.
    """
    author_checker = ['playtime_forever', 'num_reviews']

    if not isinstance(author, dict):
        return False
    
    if not all(key in author for key in author_checker):
        return False
    
    if not isinstance(author['playtime_forever'], (int,float)) or not isinstance(author['num_reviews'], (int, float)):
        return False
    
    return True

def validate_single_review(review):
    """
    Validate the structure of a single Steam review.

    Args:
        review: Dictionary containing review fields.

    Returns:
        True if all required fields are present and valid.

    NOTES:
    - Prevents malformed API data from being saved.
    - Protects ML and EDA steps from broken rows.
    """
    if not isinstance(review, dict): # Invalid review format
        return False
    
    item_checker = ['review', 'voted_up', 'timestamp_created', 'author'] # Expected keys in each review

    if not all(key in review for key in item_checker): # Check for expected keys
        return False
    
    if not isinstance(review['voted_up'], bool): # Invalid voted_up format
        return False
    
    if not isinstance(review['timestamp_created'], int): # Invalid timestamp format
        return False
    
    if not isinstance(review['review'], str): # Invalid review text format
        return False

    if not validate_review_author(review['author']): # Invalid author format
        return False
    
    return True

def validate_review_page(page):
    """
    Validate the structure of a complete Steam API review page.

    Args:
        page: Parsed JSON object from Steam.

    Returns:
        True if the page has valid structure.

    NOTES:
    - Ensures success == 1, cursor exists, and reviews list is valid.
    - Prevents Pipeline crashes during pagination.
    """
    if not isinstance(page, dict):
        return False
    
    if "success" not in page or page['success'] != 1:
        return False
    
    if "reviews" not in page or not isinstance(page['reviews'], list):
        return False
    
    if "cursor" not in page:
        return False

    if not validate_cursor(page['cursor']):
        return False
    
    for review in page['reviews']:
        if not validate_single_review(review):
            return False

    return True

def validate_cursor(cursor):
    """
    Check whether the Steam cursor is a non-empty string.

    Args:
        cursor: Cursor value returned in the API response.

    Returns:
        True if valid, False otherwise.

    NOTES:
    - Cursor controls pagination for review fetching.
    - Empty or invalid cursors break pagination loops.
    """
    if not isinstance(cursor, str):
        return False
    
    if cursor == "":
        return False
    
    return True
    
def validate_page_signature(reviews):
    """
    Create a signature from timestamps to detect duplicate review pages.

    Args:
        reviews: List of review dictionaries.

    Returns:
        A tuple of sorted timestamps, or None if invalid.

    NOTES:
    - Helps prevent infinite loops during pagination.
    """
    if not isinstance(reviews, list):
        return None
    
    for review in reviews:
        if not isinstance(review, dict):
            return None
        if "timestamp_created" not in review:
            return None
    
    signature = tuple(sorted(review['timestamp_created'] for review in reviews))
    return signature

def validate_dataframe_columns(df):
    """
    Check if the DataFrame contains all required columns.

    Args:
        df: Pandas DataFrame.

    Returns:
        True if all expected columns exist.

    NOTES:
    - Used in process.py to ensure cleaned data has full schema.
    """
    required_columns = ['game_name','review', 'appid', 'voted_up', 'timestamp_created', 'playtime_forever', 'num_reviews']

    if not all(column in df.columns for column in required_columns):
        return False
    return True

def validate_review_text(text):
    """
    Check if a review text is non-empty and contains at least one alphanumeric character.

    Args:
        text: Raw review string.

    Returns:
        True if valid.

    NOTES:
    - Prevents saving empty or meaningless reviews.
    """
    if not isinstance(text, str):
        return False
    
    stripped = text.strip()
    if stripped == "":
        return False
    
    if not any(char.isalnum() for char in stripped):
        return False
    
    return True

def validate_column_type(df, column_name, allowed_types):
    """
    Ensure all values in a column match one of the allowed types.

    Args:
        df: Pandas DataFrame.
        column_name: Column to check.
        allowed_types: Tuple of allowed data types.

    Returns:
        True if column values match allowed types.

    NOTES:
    - Helps avoid errors during EDA and ML training.
    """
    if column_name not in df.columns:
        return False
    for value in df[column_name]:
        if not isinstance(value, allowed_types):
            return False
    return True
# ==================== Cleaning Helpers ==================== #

def clean_text(text):
    """
    Clean and normalize review text.

    Steps:
        - Remove URLs
        - Remove special characters
        - Collapse whitespace
        - Lowercase output

    Args:
        text: Original review string.

    Returns:
        Cleaned text string.

    NOTES:
    - Ensures consistent input to the ML model.
    - Prevents vectorizer mismatch caused by stray characters.
    """
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"') # Standardize quotes
        
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs

    text = re.sub(r"[^A-Za-z0-9\s']+", '', text) # Remove special characters

    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace

    return text.lower() # Convert to lowercase

# ==================== General Helpers ==================== #
def get_latest_csv(LOG, base_path="Pipeline/data/"):
    """
    Return the most recently modified CSV file in a directory.

    Args:
        LOG: Logging configuration dictionary.
        base_path: Directory to search for CSV files.

    Returns:
        Path to the latest CSV file, or None if none found.

    NOTES:
    - Used in process.py, model.py, and EDA to always grab the newest dataset.
    """
    files = glob.glob(os.path.join(base_path, "*.csv")) # Get all CSV files in the directory

    if not files:
        log("No CSV files found in the directory.", level = "INFO", **LOG)
        return None
    
    else:
        latest_file = max(files, key=os.path.getmtime) # Get the most recently modified file
        log(f"Latest CSV file found: {latest_file}", level = "INFO", **LOG)

    return latest_file

def ensure_directory(paths):
    """
    Ensure each directory in a list exists.

    Args:
        paths: Iterable of directory paths.

    NOTES:
    - Creates directories if they don't exist.
    - Safe to call repeatedly (no errors if already exists).
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)

def sanitize_filename(name):
    """
    Convert any string into a safe filename.

    Args:
        name: Original game name or string.

    Returns:
        A safe version suitable for file saving.

    NOTES:
    - Removes special characters.
    - Prevents file system errors on Windows/macOS/Linux.
    """
    # Replace any character not A-Z, a-z, 0-9 with '_'
    safe = re.sub(r'[^A-Za-z0-9]+', '_', name)

    # Remove multiple consecutive underscores
    safe = re.sub(r'_+', '_', safe)

    # Strip leading/trailing underscores
    safe = safe.strip('_')

    return safe

def save_plot(plt, title, game_name, LOG, file_path=EDA_PLOTS):
    """
    Save a Matplotlib plot with a timestamped filename.

    Args:
        plt: Matplotlib pyplot module.
        title: Short identifier for the plot type.
        game_name: Used to generate a safe filename.
        LOG: Logging configuration dictionary.
        file_path: Folder to save the plot.

    NOTES:
    - Automatically clears the plot after saving.
    - Uses sanitize_filename() for safety.
    - Used by all EDA functions.
    """
    safe_game_name = sanitize_filename(game_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"{title}_{safe_game_name}_{timestamp}.png"
    file = os.path.join(file_path, filename)
    try:
        plt.savefig(file, dpi=300, format='png')
        log(f"Plot' {title}' saved in {file_path} as {filename}", level="INFO", **LOG)
    except Exception as e:
        log(f"Error saving {title}: {e}", level="ERROR", **LOG)

    plt.clf()