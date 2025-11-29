import os
from datetime import datetime
import re
import numpy as np

def log(message, level="INFO",
        base_path = "Pipeline/logs/",
        filename =  "error_log.txt"): # Error logging function for possible debugging
    
    os.makedirs(base_path, exist_ok=True)
    log_path = os.path.join(base_path, filename)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {level}: {message}"
    
    with open(log_path, "a") as logfile:
        #print(f"writing to log file at: {log_path}") # Debugging line to verify log file path
        logfile.write(formatted_message + "\n")
        
# ==================== Validation Helpers ==================== #

def validate_selected_game(selected_game):
    """Return True if the selected game contains a valid game ID"""
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
    """Return True if the author field is a valid dict with required keys and types"""
    author_checker = ['playtime_forever', 'num_reviews']

    if not isinstance(author, dict):
        return False
    
    if not all(key in author for key in author_checker):
        return False
    
    if not isinstance(author['playtime_forever'], (int,float)) or not isinstance(author['num_reviews'], (int, float)):
        return False
    
    return True

def validate_single_review(review):
    """Return True if the review object has required keys, correct types, and a valid author."""
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
    """Return True if the entire page from Steam API has valid structure, success=1, and valid reviews."""
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
    """Return True if the cursor value returned by Steam is valid."""

    if not isinstance(cursor, str):
        return False
    
    if cursor == "":
        return False
    
    return True
    
def validate_page_signature(reviews):
    """Return a signature tuple representing the page content for duplicate detection."""
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
    """Return True if the DataFrame contains all required columns."""
    required_columns = ['game_name','review', 'appid', 'voted_up', 'timestamp_created', 'playtime_forever', 'num_reviews']

    if not all(column in df.columns for column in required_columns):
        return False
    return True

def validate_review_text(text):
    if not isinstance(text, str):
        return False
    
    stripped = text.strip()
    if stripped == "":
        return False
    
    if not any(char.isalnum() for char in stripped):
        return False
    
    return True

def validate_column_type(df, column_name, allowed_types):
    if column_name not in df.columns:
        return False
    for value in df[column_name]:
        if not isinstance(value, allowed_types):
            return False
    return True
# ==================== Cleaning Helpers ==================== #

def clean_text(text):
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"') # Standardize quotes
        
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs

    text = re.sub(r"[^A-Za-z0-9\s']+", '', text) # Remove special characters

    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace

    return text.lower() # Convert to lowercase

# ==================== General Helpers ==================== #
def ensure_directory(paths):
    """Ensure all directory paths in the provided list exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)

def sanitize_filename(name):
    """
    Convert any string into a safe filename by:
    - replacing non-alphanumeric characters with '_'
    - collapsing multiple underscores into one
    - stripping leading/trailing underscores
    """
    # Replace any character not A-Z, a-z, 0-9 with '_'
    safe = re.sub(r'[^A-Za-z0-9]+', '_', name)

    # Remove multiple consecutive underscores
    safe = re.sub(r'_+', '_', safe)

    # Strip leading/trailing underscores
    safe = safe.strip('_')

    return safe

def save_plot(plt, title, game_name, LOG, file_path = "../steam-project/EDA/plots/"):
    safe_game_name = sanitize_filename(game_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"{title}_{safe_game_name}_{timestamp}.png"
    file = os.path.join(file_path, filename)
    try:
        plt.savefig(file, dpi=300, format='png')
        log(f"Plot'{title}' saved in {file_path} as {filename}", level="INFO", **LOG)
    except Exception as e:
        log(f"Error saving {title}: {e}", level="ERROR", **LOG)

    plt.clf()