import pandas as pd
import csv
import os
import glob
from datetime import datetime, timedelta, date
import re

def get_latest_csv(base_path = "Pipeline/data/clean"):
    files = glob.glob(os.path.join(base_path, "*.csv")) # Get all CSV files in the directory

    if not files:
        log_error("No CSV files found in the directory.")
        return None
    
    else:
        latest_file = max(files, key=os.path.getmtime) # Get the most recently modified file
        log_info(f"Latest CSV file found: {latest_file}")

    return latest_file

def log_error(message): # Error logging function for possible debugging
    base_path = "Pipeline/logs/"
    os.makedirs(base_path, exist_ok=True)
    filename =  "process_error_log.txt"
    log_path = os.path.join(base_path, filename)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"

    print(formatted_message)
    
    with open(log_path, "a") as logfile:
        logfile.write(f"{formatted_message}\n")
        
def log_info(message): # Logging function for general information
    base_path = "Pipeline/logs/"
    os.makedirs(base_path, exist_ok=True)
    filename =  "process_info_log.txt"
    log_path = os.path.join(base_path, filename)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"

    print(formatted_message)
    
    with open(log_path, "a") as logfile:
        logfile.write(f"{formatted_message}\n")
 
def check_dataframe(df):
    required_columns = ['review', 'appid', 'voted_up']

    if not all(col in df.columns for col in required_columns): # Check if all required columns are present
        missing_cols = [col for col in required_columns if col not in df.columns] # Identify missing columns
        log_error(f"Data Frame is missing required columns: {missing_cols}") # Log error if columns are missing
        return False
    
    else:
        log_info("Data Frame contains all required columns.")
        return True

def clean_rows(df):
    if check_dataframe(df):
        full_df = df.dropna(subset = ['review']) # Drop rows with any missing values
        full_df = full_df.drop_duplicates() # Remove duplicate rows
        full_df = full_df[full_df['review'].str.strip().astype(bool)] # Remove rows with empty 'review' field
        log_info(f"Data Frame cleaned. Remaining rows: {len(full_df)}")
        return full_df
    
    else:
        log_error("Data Frame check failed. Cleaning process aborted.")
        return None
    
def clean_reviews(text):
    if not isinstance(text, str):
        log_info("Clean review: Input is not a string.") # Log info if input is not a string as it's not really an error
        return None
    
    else:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs

        text = re.sub(r"[^A-Za-z0-9\s']+", '', text) # Remove special characters

        text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace

        return text.lower() # Convert to lowercase

def main():
    os.makedirs("Pipeline/data/processed", exist_ok=True) # Ensure processed directory exists
    latest_file = get_latest_csv()

    if latest_file is None:
        return 1 # Exit if no file found
    
    log_info(f"Processing file: {latest_file}")

    unclean_df = pd.read_csv(latest_file)

    temp_df = clean_rows(unclean_df)

    if temp_df is None:
        return 1 # Exit if cleaning failed
    
    temp_df['review'] = temp_df['review'].apply(clean_reviews)

    cleaned_file_path = os.path.join("Pipeline/data/processed", f"cleaned_{os.path.basename(latest_file)}")

    temp_df.to_csv(cleaned_file_path, index=False)
    log_info(f"Cleaned data saved to: {cleaned_file_path}")