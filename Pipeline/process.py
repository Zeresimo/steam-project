import pandas as pd
import csv
import os
import glob
from datetime import datetime, timedelta, date
from utils.logger import log
import re

log_path = "Pipeline/logs/"
filename = "process_log.txt"

def get_latest_csv(base_path = "Pipeline/data/clean"):
    files = glob.glob(os.path.join(base_path, "*.csv")) # Get all CSV files in the directory

    if not files:
        log("No CSV files found in the directory.", level = "INFO", base_path = log_path, filename = filename)
        return None
    
    else:
        latest_file = max(files, key=os.path.getmtime) # Get the most recently modified file
        log(f"Latest CSV file found: {latest_file}", level = "INFO", base_path = log_path, filename = filename)

    return latest_file

def check_dataframe(df):
    required_columns = ['review', 'id', 'voted_up']

    if not all(col in df.columns for col in required_columns): # Check if all required columns are present
        missing_cols = [col for col in required_columns if col not in df.columns] # Identify missing columns
        log(f"Data Frame is missing required columns: {missing_cols}", level = "ERROR", base_path = log_path, filename = filename) # Log error if columns are missing
        return False
    
    else:
        log("Data Frame contains all required columns.", level = "INFO", base_path = log_path, filename = filename)
        return True

def clean_rows(df):
    if check_dataframe(df):
        full_df = df.dropna(subset = ['review']) # Drop rows with any missing values
        full_df = full_df.drop_duplicates() # Remove duplicate rows
        full_df = full_df[full_df['review'].str.strip().astype(bool)] # Remove rows with empty 'review' field
        log(f"Data Frame cleaned. Remaining rows: {len(full_df)}", level = "INFO", base_path = log_path, filename = filename)
        return full_df
    
    else:
        log("Data Frame check failed. Cleaning process aborted.", level = "ERROR", base_path = log_path, filename = filename)
        return None
    
def clean_reviews(text):
    if not isinstance(text, str):
        log("Clean review: Input is not a string.", level = "INFO", base_path = log_path, filename = filename) # Log info if input is not a string as it's not really an error
        return None
    
    else:
        text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"') # Standardize quotes
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs

        text = re.sub(r"[^A-Za-z0-9\s']+", '', text) # Remove special characters

        text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace

        return text.lower() # Convert to lowercase

def main():
    os.makedirs("Pipeline/data/processed", exist_ok=True) # Ensure processed directory exists
    latest_file = get_latest_csv()

    if latest_file is None:
        log("No CSV files to process. Exiting.", level = "ERROR", base_path = log_path, filename = filename)
        return 1 # Exit if no file found
    
    cleaned_file_path = os.path.join("Pipeline/data/processed", f"cleaned_{os.path.basename(latest_file)}")
    
    log(f"Processing file: {latest_file}", level = "INFO", base_path = log_path, filename = filename)

    try:
        unclean_df = pd.read_csv(latest_file)
        log("CSV file read successfully.", level = "INFO", base_path = log_path, filename = filename)
    except Exception as e:
        log(f"Error reading CSV file: {e}", level = "ERROR", base_path = log_path, filename = filename)
        return 1 # Exit if reading fails

    temp_df = clean_rows(unclean_df)

    if temp_df is None:
        log("Data cleaning failed. Exiting process.", level = "ERROR", base_path = log_path, filename = filename)
        return 1 # Exit if cleaning failed
    
    try:
        log("Starting review text cleaning.", level = "INFO", base_path = log_path, filename = filename)
        temp_df['review'] = temp_df['review'].apply(clean_reviews)
        log("Review text cleaning completed.", level = "INFO", base_path = log_path, filename = filename)
        temp_df.to_csv(cleaned_file_path, index=False)
        log(f"Cleaned data saved to: {cleaned_file_path}", level = "INFO", base_path = log_path, filename = filename)

    except Exception as e:
        log(f"Error during review text cleaning: {e}", level = "ERROR", base_path = log_path, filename = filename)
        return 1 # Exit if cleaning fails

    return 0

if __name__ == "__main__":
    main()