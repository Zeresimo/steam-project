import pandas as pd
import csv
import os
import glob
from datetime import datetime, timedelta, date

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
        logfile.write(f"{formatted_message}")
        
def log_info(message): # Logging function for general information
    base_path = "Pipeline/logs/"
    os.makedirs(base_path, exist_ok=True)
    filename =  "process_info_log.txt"
    log_path = os.path.join(base_path, filename)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"

    print(formatted_message)
    
    with open(log_path, "a") as logfile:
        logfile.write(f"{formatted_message}")
 
def check_dataframe(df):
    required_columns = ['review', 'appid', 'voted_up']
    if not all(col in df.columns for col in required_columns): # Check if all required columns are present
        missing_cols = [col for col in required_columns if col not in df.columns] # Identify missing columns
        log_error(f"DataFrame is missing required columns: {missing_cols}") # Log error if columns are missing
        return False
    else:
        log_info("DataFrame contains all required columns.")
        return True

def clean_rows(df):
    if check_dataframe(df):
        full_df = df.dropna() # Drop rows with any missing values
        full_df = full_df.drop_duplicates() # Remove duplicate rows
        full_df = full_df[full_df['review'].str.strip().astype(bool)] # Remove rows with empty 'review' field
        log_info(f"Data Frame cleaned. Remaining rows: {len(full_df)}")
        return full_df
    else:
        log_error("DataFrame check failed. Cleaning process aborted.")
        return None
    
def clean_reviews(df):
    if 'review' in df.columns:
        
    else:
        log_error("'review' column not found in DataFrame. Review cleaning aborted.")
        return df


def main():
    latest_file = get_latest_csv()
    if latest_file is None:
        return 1 # Exit if no file found
    log_info(f"Processing file: {latest_file}")
    unclean_df = pd.read_csv(latest_file)
    clean_rows(unclean_df)