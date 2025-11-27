import pandas as pd
import csv
import os
import glob
from datetime import datetime, timedelta, date
import utils.utils as utils

LOG = { 
    "base_path": "Pipeline/logs/",
    "filename": "process_log.txt"
}

paths = [
    "Pipeline/data/clean",
    "Pipeline/data/processed",
    "Pipeline/logs/"
]

utils.ensure_directory(paths)

def get_latest_csv(base_path = "Pipeline/data/clean"):
    files = glob.glob(os.path.join(base_path, "*.csv")) # Get all CSV files in the directory

    if not files:
        utils.log("No CSV files found in the directory.", level = "INFO", **LOG)
        return None
    
    else:
        latest_file = max(files, key=os.path.getmtime) # Get the most recently modified file
        utils.log(f"Latest CSV file found: {latest_file}", level = "INFO", **LOG)

    return latest_file

def clean_rows(df):
    if not utils.validate_dataframe_columns(df):
        utils.log("Data Frame validation failed during cleaning.", level = "ERROR", **LOG)
        return None
    
    df = df.dropna(subset=['review']) # Drop rows with missing critical fields
    df = df.drop_duplicates() # Remove duplicate rows
    df = df[df['review'].apply(utils.validate_review_text)] # Keep only rows with valid review text
    
    return df
    
def clean_reviews(text):
    if not utils.validate_review_text(text):
        utils.log("Invalid review text encountered during cleaning.", level = "ERROR", **LOG)
        return None

    cleaned = utils.clean_text(text)
    
    if cleaned.strip() == "":
        return None
    
    return cleaned

def main():
    latest_file = get_latest_csv()

    if latest_file is None:
        utils.log("No CSV files to process. Exiting.", level = "ERROR", **LOG)
        return 1 # Exit if no file found
    
    cleaned_file_path = os.path.join("Pipeline/data/processed", f"cleaned_{os.path.basename(latest_file)}")
    
    utils.log(f"Processing file: {latest_file}", level = "INFO", **LOG)

    try:
        unclean_df = pd.read_csv(latest_file)
        utils.log("CSV file read successfully.", level = "INFO", **LOG)
    except Exception as e:
        utils.log(f"Error reading CSV file: {e}", level = "ERROR", **LOG)
        return 1 # Exit if reading fails

    temp_df = clean_rows(unclean_df)

    if temp_df is None:
        utils.log("Data cleaning failed. Exiting process.", level = "ERROR", **LOG)
        return 1 # Exit if cleaning failed
    
    try:
        utils.log("Starting review text cleaning.", level = "INFO", **LOG)

        temp_df['review'] = temp_df['review'].apply(clean_reviews)

        temp_df = temp_df.dropna(subset=['review']) # Drop rows where review text became None after cleaning

        utils.log("Review text cleaning completed.", level = "INFO", **LOG)

        if temp_df.empty:
            utils.log("No valid reviews remain after cleaning. Exiting process.", level = "ERROR", **LOG)
            return 1 # Exit if no data remains

        temp_df.to_csv(cleaned_file_path, index=False)

        utils.log(f"Cleaned data saved to: {cleaned_file_path}", level = "INFO", **LOG)

    except Exception as e:
        utils.log(f"Error during review text cleaning: {e}", level = "ERROR", **LOG)
        return 1 # Exit if cleaning fails

    return 0

if __name__ == "__main__":
    main()