import os
import pandas as pd

from paths import PIPE_CLEAN, PIPE_PROCESSED, PIPE_LOGS
from Pipeline.utils import utils


LOG = {
    "base_path": PIPE_LOGS + "/",
    "filename": "process_log.txt"
}

paths = [PIPE_CLEAN, PIPE_PROCESSED, PIPE_LOGS]
utils.ensure_directory(paths)

def clean_rows(df):
    """
    Perform structural cleaning on the raw reviews DataFrame.

    Steps:
        - Ensure required columns exist.
        - Remove rows with missing review text.
        - Remove duplicates.
        - Remove rows with invalid review content.

    Args:
        df: Pandas DataFrame loaded from raw CSV.

    Returns:
        Cleaned DataFrame or None if validation fails.

    NOTES:
        This does not clean text content yet — only structural issues.
    """

    # Validate required columns
    if not utils.validate_dataframe_columns(df):
        return utils.error("Data Frame validation failed during cleaning.", LOG)
        
    # Drop rows missing review text
    df = df.dropna(subset=['review'])

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Filter out invalid review content
    df = df[df['review'].apply(utils.validate_review_text)]
    
    return df
    
def clean_reviews(text):
    """
    Clean a single review's text.

    Steps:
        - Validate text is non-empty and meaningful.
        - Clean text with utils.clean_text().
        - Return None if cleaning produces empty text.

    Args:
        text: Raw review string.

    Returns:
        Cleaned text string or None if invalid.
    """

    if not utils.validate_review_text(text):
        return utils.error("Invalid review text encountered during cleaning.", LOG)

    cleaned = utils.clean_text(text)
    return cleaned.strip() or None

def main():
    """
    Main processing pipeline for converting raw cleaned CSV files into fully
    processed datasets.

    Workflow:
        1. Load latest CSV from PIPE_CLEAN.
        2. Apply structural cleaning (clean_rows).
        3. Apply text cleaning (clean_reviews).
        4. Drop failed rows.
        5. Save to PIPE_PROCESSED.

    Returns:
        0 on success, 1 on failure.
    """
    # Get latest cleaned CSV file
    latest_file = utils.get_latest_csv(LOG, base_path=PIPE_CLEAN)
    if latest_file is None:
        utils.error("No CSV files to process. Exiting.", LOG)
        return 1 
    
    cleaned_file_path = os.path.join(PIPE_PROCESSED, f"cleaned_{os.path.basename(latest_file)}")
    
    utils.info(f"Processing file: {latest_file}", LOG)

    # Load CSV
    try:
        unclean_df = pd.read_csv(latest_file)
        utils.info("CSV file read successfully.", LOG)
    except Exception as err:
        utils.error(f"Error reading CSV file: {err}", LOG)
        return 1
    
    # Structural cleaning (drop duplicates, invalid rows)
    temp_df = clean_rows(unclean_df)
    if temp_df is None:
        utils.error("Data cleaning failed. Exiting process.", LOG)
        return 1 
    
    try:
        utils.info("Starting review text cleaning.", LOG)

        # Clean review text
        temp_df['review'] = temp_df['review'].apply(clean_reviews)

        # Remove rows that failed text cleaning
        temp_df = temp_df.dropna(subset=['review'])

        utils.info("Review text cleaning completed.", LOG)

        if temp_df.empty:
            utils.error("No valid reviews remain after cleaning. Exiting process.", LOG)
            return 1
        
        # Save processed dataset
        temp_df.to_csv(cleaned_file_path, index=False)
        utils.info(f"Cleaned data saved to: {cleaned_file_path}", LOG)

    except Exception as err:
        utils.error(f"Error during review text cleaning: {err}", LOG)
        return 1 

    return 0

if __name__ == "__main__":
    main()