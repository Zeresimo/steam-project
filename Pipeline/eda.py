import pandas as pd       
import matplotlib.pyplot as plt   
import seaborn as sns     
from datetime import datetime
import os
import wordcloud
import nltk
from process import get_latest_csv
import utils.utils as utils

LOG = {
    "base_path": "../steam-project/EDA/logs/",
    "filename": "eda_log.txt"
}

paths = [
    "../steam-project/EDA/plots",
    "../steam-project/EDA/wordclouds",
    "../steam-project/EDA/logs/"
]

utils.ensure_directory(paths)

def get_basic_data(df):
    if df is None or df.empty:
        utils.log("Data frame is empty or None.", level = "ERROR", **LOG)
        return

    if "voted_up" not in df.columns:
        utils.log("'voted_up' column not found in data frame.", level = "ERROR", **LOG)
        return
    
    utils.log("Generating basic data overview.", level = "INFO", **LOG)

    print("\n - - - Data frame info: - - - \n")
    df.info(verbose=True) # Detailed info about DataFrame
    
    print("\n - - - Numeric Description: - - - \n")
    print(df.describe()) # Statistical summary of numerical columns

    print("\n - - - First Five Rows: - - - \n")
    print(df.head()) # Display first few rows

    print("\n - - - Review Sentiment Counts: - - - \n")
    print(df['voted_up'].value_counts()) # Count of positive and negative votes

    print("\n - - - Missing Values per Column: - - - \n")
    print(df.isnull().sum()) # Count of missing values per column

    print("\n - - - Duplicates in DataFrame: - - - \n")
    print(df.duplicated().sum()) # Count of duplicate rows

    utils.log("Basic data overview generated.", level = "INFO", **LOG)

#def preprocess_text(df):

def main():
    latest_file = get_latest_csv("Pipeline/data/processed/")
    if latest_file is None:
        utils.log("No file found for EDA", level="ERROR", **LOG)
        return None
    try:
        df = pd.read_csv(latest_file)
        utils.log(f"Latest processed CSV file found: {latest_file}", level = "INFO", **LOG)
        utils.log("Reading CSV file for EDA.", level = "INFO", **LOG)
        get_basic_data(df)
        
    except Exception as e:
        utils.log(f"Error reading CSV file: {e}", level = "ERROR", **LOG)

if __name__ == "__main__":
    main()