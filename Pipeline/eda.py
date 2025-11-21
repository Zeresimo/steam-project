import pandas as pd       
import matplotlib.pyplot as plt   
import seaborn as sns     
from datetime import datetime
import os
import wordcloud
import nltk
from process import get_latest_csv
from utils.logger import log

base_path = "../steam-project/EDA/logs/"
filename = "eda_log.txt"

def get_basic_data(df):
    if df is None or df.empty:
        log("Data frame is empty or None.", level = "ERROR", base_path = base_path, filename = filename)
        return

    if "voted_up" not in df.columns:
        log("'voted_up' column not found in data frame.", level = "ERROR", base_path = base_path, filename = filename)
        return
    
    log("Generating basic data overview.", level = "INFO", base_path = base_path, filename = filename)

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

    log ("Basic data overview generated.", level = "INFO", base_path = base_path, filename = filename)

#def preprocess_text(df):

def main():
    os.makedirs("../steam-project/EDA", exist_ok=True) # Ensure processed directory exists
    latest_file = get_latest_csv("Pipeline/data/processed/")
    
    try:
        df = pd.read_csv(latest_file)
        log(f"Latest processed CSV file found: {latest_file}", level = "INFO", base_path = base_path, filename = filename)
        log("Reading CSV file for EDA.", level = "INFO", base_path = base_path, filename = filename)
        get_basic_data(df)
        
    except Exception as e:
        log(f"Error reading CSV file: {e}", level = "ERROR", base_path = base_path, filename = filename)

if __name__ == "__main__":
    main()