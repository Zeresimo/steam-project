import pandas as pd       
import matplotlib.pyplot as plt   
import seaborn as sns     
from datetime import datetime
import os
import wordcloud
import nltk
from process import get_latest_csv, log_info, log_error


def get_basic_data(df):
    
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

def main():
    os.makedirs("../steam-project/EDA", exist_ok=True) # Ensure processed directory exists
    latest_file = get_latest_csv("Pipeline/data/processed/")
    df = pd.read_csv(latest_file)

    if df is not None:
        log_info("Latest File Found, Starting EDA process.", "../steam-project/EDA/logs/", "eda_log.txt")
        get_basic_data(df)
    else:
        log_error("No CSV files found in the processed directory.", "../steam-project/EDA/logs/", "eda_error.txt")

if __name__ == "__main__":
    main()