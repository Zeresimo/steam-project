import pandas as pd       
import matplotlib.pyplot as plt   
import seaborn as sns     
from datetime import datetime
import os
import wordcloud
import nltk
import utils.utils as utils
import numpy as np

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

    if not utils.validate_column_type(df, "voted_up", (bool, np.bool_)):
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

def plot_sentiment_distribution(df):
    if not utils.validate_column_type(df, "voted_up", (bool, np.bool_)):
        utils.log("Voted_up column validation error", level="ERROR", **LOG)
        return
    
    counts = df['voted_up'].value_counts()

    counts = counts.reindex([True, False], fill_value=0)

    labels = ["Positive", "Negative"]
    colors = ["#2ecc71", "#e74c3c"]
    
    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(labels, counts.values, color=colors)
    
    plt.title(f"Sentiment Distribution for {game_name}", fontsize=14)
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Number of Reviews", fontsize=12)

    for bar, value in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=11
        )

    utils.save_plot(plt, "Sentiment Distribution", game_name, LOG)

def plot_length_distribution(df):
    if not utils.validate_column_type(df, "review", (str,)):
        utils.log("Review column validation error", level="ERROR", **LOG)
        return
    
    review_length = df['review'].str.len()

    percentile = np.percentile(review_length, 99)
    filtered_reviews = review_length[review_length <= percentile]

    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.hist(filtered_reviews, edgecolor="black", linewidth=0.7, alpha=0.8)
    plt.title(f"Review Length Distribution for {game_name}", fontsize=14)
    plt.xlabel("Review Length (characters)", fontsize=12)
    plt.ylabel("Number of Reviews", fontsize=12)

    utils.save_plot(plt, "Review Length Distribution", game_name, LOG)

def plot_reviews_over_time(df):
    if not utils.validate_column_type(df, "timestamp_created", (int, np.integer)):
        utils.log("Timestamp_created column validation error", level="ERROR", **LOG)
        return
    
    df['date'] = df['timestamp_created'].apply(lambda ts: datetime.fromtimestamp(ts).date())
    daily_counts = df.groupby('date').size()
    daily_counts = daily_counts.sort_index()

    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.plot(daily_counts.index, daily_counts.values)
    
    plt.title(f"Reviews Over Time for {game_name}", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Number of Reviews", fontsize=12)
    plt.xticks(rotation=30)

    utils.save_plot(plt, "Reviews Over Time", game_name, LOG)

def plot_playtime_distribution(df):
    if not utils.validate_column_type(df, "playtime_forever", (int, float, np.integer, np.floating)):
        utils.log("Playtime_forever column validation error", level="ERROR", **LOG)
        return
    
    playtimes = df['playtime_forever']
    percentile = np.percentile(playtimes, 99)
    filtered_playtime = playtimes[playtimes <= percentile]
    
    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.hist(filtered_playtime, bins=30, edgecolor="black", linewidth=0.7, alpha=0.8)
    plt.title(f"Playtime Distribution for {game_name}", fontsize=14)
    plt.xlabel("Playtime (hours)", fontsize=12)
    plt.ylabel("Number of Reviews", fontsize=12)
    plt.xscale("log")

    utils.save_plot(plt, "Playtime Distribution", game_name, LOG)

def plot_playtime_vs_sentiment(df):
    if not utils.validate_column_type(df, "voted_up", (bool, np.bool_)):
        utils.log("Voted_up column validation error", level="ERROR", **LOG)
        return
    
    if not utils.validate_column_type(df, "playtime_forever", (int, float, np.integer, np.floating)):
        utils.log("Playtime_forever column validation error", level="ERROR", **LOG)
        return
    
    positive = df[df['voted_up'] == True]['playtime_forever']
    negative = df[df['voted_up'] == False]['playtime_forever']
    cutoff = np.percentile(df['playtime_forever'], 99)
    positive = positive[positive <= cutoff]
    negative = negative[negative <= cutoff]

    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.boxplot([positive, negative], tick_labels=["Positive", "Negative"], patch_artist=True,
            boxprops=dict(facecolor="#82caff", alpha=0.7), whiskerprops=dict(linewidth=1.5))
    plt.title(f"Playtime versus Sentiment for {game_name}", fontsize=14)
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Playtime (hours)", fontsize=12)
    plt.yscale("log")

    utils.save_plot(plt, "Playtime versus Sentiment", game_name, LOG)

def plot_sentiment_ratio_vs_time(df):
    if not utils.validate_column_type(df, "voted_up", (bool, np.bool_)):
        utils.log("Voted_up column validation error", level="ERROR", **LOG)
        return
    
    if not utils.validate_column_type(df, "timestamp_created", (int, np.integer)):
        utils.log("Timestamp_created column validation error", level="ERROR", **LOG)
        return
    
    df['date'] = df['timestamp_created'].apply(lambda ts: datetime.fromtimestamp(ts).date())
    daily_positive = df[df["voted_up"] == True].groupby("date").size()
    daily_total = df.groupby("date").size()

    sentiment_ratio = daily_positive / daily_total
    sentiment_ratio = sentiment_ratio.fillna(0)

    sentiment_ratio = sentiment_ratio.sort_index()

    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.plot(sentiment_ratio.index, sentiment_ratio.values)
    
    plt.title(f"Daily Sentiment Ratio for {game_name}", fontsize=14)
    plt.xlabel("Dates", fontsize=12)
    plt.ylabel("Positive Review Ratio", fontsize=12)
    plt.xticks(rotation=30)
    plt.tight_layout()

    utils.save_plot(plt, "Daily Sentiment Ratio", game_name, LOG)

def plot_wordcloud(df):
    if not utils.validate_column_type(df, "review", (str,)):
        utils.log("Review column validation error", level="ERROR", **LOG)
        return
    
    if not utils.validate_column_type(df, "voted_up", (bool, np.bool_)):
        utils.log("Voted_up column validation error", level="ERROR", **LOG)
        return
    
    text = " ".join(df["review"])
    negative = df[df["voted_up"] == False]["review"]
    positive = df[df["voted_up"] == True]["review"]

    ntext = " ".join(negative)
    ptext = " ".join(positive)

    def wordcloud_plot(title, text):
        wc = wordcloud.WordCloud(
        width=1600,
        height=800,
        background_color="white"
        ).generate(text)

        game_name = df["game_name"].iloc[0]

        plt.figure(figsize=(16, 8))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")

        utils.save_plot(plt, title, game_name, LOG, file_path="../steam-project/EDA/wordclouds/")

    wordcloud_plot("wordcloud_ALL", text)
    wordcloud_plot("wordcloud_positive", ptext)
    wordcloud_plot("wordcloud_negative", ntext)

def plot_correlation_heatmap(df):
    if not utils.validate_column_type(df, "playtime_forever", (int, float, np.integer, np.floating)):
        utils.log("Playtime_forever column validation error", level="ERROR", **LOG)
        return
    
    if not utils.validate_column_type(df, "review", (str,)):
        utils.log("Review column validation error", level="ERROR", **LOG)
        return
    
    if not utils.validate_column_type(df, "voted_up", (bool, np.bool_)):
        utils.log("Voted_up column validation error", level="ERROR", **LOG)
        return
    
    if not utils.validate_column_type(df, "num_reviews", (int, np.integer)):
        utils.log("Num_reviews column validation error", level="ERROR", **LOG)
        return
    
    df["sentiment"] = df["voted_up"].astype(int)
    df["review_length"] = df['review'].str.len()
    
    numeric_df = df[["playtime_forever", "num_reviews", "review_length", "sentiment"]]
    corr = numeric_df.corr()

    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

    utils.save_plot(plt, "Correlation Heatmap", game_name, LOG)

def plot_playtime_vs_length(df):
    if not utils.validate_column_type(df, "review", (str,)):
        utils.log("Review column validation error", level="ERROR", **LOG)
        return
    
    if not utils.validate_column_type(df, "playtime_forever", (int, float, np.integer, np.floating)):
        utils.log("Playtime_forever column validation error", level="ERROR", **LOG)
        return

    df["review_length"] = df['review'].str.len()
    cutoff = np.percentile(df['playtime_forever'], 99)
    filtered = df[df["playtime_forever"] <= cutoff]

    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.scatter(filtered["playtime_forever"], filtered["review_length"], alpha=0.4)
    plt.title(f"Playtime versus Review Length for {game_name}", fontsize=14)
    plt.xlabel("Playtime [hours]", fontsize=12)
    plt.ylabel("Review Length (characters)", fontsize=12)
    plt.xscale("log")

    utils.save_plot(plt, "Playtime versus Review Length", game_name, LOG)

def plot_length_vs_sentiment(df):
    if not utils.validate_column_type(df, "review", (str,)):
        utils.log("Review column validation error", level="ERROR", **LOG)
        return
    
    if not utils.validate_column_type(df, "voted_up", (bool, np.bool_)):
        utils.log("Voted_up column validation error", level="ERROR", **LOG)
        return
    
    df["review_length"] = df['review'].str.len()
    negative = df[df["voted_up"] == False]["review_length"]
    positive = df[df["voted_up"] == True]["review_length"]

    cutoff = np.percentile(df["review_length"], 99)
    positive = positive[positive <= cutoff]
    negative = negative[negative <= cutoff]

    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.boxplot([positive, negative], tick_labels=["Positive", "Negative"], patch_artist=True,
            boxprops=dict(facecolor="#82caff", alpha=0.7), whiskerprops=dict(linewidth=1.5))
    plt.title(f"Review Length versus Sentiment for {game_name}", fontsize=14)
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Review Length (characters)", fontsize=12)
    plt.yscale("log")

    utils.save_plot(plt, "Review Length versus Sentiment", game_name, LOG)

def plot_reviewer_exp(df):
    if not utils.validate_column_type(df, "num_reviews", (int, np.integer)):
        utils.log("Num_reviews column validation error", level="ERROR", **LOG)
        return
    
    exp = df["num_reviews"]
    cutoff = np.percentile(exp, 99)
    filtered = exp[exp <= cutoff]

    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.hist(filtered, bins=30, edgecolor="black", linewidth=0.7, alpha=0.8)
    plt.title(f"Reviewer Experience Distribution for {game_name}", fontsize=14)
    plt.xlabel("Total Reviews by User", fontsize=12)
    plt.ylabel("Number of Players", fontsize=12)
    plt.xscale("log")

    utils.save_plot(plt, "Reviewer Experience Distribution", game_name, LOG)

def plot_hourly_review_distribution(df):
    if not utils.validate_column_type(df, "timestamp_created", (int, np.integer)):
        utils.log("Timestamp_created column validation error", level="ERROR", **LOG)
        return
    
    df["hour"] = df["timestamp_created"].apply(lambda ts: datetime.fromtimestamp(ts).hour)

    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.hist(df["hour"], bins=24, edgecolor="black", linewidth=0.7, alpha=0.8)
    plt.title(f"Review Activity by Hour (24h) for {game_name}", fontsize=14)
    plt.xlabel("Hour of Day (0-23)", fontsize=12)
    plt.ylabel("Number of Reviews", fontsize=12)

    utils.save_plot(plt, "Hourly Review Distribution", game_name, LOG)

def plot_weekday_review_distribution(df):
    if not utils.validate_column_type(df, "timestamp_created", (int, np.integer)):
        utils.log("Timestamp_created column validation error", level="ERROR", **LOG)
        return
    
    # Convert timestamp → weekday name
    df["weekday"] = df["timestamp_created"].apply(
        lambda ts: datetime.fromtimestamp(ts).strftime("%a")
    )

    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    counts = df["weekday"].value_counts().reindex(weekday_order, fill_value=0)

    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.bar(weekday_order, counts.values)
    plt.title(f"Review Activity by Weekday for {game_name}", fontsize=14)
    plt.xlabel("Day of Week", fontsize=12)
    plt.ylabel("Number of Reviews", fontsize=12)

    utils.save_plot(plt, "Weekday Review Distribution", game_name, LOG)


def run_all_eda(df):
    plot_sentiment_distribution(df)
    plot_length_distribution(df)
    plot_reviews_over_time(df)
    plot_playtime_distribution(df)
    plot_playtime_vs_sentiment(df)
    plot_sentiment_ratio_vs_time(df)
    plot_wordcloud(df)
    plot_correlation_heatmap(df)
    plot_playtime_vs_length(df)
    plot_length_vs_sentiment(df)
    plot_reviewer_exp(df)
    plot_hourly_review_distribution(df)
    plot_weekday_review_distribution(df)

def main():
    latest_file = utils.get_latest_csv(LOG, base_path="Pipeline/data/processed")
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
    
    run_all_eda(df)

if __name__ == "__main__":
    main()