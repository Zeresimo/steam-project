import pandas as pd       
import matplotlib.pyplot as plt   
import seaborn as sns     
from datetime import datetime
import wordcloud
import nltk
import numpy as np

from paths import EDA_PLOTS, EDA_WORDCLOUDS, EDA_LOGS
from Pipeline.utils import utils


LOG = {
    "base_path": EDA_LOGS + "/",
    "filename": "eda_log.txt"
}


paths = [EDA_PLOTS, EDA_WORDCLOUDS, EDA_LOGS]


utils.ensure_directory(paths)

def get_basic_data(df):
    """
    Print a basic overview of the DataFrame:
    - DataFrame info
    - Numerical stats
    - First rows
    - Sentiment counts
    - Missing values
    - Duplicate count
    """

    if df is None or df.empty:
        return utils.error("get_basic_data: DataFrame is empty or None.", LOG)

    if not utils.validate_column_type(df, "voted_up", (bool, np.bool_)):
        return utils.error("get_basic_data: Missing or invalid 'voted_up' column.", LOG)
        
    utils.info("Generating basic data overview.", LOG)

    print("\n - - - Data frame info: - - - \n")
    df.info(verbose=True)
    
    print("\n - - - Descriptive Statistics - - - \n")
    print(df.describe())

    print("\n - - - First Five Rows - - - \n")
    print(df.head())

    print("\n - - - Review Sentiment Counts - - - \n")
    print(df['voted_up'].value_counts())

    print("\n - - - Missing Values per Column - - - \n")
    print(df.isnull().sum()) 

    print("\n - - - Duplicates in DataFrame - - - \n")
    print(df.duplicated().sum()) 

    utils.info("Basic data overview generated.", LOG)

def plot_sentiment_distribution(df):
    """
    Plot the distribution of positive vs negative reviews.

    Args:
        df (DataFrame): Processed review dataset containing 'voted_up'.

    Returns:
        None
    """

    if not utils.validate_column_type(df, "voted_up", (bool, np.bool_)):
        return utils.error("plot_sentiment_distribution: Invalid 'voted_up' column.", LOG)

    counts = df["voted_up"].value_counts().reindex([True, False], fill_value=0)
    labels = ["Positive", "Negative"]
    colors = ["#2ecc71", "#e74c3c"]
    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(labels, counts.values, color=colors)

    # Add value labels above bars
    for bar, value in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=11
        )

    plt.title(f"Sentiment Distribution for {game_name}", fontsize=14)
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Number of Reviews", fontsize=12)

    utils.save_plot(plt, "Sentiment Distribution", game_name, LOG)

def plot_length_distribution(df):
    """
    Plot the distribution of review lengths (in characters).

    Args:
        df (DataFrame): Processed review dataset containing 'review'.

    Returns:
        None
    """

    if not utils.validate_column_type(df, "review", (str,)):
        return utils.error("plot_length_distribution: Invalid 'review' column.", LOG)
    
    lengths = df['review'].str.len()
    cutoff = np.percentile(lengths, 99)
    filtered = lengths[lengths <= cutoff]
    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.hist(filtered, edgecolor="black", linewidth=0.7, alpha=0.8)

    plt.title(f"Review Length Distribution for {game_name}", fontsize=14)
    plt.xlabel("Review Length (characters)", fontsize=12)
    plt.ylabel("Number of Reviews", fontsize=12)

    utils.save_plot(plt, "Review Length Distribution", game_name, LOG)

def plot_reviews_over_time(df):
    """
    Plot the number of reviews submitted per day.

    Args:
        df (DataFrame): Review dataset containing 'timestamp_created'.

    Returns:
        None
    """

    if not utils.validate_column_type(df, "timestamp_created", (int, np.integer)):
        return utils.error("plot_reviews_over_time: Invalid 'timestamp_created' column.", LOG)
    
    df['date'] = df['timestamp_created'].apply(lambda ts: datetime.fromtimestamp(ts).date())
    daily_counts = df.groupby('date').size().sort_index()
    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.plot(daily_counts.index, daily_counts.values)
    
    plt.title(f"Reviews Over Time for {game_name}", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Number of Reviews", fontsize=12)
    plt.xticks(rotation=30)

    utils.save_plot(plt, "Reviews Over Time", game_name, LOG)

def plot_playtime_distribution(df):
    """
    Plot the distribution of playtime (in hours).
    Values above the 99th percentile are removed to avoid extreme outliers.
    
    Args:
        df (DataFrame): Processed review dataset containing 'playtime_forever'.

    Returns:
        None
    """

    if not utils.validate_column_type(df, "playtime_forever", (int, float, np.integer, np.floating)):
        return utils.error("plot_playtime_distribution: Invalid 'playtime_forever' column.", LOG)
    
    playtimes = df['playtime_forever']
    cutoff = np.percentile(playtimes, 99)
    filtered = playtimes[playtimes <= cutoff]
    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.hist(filtered, bins=30, edgecolor="black", linewidth=0.7, alpha=0.8)
    plt.xscale("log")

    plt.title(f"Playtime Distribution for {game_name}", fontsize=14)
    plt.xlabel("Playtime (hours)", fontsize=12)
    plt.ylabel("Number of Reviews", fontsize=12)
    
    utils.save_plot(plt, "Playtime Distribution", game_name, LOG)

def plot_playtime_vs_sentiment(df):
    """
    Plot playtime distribution for positive vs negative reviews.
    Extreme outliers above the 99th percentile are removed.

    Args:
        df (DataFrame): Review dataset containing 'playtime_forever' and 'voted_up'.

    Returns:
        None
    """

    if not utils.validate_column_type(df, "voted_up", (bool, np.bool_)):
        return utils.error("plot_playtime_vs_sentiment: Invalid 'voted_up' column.", LOG)
    
    if not utils.validate_column_type(df, "playtime_forever", (int, float, np.integer, np.floating)):
        return utils.error("plot_playtime_vs_sentiment: Invalid 'playtime_forever' column.", LOG)
    
    playtimes = df['playtime_forever']
    cutoff = np.percentile(playtimes, 99)
    filtered = df[playtimes <= cutoff]

    pos = filtered[filtered['voted_up'] == True]['playtime_forever']
    neg = filtered[filtered['voted_up'] == False]['playtime_forever']

    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.boxplot(
        [pos, neg], 
        tick_labels=["Positive", "Negative"], 
        patch_artist=True,
        boxprops=dict(facecolor="#82caff", alpha=0.7), 
        whiskerprops=dict(linewidth=1.5)
    )

    plt.title(f"Playtime versus Sentiment for {game_name}", fontsize=14)
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Playtime (hours)", fontsize=12)
    plt.yscale("log")

    utils.save_plot(plt, "Playtime versus Sentiment", game_name, LOG)

def plot_sentiment_ratio_vs_time(df):
    """
    Plot the daily sentiment ratio (positive reviews / total reviews).

    Args:
        df (DataFrame): Review dataset containing 'voted_up' and 'timestamp_created'.

    Returns:
        None
    """

    if not utils.validate_column_type(df, "voted_up", (bool, np.bool_)):
        return utils.error("plot_sentiment_ratio_vs_time: Invalid 'voted_up' column.", LOG)
    
    if not utils.validate_column_type(df, "timestamp_created", (int, np.integer)):
        utils.error("plot_sentiment_ratio_vs_time: Invalid 'timestamp_created' column.", LOG)
    
    # Convert timestamp to date
    df['date'] = df['timestamp_created'].apply(lambda ts: datetime.fromtimestamp(ts).date())
    
    # Group by date
    daily_pos = df[df["voted_up"] == True].groupby("date").size()
    daily_total = df.groupby("date").size()

    ratio = (daily_pos / daily_total).fillna(0).sort_index()
    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.plot(ratio.index, ratio.values)
    
    plt.title(f"Daily Sentiment Ratio for {game_name}", fontsize=14)
    plt.xlabel("Dates", fontsize=12)
    plt.ylabel("Positive Review Ratio", fontsize=12)
    plt.xticks(rotation=30)

    utils.save_plot(plt, "Daily Sentiment Ratio", game_name, LOG)

def plot_wordcloud(df):
    """
    Generate word clouds for:
        - All reviews combined
        - Positive reviews
        - Negative reviews

    Args:
        df (DataFrame): Review dataset containing 'review' and 'voted_up'.

    Returns:
        None
    """

    if not utils.validate_column_type(df, "review", (str,)):
        return utils.error("plot_wordcloud: Invalid 'review' column.", LOG)
    
    if not utils.validate_column_type(df, "voted_up", (bool, np.bool_)):
        return utils.error("plot_wordcloud: Invalid 'voted_up' column.", LOG)
    
    game_name = df['game_name'].iloc[0]

    # Prepare text
    text_all = " ".join(df["review"])
    text_neg = " ".join(df[df["voted_up"] == True]["review"])
    text_pos = " ".join(df[df["voted_up"] == False]["review"])

    def generate_cloud(title, text):
        wc = wordcloud.WordCloud(
            width=1600,
            height=800,
            background_color="white"
        ).generate(text)

        plt.figure(figsize=(16, 8))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")

        utils.save_plot(plt, title, game_name, LOG, file_path=EDA_WORDCLOUDS)

    generate_cloud("wordcloud_ALL", text_all)
    generate_cloud("wordcloud_positive", text_pos)
    generate_cloud("wordcloud_negative", text_neg)

def plot_correlation_heatmap(df):
    """
    Plot a heatmap showing correlations between:
        - playtime_forever
        - num_reviews
        - review length
        - sentiment (voted_up as int)

    Args:
        df (DataFrame): Cleaned review dataset.

    Returns:
        None
    """

    if not utils.validate_column_type(df, "playtime_forever", (int, float, np.integer, np.floating)):
        return utils.error("plot_correlation_heatmap: Invalid 'playtime_forever' column.", LOG)
    
    if not utils.validate_column_type(df, "review", (str,)):
        return utils.error("plot_correlation_heatmap: Invalid 'review' column.", LOG)
        
    if not utils.validate_column_type(df, "voted_up", (bool, np.bool_)):
        return utils.error("plot_correlation_heatmap: Invalid 'voted_up' column.", LOG)
          
    if not utils.validate_column_type(df, "num_reviews", (int, np.integer)):
        return utils.error("plot_correlation_heatmap: Invalid 'num_reviews' column.", LOG)
        
    # Compute features
    df["review_length"] = df['review'].str.len()
    df["sentiment"] = df["voted_up"].astype(int)
    
    numeric_df = df[["playtime_forever", "num_reviews", "review_length", "sentiment"]]
    corr = numeric_df.corr()

    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

    plt.title(f"Correlation Heatmap for {game_name}", fontsize=14)

    utils.save_plot(plt, "Correlation Heatmap", game_name, LOG)

def plot_playtime_vs_length(df):
    """
    Plot the relationship between playtime and review length.
    Reviews above the 99th percentile in playtime are removed to avoid distortion.

    Args:
        df (DataFrame): Review dataset containing 'review' and 'playtime_forever'.

    Returns:
        None
    """

    if not utils.validate_column_type(df, "review", (str,)):
        return utils.error("plot_playtime_vs_length: Invalid 'review' column.", LOG)
    
    if not utils.validate_column_type(df, "playtime_forever", (int, float, np.integer, np.floating)):
        return utils.error("plot_playtime_vs_length: Invalid 'playtime_forever' column.", LOG)

    df["review_length"] = df['review'].str.len()

    cutoff = np.percentile(df['playtime_forever'], 99)
    filtered = df[df["playtime_forever"] <= cutoff]

    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.scatter(filtered["playtime_forever"], 
                filtered["review_length"], 
                alpha=0.4
                )
    
    plt.title(f"Playtime versus Review Length for {game_name}", fontsize=14)
    plt.xlabel("Playtime [hours]", fontsize=12)
    plt.ylabel("Review Length (characters)", fontsize=12)
    plt.xscale("log")

    utils.save_plot(plt, "Playtime versus Review Length", game_name, LOG)

def plot_length_vs_sentiment(df):
    """
    Plot review length distribution for positive vs negative reviews.
    Review lengths above the 99th percentile are removed to avoid distortion.

    Args:
        df (DataFrame): Review dataset containing 'review' and 'voted_up'.

    Returns:
        None
    """

    if not utils.validate_column_type(df, "review", (str,)):
        return utils.error("plot_length_vs_sentiment: Invalid 'review' column.", LOG)
    
    if not utils.validate_column_type(df, "voted_up", (bool, np.bool_)):
        return utils.error("plot_length_vs_sentiment: Invalid 'voted_up' column.", LOG)
    
    df["review_length"] = df['review'].str.len()

    cutoff = np.percentile(df["review_length"], 99)
    filtered = df[df["review_length"] <= cutoff]

    neg = filtered[filtered["voted_up"] == False]["review_length"]
    pos = filtered[filtered["voted_up"] == True]["review_length"]

    game_name = df["game_name"].iloc[0]
    
    plt.figure(figsize=(14, 6))
    plt.boxplot(
        [pos, neg], 
        tick_labels=["Positive", "Negative"], 
        patch_artist=True,
        boxprops=dict(facecolor="#82caff", alpha=0.7), 
        whiskerprops=dict(linewidth=1.5)
        )
    
    plt.title(f"Review Length versus Sentiment for {game_name}", fontsize=14)
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Review Length (characters)", fontsize=12)
    plt.yscale("log")

    utils.save_plot(plt, "Review Length versus Sentiment", game_name, LOG)

def plot_reviewer_exp(df):
    """
    Plot the distribution of reviewer experience (number of reviews by each author).
    Values above the 99th percentile are removed to avoid extreme outliers.

    Args:
        df (DataFrame): Review dataset containing 'num_reviews'.

    Returns:
        None
    """

    if not utils.validate_column_type(df, "num_reviews", (int, np.integer)):
        return utils.error("plot_reviewer_exp: Invalid 'num_reviews' column.", LOG)
    
    exp = df["num_reviews"]
    cutoff = np.percentile(exp, 99)
    filtered = exp[exp <= cutoff]
    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.hist(filtered, bins=30, edgecolor="black", linewidth=0.7, alpha=0.8)
    plt.xscale("log")
    
    plt.title(f"Reviewer Experience Distribution for {game_name}", fontsize=14)
    plt.xlabel("Total Reviews by User", fontsize=12)
    plt.ylabel("Number of Players", fontsize=12)
    
    utils.save_plot(plt, "Reviewer Experience Distribution", game_name, LOG)

def plot_hourly_review_distribution(df):
    """
    Plot the distribution of reviews by hour of the day (0–23).

    Args:
        df (DataFrame): Review dataset containing 'timestamp_created'.

    Returns:
        None
    """

    if not utils.validate_column_type(df, "timestamp_created", (int, np.integer)):
        return utils.error("plot_hourly_review_distribution: Invalid 'timestamp_created' column.", LOG)
    
    df["hour"] = df["timestamp_created"].apply(lambda ts: datetime.fromtimestamp(ts).hour)
    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.hist(df["hour"], bins=24, edgecolor="black", linewidth=0.7, alpha=0.8)
    
    plt.title(f"Review Activity by Hour (24h) for {game_name}", fontsize=14)
    plt.xlabel("Hour of Day (0-23)", fontsize=12)
    plt.ylabel("Number of Reviews", fontsize=12)

    utils.save_plot(plt, "Hourly Review Distribution", game_name, LOG)

def plot_weekday_review_distribution(df):
    """
    Plot the distribution of reviews by weekday.

    Args:
        df (DataFrame): Review dataset containing 'timestamp_created'.

    Returns:
        None
    """

    if not utils.validate_column_type(df, "timestamp_created", (int, np.integer)):
        return utils.error("plot_weekday_review_distribution: Invalid 'timestamp_created' column.", LOG)
    
    # Convert timestamp = weekday name
    df["weekday"] = df["timestamp_created"].apply(
        lambda ts: datetime.fromtimestamp(ts).strftime("%a")
    )

    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    counts = df["weekday"].value_counts().reindex(order, fill_value=0)

    game_name = df["game_name"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.bar(order, counts.values)

    plt.title(f"Review Activity by Weekday for {game_name}", fontsize=14)
    plt.xlabel("Day of Week", fontsize=12)
    plt.ylabel("Number of Reviews", fontsize=12)

    utils.save_plot(plt, "Weekday Review Distribution", game_name, LOG)

def run_all_eda(df):
    """
    Run all EDA plots on the provided DataFrame.

    Args:
        df (DataFrame): Processed review dataset.

    Returns:
        None
    """

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
    """
    Entry point for running EDA on the most recent processed CSV.

    Workflow:
        1. Load latest processed CSV.
        2. Print basic data overview.
        3. Generate all EDA plots.

    Returns:
        None | int: Returns 1 on failure, otherwise None.
    """

    latest_file = utils.get_latest_csv(LOG, base_path="Pipeline/data/processed")
    if latest_file is None:
        return utils.error("main: No processed CSV file found for EDA.", LOG)
    
    try:
        df = pd.read_csv(latest_file)
        utils.info(f"Loaded processed CSV: {latest_file}", LOG)
    except Exception as err:
        return utils.error(f"main: Error reading CSV ({err})", LOG)
    
    # Basic data summary before plotting
    get_basic_data(df)
    
    # Generate all plots
    run_all_eda(df)

if __name__ == "__main__":
    main()