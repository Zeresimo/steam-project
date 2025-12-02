import os
import sys
import pandas as pd

# Ensure project root is in sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from paths import ROOT
from Pipeline.pipeline import search_games, fetch_reviews
from Pipeline.utils import utils

# API Wrapper: Search Games
def api_search_games(query):
    """
    Wrapper for search_games() used by the dashboard.

    Args:
        query (str): Game name or search text.

    Returns:
        list: Search results or empty list.
    """

    results = search_games(query)
    if results is None:
        utils.error("api_search_games: search_games() returned None.", None)
        return []

    return results

# API Wrapper: Fetch Reviews
def api_fetch_reviews(appid, limit):
    """
    Fetch *limit* reviews for use in the dashboard.

    Args:
        appid (int): Steam AppID.
        limit (int): Max number of reviews.

    Returns:
        list | None: List of review dicts, or None on failure.
    """

    utils.info(f"api_fetch_reviews: Fetching {limit} reviews for appid={appid}.", None)

    stop_condition = lambda review, reviews, lim: len(reviews) >= lim

    reviews = fetch_reviews(appid, limit, stop_condition)

    if reviews is None:
        utils.error("api_fetch_reviews: fetch_reviews() returned None.", None)
        return None

    return reviews

# API Wrapper: Clean Reviews + Predict
def api_clean_and_predict(reviews, model, vectorizer, appid, game_name):
    """
    Clean raw reviews and run predictions in batch for the dashboard.

    Args:
        reviews (list): List of raw review dictionaries.
        model: Trained ML model.
        vectorizer: TF-IDF vectorizer.
        appid (int): Steam AppID.
        game_name (str): Name of the selected game.

    Returns:
        pd.DataFrame: Cleaned + predicted dataset.
    """

    if reviews is None or len(reviews) == 0:
        utils.error("api_clean_and_predict: No reviews received.", None)
        return pd.DataFrame()

    utils.info(f"api_clean_and_predict: Cleaning and predicting {len(reviews)} reviews.", None)

    cleaned = []
    texts = []

    # Clean each review
    for r in reviews:
        raw_text = r.get("review", "")
        cleaned_text = utils.clean_text(raw_text)

        if not cleaned_text.strip():
            continue

        texts.append(cleaned_text)

        cleaned.append({
            "appid": appid,
            "game_name": game_name,
            "original_review": raw_text,
            "cleaned_review": cleaned_text,
            "timestamp_created": r.get("timestamp_created"),
            "playtime_forever": r.get("author", {}).get("playtime_forever"),
            "num_reviews": r.get("author", {}).get("num_reviews"),
            "voted_up": r.get("voted_up")
        })

    if len(cleaned) == 0:
        utils.error("api_clean_and_predict: All reviews filtered out after cleaning.", None)
        return pd.DataFrame()

    # Vectorize in one batch
    try:
        X = vectorizer.transform(texts)
    except Exception as err:
        utils.error(f"api_clean_and_predict: Vectorization failed ({err}).", None)
        return pd.DataFrame()

    # Predict in one batch
    try:
        preds = model.predict(X)
    except Exception as err:
        utils.error(f"api_clean_and_predict: Prediction failed ({err}).", None)
        return pd.DataFrame()

    # Confidence scores (if supported)
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            confidence = model.predict_proba(X)[:, 1]
        except Exception:
            utils.info("api_clean_and_predict: predict_proba failed; confidence omitted.", None)
    
    # Attach predictions
    for i, row in enumerate(cleaned):
        sentiment = "Positive" if preds[i] == 1 else "Negative"
        row["predicted_sentiment"] = sentiment
        row["confidence"] = float(confidence[i]) if confidence is not None else None

    df = pd.DataFrame(cleaned)
    return df
