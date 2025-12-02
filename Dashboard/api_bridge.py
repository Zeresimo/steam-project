import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from paths import ROOT

from Pipeline.pipeline import search_games, fetch_reviews, save_reviews
from Pipeline.process import clean_reviews
from Pipeline.utils.utils import clean_text, validate_single_review
import pandas as pd


def api_search_games(query):
    """Search Steam for games using your existing pipeline function."""
    results = search_games(query)
    if results is None:
        return []
    return results


def api_fetch_reviews(appid, limit):
    """Fetch N live reviews for the given game."""
    # Use your pipeline’s core fetch function
    # We bypass interactive input by directly calling fetch_reviews()
    stop_condition = lambda review, reviews, limit: len(reviews) >= limit
    reviews = fetch_reviews(appid, limit, stop_condition)
    return reviews


def api_clean_and_predict(reviews, model, vectorizer, appid, game_name):
    """Clean reviews and run predictions using your model."""
    cleaned = []
    texts = []

    for r in reviews:
        # Clean review text
        ct = clean_text(r["review"])
        if ct.strip() == "":
            continue

        texts.append(ct)
        cleaned.append({
            "appid": appid,
            "game_name": game_name,
            "original_review": r["review"],
            "cleaned_review": ct,
            "timestamp_created": r["timestamp_created"],
            "playtime_forever": r["author"]["playtime_forever"],
            "num_reviews": r["author"]["num_reviews"],
            "voted_up": r["voted_up"],
        })


    # Vectorize in one batch
    X = vectorizer.transform(texts)
    preds = model.predict(X)

    conf = None
    if hasattr(model, "predict_proba"):
        conf = model.predict_proba(X)[:, 1]

    # Attach predictions
    for i in range(len(cleaned)):
        cleaned[i]["predicted_sentiment"] = "Positive" if preds[i] == 1 else "Negative"
        cleaned[i]["confidence"] = float(conf[i]) if conf is not None else None

    df = pd.DataFrame(cleaned)
    return df
