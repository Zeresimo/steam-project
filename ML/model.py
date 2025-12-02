import pandas as pd
import numpy as np
import joblib
import json
import os

from datetime import datetime
from paths import MODEL_DIR, LOG_DIR, OUTPUT_DIR

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from Pipeline.utils import utils

LOG = {
    "base_path": LOG_DIR + "/",
    "filename": "ML_train_log.txt"
}

paths = [LOG_DIR, MODEL_DIR, OUTPUT_DIR]

utils.ensure_directory(paths)

def load_model_data():
    """
    Load the latest processed dataset and return review texts and labels.

    Returns:
        (Series, Series) or (None, None):
            x: cleaned review text
            y: integer sentiment labels (0/1)
    """

    latest_file = utils.get_latest_csv(LOG, base_path="Pipeline/data/processed")
    if not latest_file:
        return utils.error2("load_model_data: No processed CSV file found.", LOG)
    
    try:
        utils.info(f"Loading processed CSV: {latest_file}", LOG)
        df = pd.read_csv(latest_file)
        utils.info(f"Loaded dataset with {len(df)} rows.", LOG)
    except Exception as e:
        return utils.error2(f"Error reading CSV file: {e}", LOG)

    # Remove missing reviews
    df = df.dropna(subset=['review'])

    # Validate essential columns
    if not utils.validate_column_type(df, "review", (str,)):
        return utils.error2("load_model_data: Invalid 'review' column.", LOG)

    if not utils.validate_column_type(df, "voted_up", (bool, np.bool_)):
        return utils.error2("load_model_data: Invalid 'voted_up' column.", LOG)

    # Clean text again for ML
    df['review'] = df['review'].apply(utils.clean_text)
    df = df.dropna(subset=['review'])
    utils.info("Secondary Review text cleaning completed.", LOG)

    df['voted_up'] = df['voted_up'].astype(int)

    x = df["review"]
    y = df["voted_up"]

    return x, y

def split_data(x, y):
    """
    Split review text and labels into training and test sets.

    Args:
        x (Series): Clean review text.
        y (Series): Sentiment labels (0/1).

    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    """

    utils.info("split_data: Starting train/test split.", LOG)

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x, 
            y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
    except Exception as err:
        return utils.error2(f"split_data: Train/test split failed ({err}).", LOG)

    utils.info("split_data: Train/test split completed.", LOG)
    utils.info(f"split_data: Training size = {len(x_train)}, Test size = {len(x_test)}", LOG)
    utils.info(f"split_data: Training label distribution:\n{y_train.value_counts()}", LOG)
    utils.info(f"split_data: Test label distribution:\n{y_test.value_counts()}", LOG)

    return x_train, x_test, y_train, y_test

def vectorize_text(x_train, x_test):
    """
    Fit a TF-IDF vectorizer on training text and transform both
    training and test sets.

    Args:
        x_train (Series): Training review text.
        x_test  (Series): Test review text.

    Returns:
        tuple:
            - x_train_vector (sparse matrix)
            - x_test_vector  (sparse matrix)
            - vectorizer     (TfidfVectorizer)
    """

    utils.info("vectorize_text: Starting TF-IDF vectorization.", LOG)

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), 
            max_features=20000, 
            min_df=2
        )
        vectorizer.fit(x_train)

        x_train_vec = vectorizer.transform(x_train)
        x_test_vec = vectorizer.transform(x_test)

    except Exception as err:
        return utils.error2(f"vectorize_text: Vectorization failed ({err}).", LOG)

    utils.info(f"vectorize_text: Vocabulary size = {len(vectorizer.vocabulary_)}", LOG)
    utils.info(f"vectorize_text: Training vector shape = {x_train_vec.shape}", LOG)
    utils.info(f"vectorize_text: Test vector shape = {x_test_vec.shape}", LOG)
    utils.info("vectorize_text: TF-IDF vectorization completed.", LOG)

    return x_train_vec, x_test_vec, vectorizer

def train_logistic_regression(x_train_vec, y_train):
    """
    Train a Logistic Regression classifier on the TF-IDF vectors.

    Args:
        x_train_vec (sparse matrix): Vectorized training text.
        y_train     (Series): Training labels.

    Returns:
        model (LogisticRegression) or None on failure.
    """

    utils.info("train_logistic_regression: Starting training.", LOG)

    try:
        model = LogisticRegression(
            max_iter=2000,
            solver="liblinear"
        )
        model.fit(x_train_vec, y_train)
    except Exception as err:
        return utils.error(f"train_logistic_regression: Training failed ({err}).", LOG)

    utils.info("train_logistic_regression: Training completed.", LOG)
    utils.info(f"train_logistic_regression: Model classes = {list(model.classes_)}", LOG)

    return model

def train_naive_bayes(x_train_vec, y_train):
    """
    Train a Multinomial Naive Bayes classifier.

    Args:
        x_train_vec (sparse matrix): Vectorized training text.
        y_train     (Series): Training labels.

    Returns:
        model (MultinomialNB) or None on failure.
    """

    utils.info("train_naive_bayes: Starting training.", LOG)

    try:
        model = MultinomialNB()
        model.fit(x_train_vec, y_train)
    except Exception as err:
        return utils.error(f"train_naive_bayes: Training failed ({err}).", LOG)

    utils.info("train_naive_bayes: Training completed.", LOG)
    return model

def evaluate_model(model, x_test_vec, y_test, model_type="NONE"):
    """
    Evaluate a trained model on test data and return metrics.

    Args:
        model: Trained classifier.
        x_test_vec (matrix): Vectorized test text.
        y_test (Series): True sentiment labels.
        model_type (str): Name of the model (for logging).

    Returns:
        dict: Dictionary of evaluation metrics.
    """

    utils.info(f"evaluate_model: Evaluating model [{model_type}].", LOG)
    report = {}
    
    try:
        y_pred = model.predict(x_test_vec)
    except Exception as err:
        return utils.error(f"evaluate_model: Prediction failed ({err}).", LOG)
    
    # Core metrics
    report['accuracy'] = accuracy_score(y_test, y_pred)
    report['precision'] = precision_score(y_test, y_pred)
    report['recall'] = recall_score(y_test, y_pred)
    report['f1'] = f1_score(y_test, y_pred)

    utils.info(
    f"Metrics produced: Accuracy: {report['accuracy']}, "
    f"Precision: {report['precision']}, "
    f"Recall: {report['recall']}, "
    f"F1: {report['f1']}",
    LOG
    )

    # Classification report (text form)
    cls_report = classification_report(y_test, y_pred)
    utils.info(f"evaluate_model: Classification Report:\n{cls_report}", LOG)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    report["conf_matrix"] = cm.tolist()
    utils.info(f"evaluate_model: Confusion Matrix:\n{cm}", LOG)

    # ROC AUC (if supported by model)
    if hasattr(model, "predict_proba"):
        try: 
            y_prob = model.predict_proba(x_test_vec)[:, 1]
            report["roc_auc"] = roc_auc_score(y_test, y_prob)
            utils.info(f"evaluate_model: ROC AUC = {report['roc_auc']:.4f}", LOG)
        except Exception:
            utils.info("evaluate_model: ROC AUC skipped — probability prediction failed.", LOG)
    else:
        utils.info("evaluate_model: ROC AUC skipped — model lacks predict_proba().", LOG)

    return report

def save_model_and_vectorizer(model, vectorizer, name):
    """
    Save a trained model and its TF-IDF vectorizer.

    Args:
        model: Trained ML model.
        vectorizer: TfidfVectorizer instance.
        name (str): Base filename for saving.

    Returns:
        bool: True on success, False on failure.
    """

    utils.info(f"save_model_and_vectorizer: Saving artifacts for [{name}].", LOG)

    model_path = os.path.join(MODEL_DIR, f"{name}_model.joblib")
    vec_path = os.path.join(MODEL_DIR, f"{name}_vectorizer.joblib")

    # Save model
    try:
        joblib.dump(model, model_path)
        utils.info(f"save_model_and_vectorizer: Model saved - {model_path}", LOG)
    except Exception as err:
        utils.error(f"save_model_and_vectorizer: Failed to save model ({err}).", LOG)
        return False

    # Save vectorizer
    try:
        joblib.dump(vectorizer, vec_path)
        utils.info(f"save_model_and_vectorizer: Vectorizer saved - {vec_path}", LOG)
    except Exception as err:
        utils.error(f"save_model_and_vectorizer: Failed to save vectorizer ({err}).", LOG)
        return False

    return True

def save_evaluation_report(metrics, name):
    """
    Save evaluation metrics to a JSON file.

    Args:
        metrics (dict): Evaluation metrics returned by evaluate_model().
        name (str): Base name of the model (e.g., "logistic_regression").

    Returns:
        bool: True on success, False on failure.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_metrics_{timestamp}.json"
    output_path = os.path.join(OUTPUT_DIR, filename)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        utils.info(f"save_evaluation_report: Saved metrics → {output_path}", LOG)
        return True

    except Exception as err:
        utils.error(f"save_evaluation_report: Failed to save metrics ({err}).", LOG)
        return False

def main():
    """
    Full ML training pipeline:
        1. Load processed data
        2. Split into train/test
        3. Vectorize text
        4. Train Logistic Regression and Naive Bayes
        5. Evaluate both models
        6. Save both models + vectorizer

    Returns:
        int: 0 on success, 1 on failure.
    """

    # Load data
    x, y = load_model_data()
    if x is None or y is None:
        return utils.error("main: Failed to load training data.", LOG) or 1

    # Train/test split
    split = split_data(x, y)
    if split[0] is None:
        return utils.error("main: Train/test split failed.", LOG) or 1

    x_train, x_test, y_train, y_test = split

    # Vectorize text
    vec_result = vectorize_text(x_train, x_test)
    if vec_result[0] is None:
        return utils.error("main: Vectorization failed.", LOG) or 1

    x_train_vec, x_test_vec, vectorizer = vec_result

    # Train models
    model_lr = train_logistic_regression(x_train_vec, y_train)
    if model_lr is None:
        return utils.error("main: Logistic Regression training failed.", LOG) or 1

    model_nb = train_naive_bayes(x_train_vec, y_train)
    if model_nb is None:
        return utils.error("main: Naive Bayes training failed.", LOG) or 1

    # Evaluate models
    metrics_lr = evaluate_model(model_lr, x_test_vec, y_test, model_type="Logistic Regression")
    metrics_nb = evaluate_model(model_nb, x_test_vec, y_test, model_type="Naive Bayes")

    # Save evaluation reports
    if save_evaluation_report(metrics_lr, "logistic_regression"):
        utils.info("main: Saved Logistic Regression evaluation report.", LOG)
    else:
        return utils.error("main: Failed to save Logistic Regression evaluation report.", LOG) or 1

    if save_evaluation_report(metrics_nb, "naive_bayes"):
        utils.info("main: Saved Naive Bayes evaluation report.", LOG)
    else:
        return utils.error("main: Failed to save Naive Bayes evaluation report.", LOG) or 1

    # Save artifacts 
    if save_model_and_vectorizer(model_lr, vectorizer, "logistic_regression"):
        utils.info("main: Saved Logistic Regression model and vectorizer.", LOG)
    else:
        return utils.error("main: Failed to save Logistic Regression artifacts.", LOG) or 1

    if save_model_and_vectorizer(model_nb, vectorizer, "naive_bayes"):
        utils.info("main: Saved Naive Bayes model and vectorizer.", LOG)
    else:
        return utils.error("main: Failed to save Naive Bayes artifacts.", LOG) or 1

    utils.info("main: Model training pipeline completed successfully.", LOG)
    return 0

if __name__ == "__main__":
    main()