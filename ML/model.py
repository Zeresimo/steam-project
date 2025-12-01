import pandas as pd
import numpy as np
import joblib

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(ROOT, "ML")
MODEL_DIR = os.path.join(ML_DIR, "models")
LOG_DIR = os.path.join(ML_DIR, "logs")
OUTPUT_DIR = os.path.join(ML_DIR, "output")





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

from Pipeline.utils.utils import (
    get_latest_csv,
    log,
    ensure_directory,
    clean_text,
    validate_column_type
)

LOG = {
    "base_path": LOG_DIR + "/",
    "filename": "ML_train_log.txt"
}

paths = [LOG_DIR, MODEL_DIR, OUTPUT_DIR]

ensure_directory(paths)

def load_model_data():
    latest_file = get_latest_csv(LOG, base_path="Pipeline/data/processed")

    if not latest_file:
        log("No file found for modeling", level="ERROR", **LOG)
        return None, None
    
    try:
        log(f"Latest processed CSV file found: {latest_file}", level = "INFO", **LOG)
        df = pd.read_csv(latest_file)
        log(f"File read successfully, processing", level = "INFO", **LOG)
        rows, columns = df.shape
        log(f"Loaded processed dataset containing {rows} rows and {columns} columns", level = "INFO", **LOG)
    except Exception as e:
        log(f"Error reading CSV file: {e}", level = "ERROR", **LOG)
        return None, None

    df = df.dropna(subset=['review'])

    if not validate_column_type(df, "review", (str,)):
        log("Review column validation error", level="ERROR", **LOG)
        return None, None
    if not validate_column_type(df, "voted_up", (bool, np.bool_)):
        log("Voted_up column validation error", level="ERROR", **LOG)
        return None, None

    df['review'] = df['review'].apply(clean_text)
    df = df.dropna(subset=['review'])
    log("Secondary Review text cleaning completed.", level = "INFO", **LOG)

    df['voted_up'] = df['voted_up'].astype(int)

    x = df["review"]
    y = df["voted_up"]

    return x, y

def split_data(x, y):
    log("Starting train/test split", level="INFO", **LOG)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    log("Train/test split successful", level="INFO", **LOG)

    log(f"Training Set Size {len(x_train)}", level="INFO", **LOG)
    log(f"Testing Set Size {len(x_test)}", level="INFO", **LOG)

    log(f"Training Level Distribution: {y_train.value_counts()}", level="INFO", **LOG)
    log(f"Testing Level Distribution: {y_test.value_counts()}", level="INFO", **LOG)

    return x_train, x_test, y_train, y_test

def vectorize_text(x_train, x_test):
    log("Starting TF-IDF vectorization", level="INFO", **LOG)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)
    vectorizer.fit(x_train)

    x_train_vector = vectorizer.transform(x_train)
    x_test_vector = vectorizer.transform(x_test)

    log(f"Vocabulary Size: {len(vectorizer.vocabulary_)}", level="INFO", **LOG)

    train_shape_x, train_shape_y = x_train_vector.shape
    test_shape_x, test_shape_y = x_test_vector.shape

    log(f"Shape of training vector: {train_shape_x} rows, {train_shape_y} columns", level="INFO", **LOG)
    log(f"Shape of test vector: {test_shape_x} rows, {test_shape_y} columns", level="INFO", **LOG)
    log("TF-IDF vectorization completed successfully", level="INFO", **LOG)

    return x_train_vector, x_test_vector, vectorizer

def train_logistic_regression(x_train_vec, y_train):
    log("Starting training using logistic regression", level="INFO", **LOG)

    log_reg_model = LogisticRegression(max_iter=2000,solver="liblinear")
    log_reg_model.fit(x_train_vec, y_train)

    log("Logistic Regression training completed", level="INFO", **LOG)
    log(f"Model classes: {log_reg_model.classes_}", level="INFO", **LOG)

    return log_reg_model

def train_naive_bayes(x_train_vec, y_train):
    log("Training Multinomial Naive Bayes model", level="INFO", **LOG)

    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(x_train_vec, y_train)

    log("Naive Bayes training completed", level="INFO", **LOG)

    return naive_bayes_model

def evaluate_model(model, x_test_vec, y_test, model_type="NONE"):
    report_dict = {}
    log(f"Starting model evaluation for {model_type}", level="INFO", **LOG)

    y_prediction = model.predict(x_test_vec)

    report_dict['accuracy'] = accuracy_score(y_test, y_prediction)
    report_dict['precision'] = precision_score(y_test, y_prediction)
    report_dict['recall'] = recall_score(y_test, y_prediction)
    report_dict['f1'] = f1_score(y_test, y_prediction)

    log(
    f"Metrics produced: Accuracy: {report_dict['accuracy']}, "
    f"Precision: {report_dict['precision']}, "
    f"Recall: {report_dict['recall']}, "
    f"F1: {report_dict['f1']}",
    level="INFO",
    **LOG
    )

    report = classification_report(y_test, y_prediction)
    log("Classification Report: \n" + report, level="INFO", **LOG)

    report_dict['conf_matrix'] = confusion_matrix(y_test, y_prediction)
    log("Confusion Matrix:\n" + str(report_dict['conf_matrix']), level="INFO", **LOG)
    report_dict['conf_matrix'] = report_dict['conf_matrix'].tolist()

    if hasattr(model, "predict_proba"):
        y_probability = model.predict_proba(x_test_vec)[:, 1]
        report_dict['roc_auc'] = roc_auc_score(y_test, y_probability)
        log(f"ROC AUC: {report_dict['roc_auc']}", level="INFO", **LOG)

    else:
        log("ROC AUC skipped: this model does not support probability predictions", level="INFO", **LOG)


    return report_dict

def save_model_and_vectorizer(model, vectorizer, name):
    log(f"Saving model and vectorizer for {name}", level="INFO", **LOG)

    filename = name
    file_path = MODEL_DIR
    vec_path = os.path.join(file_path, filename + "_vectorizer.joblib")
    model_path = os.path.join(file_path, filename + "_model.joblib")

    try:
        joblib.dump(model, model_path)
        log(f"Saved model: {model_path}", level="INFO", **LOG)

    except Exception as e:
        log(f"Error saving model: {e}", level="ERROR", **LOG)
        return False

    try:
        joblib.dump(vectorizer, vec_path)
        log(f"Saved vectorizer: {vec_path}", level="INFO", **LOG)
    except Exception as e:
        log(f"Error saving vectorizer: {e}", level="ERROR", **LOG)
        return False
    
    return True

def main():
    x, y = load_model_data()
    x_train, x_test, y_train, y_test = split_data(x, y)
    train_vector, test_vector, vectorizer = vectorize_text(x_train, x_test)
    model_lr = train_logistic_regression(train_vector, y_train)
    model_nb = train_naive_bayes(train_vector, y_train)

    metrics_lr = evaluate_model(model_lr, test_vector, y_test, model_type="Logistic Regression")
    metrics_nb = evaluate_model(model_nb, test_vector, y_test, model_type="Naive Bayes")

    if save_model_and_vectorizer(model_lr, vectorizer, "logistic_regression"):
        log("Successfully saved Logistic Regression model and vectorizer", level="INFO", **LOG)
    else:
        log("Saving failed for Logistic Regression model and vectorizer", level="ERROR", **LOG)

    if save_model_and_vectorizer(model_nb, vectorizer, "naive_bayes"):
        log("Successfully saved Naive Bayes model and vectorizer", level="INFO", **LOG)
    else:
        log("Saving failed for Naive Bayes model and vectorizer", level="ERROR", **LOG)

if __name__ == "__main__":
    main()