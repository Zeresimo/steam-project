import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from paths import MODEL_DIR, LOG_DIR, OUTPUT_DIR
from Pipeline.utils import utils

LOG = {
    "base_path": LOG_DIR + "/",
    "filename": "ML_predict_log.txt"
}

paths = [LOG_DIR, MODEL_DIR, OUTPUT_DIR]
utils.ensure_directory(paths)

def load_model(name):
    """
    Load a saved model and its vectorizer from disk.

    Args:
        name (str): Base model name (e.g., 'logistic_regression').

    Returns:
        (model, vectorizer) or (None, None) on failure.
    """

    model_path = os.path.join(MODEL_DIR, f"{name}_model.joblib")
    vector_path = os.path.join(MODEL_DIR, f"{name}_vectorizer.joblib")

    # Load model
    try:
        utils.info(f"load_model: Loading model [{name}].", LOG)
        model = joblib.load(model_path)
    except Exception as err:
        return utils.error2(f"load_model: Failed to load model ({err}).", LOG)

    # Load vectorizer
    try:
        utils.info(f"load_model: Loading vectorizer [{name}].", LOG)
        vectorizer = joblib.load(vector_path)
    except Exception as err:
        return utils.error2(f"load_model: Failed to load vectorizer ({err}).", LOG)

    utils.info(f"load_model: Successfully loaded model + vectorizer [{name}].", LOG)
    return model, vectorizer

def predict_review(text, model_name):
    """
    Predict the sentiment of a single review.

    Args:
        text (str): Raw review text.
        model_name (str): Model base name (e.g. 'logistic_regression').

    Returns:
        dict | None:
            {
                "cleaned_text": str,
                "sentiment": "Positive" | "Negative",
                "confidence": float | None
            }
    """

    model, vectorizer = load_model(model_name)
    if model is None or vectorizer is None:
        return utils.error("predict_review: Model or vectorizer not loaded.", LOG)

    utils.info(f"predict_review: Running prediction with [{model_name}].", LOG)

    # Clean review text
    cleaned = utils.clean_text(text)

    # Vectorize
    try:
        vectorized = vectorizer.transform([cleaned])
        utils.info("predict_review: Vectorization successful.", LOG)
    except Exception as err:
        return utils.error(f"predict_review: Vectorization failed ({err}).", LOG)

    # Predict label
    try:
        pred_label = model.predict(vectorized)[0]
        sentiment = "Positive" if pred_label == 1 else "Negative"
        utils.info(f"predict_review: Prediction = {sentiment}.", LOG)
    except Exception as err:
        return utils.error(f"predict_review: Prediction failed ({err}).", LOG)

    # Compute confidence (if available)
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            confidence = float(model.predict_proba(vectorized)[0][1])
            utils.info(f"predict_review: Confidence = {confidence:.4f}.", LOG)
        except Exception:
            utils.info("predict_review: Confidence unavailable.", LOG)

    return {
        "cleaned_text": cleaned,
        "sentiment": sentiment,
        "confidence": confidence
    }

def run_single_prediction_loop(model_name):
    """
    Interactive loop for predicting sentiment on individual user-entered reviews.

    Args:
        model_name (str): Model base name (e.g. 'logistic_regression').

    Returns:
        None
    """

    while True:
        text = input("Enter review (or 'back'): ").strip()

        if text.lower() == "back":
            return

        if text == "":
            print("Please enter a non-empty review.")
            continue

        result = predict_review(text, model_name)

        if result is None:
            print("Prediction failed. Check logs.")
            utils.error("run_single_prediction_loop: Prediction returned None.", LOG)
            continue

        sentiment = result["sentiment"]
        confidence = result["confidence"]

        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence if confidence is not None else 'N/A'}")

def run_batch_prediction(model_name, override_input_file=None):
    """
    Run sentiment prediction on a batch of reviews from a .txt or .csv file.

    Args:
        model_name (str): Base name of the model.
        override_input_file (str | None): If provided, bypass user input 
                                          and load the specified file.

    Returns:
        str | None: Output CSV filename or None on failure.
    """

    
    # Get file path
    file_path = override_input_file or input("Enter the file path (.txt or .csv): ").strip()

    if not os.path.isfile(file_path):
        utils.error(f"run_batch_prediction: File not found → {file_path}", LOG)
        print("File not found. Check the path.")
        return None

    utils.info(f"run_batch_prediction: Starting batch prediction with [{model_name}].", LOG)
    print("\n----- Batch Prediction Results -----\n")

    results = []

    # Handle CSV files
    if file_path.lower().endswith(".csv"):

        try:
            df = pd.read_csv(file_path)
        except Exception as err:
            utils.error(f"run_batch_prediction: Failed to read CSV ({err}).", LOG)
            print("Could not read CSV file.")
            return None

        if "review" not in df.columns:
            utils.error("run_batch_prediction: CSV missing 'review' column.", LOG)
            print("CSV must contain a 'review' column.")
            return None

        reviews = df["review"].dropna().tolist()

        for idx, review in enumerate(reviews):
            result = predict_review(review, model_name)

            if result is None:
                print(f"Row {idx+1}: Prediction failed.")
                utils.error("run_batch_prediction: Prediction returned None.", LOG)
                continue

            sentiment = result["sentiment"]
            conf = result["confidence"] if result["confidence"] is not None else "N/A"

            print(f"Row {idx+1}: {sentiment} (Confidence={conf})")

            results.append({
                "original_text": review,
                "cleaned_text": result["cleaned_text"],
                "sentiment": sentiment,
                "confidence": result["confidence"]
            })

    
    # Handle TXT files
    elif file_path.lower().endswith(".txt"):

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as err:
            utils.error(f"run_batch_prediction: Failed to read TXT ({err}).", LOG)
            print("Error reading TXT file.")
            return None

        for idx, line in enumerate(lines):
            review = line.strip()

            if not review:
                continue  # Skip blank lines

            result = predict_review(review, model_name)

            if result is None:
                print(f"Line {idx+1}: Prediction failed.")
                utils.error("run_batch_prediction: Prediction returned None.", LOG)
                continue

            sentiment = result["sentiment"]
            conf = result["confidence"] if result["confidence"] is not None else "N/A"

            print(f"Line {idx+1}: {sentiment} (Confidence={conf})")

            results.append({
                "original_text": review,
                "cleaned_text": result["cleaned_text"],
                "sentiment": sentiment,
                "confidence": result["confidence"]
            })

    
    # Unsupported file extension
    else:
        utils.error(f"run_batch_prediction: Unsupported file type → {file_path}", LOG)
        print("Unsupported file type. Use .txt or .csv.")
        return None

    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(
        OUTPUT_DIR,
        f"batch_{model_name}_{timestamp}.csv"
    )

    save_batch_results(results, output_filename)

    utils.info(
        f"run_batch_prediction: Completed. {len(results)} reviews processed.",
        LOG
    )

    print(f"\nBatch prediction complete! Saved results to:\n{output_filename}")

    return output_filename

def save_batch_results(results, filename):
    """
    Save batch prediction results to a CSV file.

    Args:
        results (list of dict): Each dict contains:
            - original_text
            - cleaned_text
            - sentiment
            - confidence
        filename (str): Output CSV path.

    Returns:
        bool: True on success, False on failure.
    """

    # If empty, warn but still write an empty file
    if not results:
        utils.info("save_batch_results: No results to save (writing empty file).", LOG)

    try:
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        utils.info(f"save_batch_results: Saved → {filename}", LOG)
        return True

    except Exception as err:
        utils.error(f"save_batch_results: Failed to save file ({err}).", LOG)
        return False

def main():
    """
    CLI interface for running sentiment predictions using trained models.
    
    Workflow:
        1. Choose model (LR or NB)
        2. Choose prediction mode (single or batch)
        3. Run predictions
    """

    while True:
        print("\n===== Steam Review Sentiment Analyzer =====")
        print("Choose a model:")
        print("1. Logistic Regression")
        print("2. Naive Bayes")
        print("3. Quit")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            model_name = "logistic_regression"
        elif choice == "2":
            model_name = "naive_bayes"
        elif choice == "3":
            print("Exiting program.")
            return
        else:
            print("Invalid option. Try again.")
            continue

        # Model chosen - choose mode
        while True:
            print(f"\nModel selected: {model_name.replace('_',' ').title()}")
            print("Choose prediction mode:")
            print("1. Single Review Prediction")
            print("2. Batch Prediction from File")
            print("3. Back to Model Selection")

            mode_choice = input("Enter your choice: ").strip()

            if mode_choice == "1":
                run_single_prediction_loop(model_name)

            elif mode_choice == "2":
                run_batch_prediction(model_name)

            elif mode_choice == "3":
                print("Returning to model selection...")
                break

            else:
                print("Invalid option. Try again.")


if __name__ == "__main__":
    main()