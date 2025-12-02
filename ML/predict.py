import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from paths import MODEL_DIR, LOG_DIR, OUTPUT_DIR
from Pipeline.utils.utils import clean_text, log, ensure_directory

LOG = {
    "base_path": LOG_DIR + "/",
    "filename": "ML_predict_log.txt"
}


paths = [LOG_DIR, MODEL_DIR, OUTPUT_DIR]



ensure_directory(paths)

def load_model(name):
    file_path = MODEL_DIR
    vector_name = name + "_vectorizer.joblib"
    model_name = name + "_model.joblib"
    model_path = os.path.join(file_path, model_name )
    vector_path = os.path.join(file_path, vector_name)

    try:
        log(f"Loading {name} model", level="INFO", **LOG)
        model = joblib.load(model_path)
        log(f"Successfully loaded {name} model", level="INFO", **LOG)
        
    except Exception as e:
        log(f"Error loading {name} model", level="ERROR", **LOG)
        return None, None

    try:
        log(f"Loading {name} vectorizer", level="INFO", **LOG)
        vector = joblib.load(vector_path)
        log(f"Successfully loaded {name} vectorizer", level="INFO", **LOG)

    except Exception as e:
        log(f"Error loading {name} vectorizer", level="ERROR", **LOG)
        return None, None
    
    return model, vector

def predict_review(text, model_name):
    model, vectorizer = load_model(model_name)
    if not model:
        log("Model was not loaded properly", level="ERROR", **LOG)
        return
    
    if not vectorizer:
        log("Vectorizer was not loaded properly", level="ERROR", **LOG)
        return

    log(f"Running prediction using {model_name}", level="INFO", **LOG)
    cleaned = clean_text(text)
    
    try:
        vectorized = vectorizer.transform([cleaned])
        log(f"Vectorize successful", level="INFO", **LOG)
    except Exception as e:
        log(f"Vectorize unsuccessful: {e}", level="ERROR", **LOG)
        return

    try:
        prediction = model.predict(vectorized)[0]
        if prediction == 1:
            prediction = "Positive"
        else:
            prediction = "Negative"
        log(f"Prediction successful: {prediction}", level="INFO", **LOG)

    except Exception as e:
        log(f"Prediction unsuccessful: {e}", level="ERROR", **LOG)
        return
    
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(vectorized)[0][1]
        confidence = float(confidence)
        log(f"Confidence Score: {confidence}", level="INFO", **LOG)

    else:
        confidence = None

    return {
        "cleaned_text": cleaned,
        "sentiment": prediction,
        "confidence": confidence
    }

def run_single_prediction_loop(model_name):

    while True:
        text = input("Enter review (or 'back'): ")

        if text.lower() == "back":
            break

        result = predict_review(text, model_name)

        if result is None:
            print("Prediction failed. Check logs.")
            continue

        print("Sentiment:", result['sentiment'])
        print("Confidence:", result['confidence'])

def run_batch_prediction(model_name, override_input_file=None):
    if override_input_file is not None:
        file_path = override_input_file
    else:
        file_path = input("Enter the file path (.txt or .csv): ").strip()

    # Validate path
    if not os.path.isfile(file_path):
        print("File not found. Please check the path.")
        log(f"Batch file not found: {file_path}", level="ERROR", **LOG)
        return

    log(f"Starting batch prediction using {model_name}", level="INFO", **LOG)
    print("\n----- Batch Prediction Results -----\n")

    results = []

    # Case 1: CSV file
    if file_path.lower().endswith(".csv"):

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print("Could not read CSV file.")
            log(f"Error reading CSV file: {e}", level="ERROR", **LOG)
            return

        if "review" not in df.columns:
            print("CSV must contain a 'review' column.")
            log("CSV missing 'review' column.", level="ERROR", **LOG)
            return

        reviews = df["review"].dropna().tolist()

        for idx, review in enumerate(reviews):
            result = predict_review(review, model_name)

            if result is None:
                print("Prediction failed. Check logs.")
                continue

            conf = result['confidence'] if result['confidence'] is not None else "N/A"
            print(f"Row {idx+1}: {result['sentiment']} (Confidence={conf})")

            results.append({
                "original_text": review,
                "cleaned_text": result["cleaned_text"],
                "sentiment": result["sentiment"],
                "confidence": result["confidence"]
            })

    # Case 2: TXT file
    elif file_path.lower().endswith(".txt"):

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
        except Exception as e:
            print("Error reading TXT file.")
            log(f"Error reading TXT file: {e}", level="ERROR", **LOG)
            return

        for idx, line in enumerate(lines):
            review = line.strip()

            if review == "":
                continue  # skip blank lines

            result = predict_review(review, model_name)

            if result is None:
                print(f"Line {idx+1}: Prediction failed. Check logs.")
                continue

            conf = result['confidence'] if result['confidence'] is not None else "N/A"
            print(f"Line {idx+1}: {result['sentiment']} (Confidence={conf})")

            results.append({
                "original_text": review,
                "cleaned_text": result["cleaned_text"],
                "sentiment": result["sentiment"],
                "confidence": result["confidence"]
            })

    else:
        print("Unsupported file type. Use .txt or .csv files.")
        log(f"Unsupported batch file type: {file_path}", level="ERROR", **LOG)
        return

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_filename = os.path.join(
    OUTPUT_DIR,
    f"batch_{model_name}_{timestamp}.csv"
    )

    save_batch_results(results, output_filename)

    print(f"\nBatch prediction complete! Saved results to:\n{output_filename}")
    log(f"Batch prediction completed. {len(results)} reviews processed.", level="INFO", **LOG)

    return output_filename


def save_batch_results(results, filename):
    try:
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        log(f"Batch results saved successfully: {filename}", level="INFO", **LOG)
    except Exception as e:
        log(f"Error saving batch results: {e}", level="ERROR", **LOG)

if __name__ == "__main__":
     while True:
        print("\n===== Steam Review Sentiment Analyzer =====")
        print("Choose a model:")
        print("1. Logistic Regression")
        print("2. Naive Bayes")
        print("3. Quit")

        choice = input("Enter your choice: ").strip()

        match choice:
            case "1":
                selected_model = "logistic_regression"
            case "2":
                selected_model = "naive_bayes"
            case "3":
                print("Exiting program.")
                break
            case _:
                print("Invalid option. Try again.")
                continue

        # Model selected — now choose prediction mode
        while True:
            print(f"\nModel selected: {selected_model.replace('_',' ').title()}")
            print("Choose prediction mode:")
            print("1. Single Review Prediction")
            print("2. Batch Prediction from File")
            print("3. Back to Model Selection")

            mode_choice = input("Enter your choice: ").strip()

            match mode_choice:
                case "1":
                    run_single_prediction_loop(selected_model)

                case "2":
                    run_batch_prediction(selected_model)

                case "3":
                    print("Returning to model selection...")
                    break

                case _:
                    print("Invalid option. Try again.")

