import os
import sys
import streamlit as st
import pandas as pd


# Ensure project root is available in sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from paths import ROOT, EDA_PLOTS, EDA_WORDCLOUDS
from Dashboard.api_bridge import (
    api_search_games, 
    api_fetch_reviews, 
    api_clean_and_predict
)
from ML.predict import load_model, predict_review, run_batch_prediction

from Pipeline.utils import utils

# Streamlit Page Configuration
st.set_page_config(
    page_title="Steam Review Analyzer",
    layout="wide"
)

# App Header
st.title("🎮 Steam Review Sentiment Analyzer")
st.write("Welcome to the dashboard. Use the sidebar to navigate.")

# Sidebar Navigation
st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Go to:",
    [
    "Home", 
    "EDA Visualizations", 
    "Single Review Prediction", 
    "Batch Prediction", 
    "Fetch Live Reviews"
    ]
)

# Home Page
if page == "Home":
    st.header("📌 Project Overview")

    st.markdown(
        """
Welcome to the **Steam Review Sentiment Analyzer Dashboard**!

This tool allows you to explore and analyze Steam game reviews using:
- 📊 **EDA visualizations** generated from your processed review dataset  
- 🔮 **Single review sentiment prediction**  
- 📁 **Batch prediction** from TXT or CSV files  
- 🌐 **Live Steam review fetching + prediction**  
- 🤖 **Model selection** (Logistic Regression or Naive Bayes)

Use the sidebar to navigate through each feature.
        """
    )

# EDA Visualizations
elif page == "EDA Visualizations":
    st.header("📊 EDA Visualizations")
    st.write("Below are all the EDA plots generated from your processed dataset.")

    plot_dirs = {
        "Standard Plots": EDA_PLOTS,
        "Word Clouds": EDA_WORDCLOUDS
    }

    for label, directory in plot_dirs.items():
        st.subheader(f"📁 {label} — `{directory}`")

        # Directory exists?
        if not os.path.exists(directory):
            st.warning(f"Directory not found: `{directory}`")
            st.markdown("---")
            continue

        # List images
        images = [
            img for img in os.listdir(directory)
            if img.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not images:
            st.info("No images found in this folder.")
            st.markdown("---")
            continue

        # Sort images alphabetically for consistent display
        images.sort()

        # Display images
        for img in images:
            img_path = os.path.join(directory, img)
            st.image(img_path, caption=img, use_column_width=True)

        st.markdown("---")

# SINGLE REVIEW PREDICTION
elif page == "Single Review Prediction":
    st.header("🔮 Single Review Sentiment Prediction")
    st.write("Enter a Steam review below and choose a model to classify its sentiment.")

    review_text = st.text_area("Enter review text:", height=180, placeholder="Type or paste a review...")

    model_choice = st.selectbox(
        "Select model:",
        ["logistic_regression", "naive_bayes"]
    )

    if st.button("Predict Sentiment"):
        # Validate input
        if not review_text.strip():
            st.warning("Please enter a non-empty review before predicting.")
            st.stop()

        # Run prediction (using cleaned predict_review)
        result = predict_review(review_text, model_choice)

        if result is None:
            st.error("Prediction failed. Check logs for details.")
            st.stop()

        # Display Results
        st.success("Prediction Complete! 🎉")

        st.markdown("### 📝 Cleaned Text")
        st.write(result["cleaned_text"])

        st.markdown("### 🎯 Sentiment")
        sentiment_color = "🟩 Positive" if result["sentiment"] == "Positive" else "🟥 Negative"
        st.write(f"**{sentiment_color}**")

        st.markdown("### 📊 Confidence Score")
        conf = result["confidence"]
        st.write(conf if conf is not None else "N/A")


# BATCH PREDICTION
elif page == "Batch Prediction":
    st.header("📁 Batch Review Prediction")
    st.write("Upload a **TXT** or **CSV** file containing multiple reviews. The model will classify each one and produce a downloadable CSV output.")

    uploaded_file = st.file_uploader("Upload your file", type=["txt", "csv"])

    model_choice = st.selectbox(
        "Select model:",
        ["logistic_regression", "naive_bayes"]
    )

    if uploaded_file is not None:
        st.info(f"Uploaded file: **{uploaded_file.name}**")

        # Save uploaded file to a temporary location
        file_ext = uploaded_file.name.split(".")[-1]
        temp_path = os.path.join(ROOT, f"temp_upload.{file_ext}")

        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except Exception as err:
            st.error("Failed to save the uploaded file.")
            utils.error(f"Batch Prediction: Error saving temporary file ({err})", None)
            st.stop()

        st.write("Processing batch prediction...")

        # Run prediction
        try:
            result_path = run_batch_prediction(model_choice, override_input_file=temp_path)
        except Exception as err:
            st.error("Batch prediction failed. Check logs for details.")
            utils.error(f"Batch Prediction: run_batch_prediction crashed ({err})", None)
            st.stop()

        if result_path is None:
            st.error("Batch prediction failed. Check logs for details.")
            st.stop()

        # Load results and display
        try:
            df_results = pd.read_csv(result_path)
        except Exception as err:
            st.error("Failed to load generated results CSV.")
            utils.error(f"Batch Prediction: Error reading results CSV ({err})", None)
            st.stop()

        st.success("Batch processing complete! 🎉")
        st.dataframe(df_results, use_container_width=True)

        # Download button
        with open(result_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Results CSV",
                data=f,
                file_name=os.path.basename(result_path),
                mime="text/csv"
            )

        os.remove(temp_path)


# FETCH LIVE STEAM REVIEWS
elif page == "Fetch Live Reviews":
    st.header("🌐 Fetch Live Steam Reviews")

    # Select Model
    st.subheader("1. Select ML Model")

    model_name = st.selectbox(
        "Choose a model:",
        ["logistic_regression", "naive_bayes"]
    )

    model, vectorizer = load_model(model_name)
    if model is None or vectorizer is None:
        st.error("Failed to load the selected model. Check logs.")
        st.stop()

    # Search for a Game
    st.subheader("2. Search for a Steam Game")
    query = st.text_input("Enter game name:")

    if st.button("Search"):
        results = api_search_games(query)

        if not results:
            st.warning("No results found. Try a different search.")
            st.session_state["game_results"] = []
        else:
            st.session_state["game_results"] = results

    # Game Selection
    if st.session_state.get("game_results"):
        results = st.session_state["game_results"]
        choices = [f"{g['name']} (id: {g['id']})" for g in results]

        st.subheader("3. Select a Game")
        selected_label = st.selectbox("Choose game:", choices)

        # Map selection back to dict
        selected_game = results[choices.index(selected_label)]
        appid = selected_game["id"]
        game_name = selected_game["name"]

        # Choose Number of Reviews
        st.subheader("4. Choose Number of Reviews")
        limit = st.number_input(
            "Number of reviews to fetch:",
            min_value=10,
            max_value=2000,
            value=200,
            step=10
        )

        # Fetch + Predict
        if st.button("Fetch Reviews"):
            with st.spinner("Fetching reviews from Steam..."):
                reviews = api_fetch_reviews(appid, limit)

            if reviews is None:
                st.error("Failed to fetch reviews. Check logs.")
                st.stop()

            if len(reviews) == 0:
                st.warning("No reviews returned from Steam.")
                st.stop()

            with st.spinner("Cleaning + Running Predictions..."):
                df = api_clean_and_predict(
                    reviews, model, vectorizer, appid, game_name
                )

            if df is None or df.empty:
                st.warning("No clean reviews available for prediction.")
                st.stop()

            st.success(f"Fetched {len(df)} reviews for **{game_name}**!")

            # Display results
            st.dataframe(df, use_container_width=True)

            # CSV download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"live_reviews_{appid}.csv",
                mime="text/csv"
            )

    else:
        # Only show this if user has already attempted a search
        if query.strip():
            st.info("Search for a game to continue.")