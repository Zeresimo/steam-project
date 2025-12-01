import streamlit as st
import pandas as pd
import sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))            # ML/
ROOT = os.path.dirname(ROOT)                                 # steam-project/
sys.path.insert(0, ROOT)

from ML.predict import predict_review 

st.set_page_config(
    page_title="Steam Review Analyzer",
    layout="wide"
)

st.title("🎮 Steam Review Sentiment Analyzer")
st.write("Welcome to the dashboard. Use the sidebar to navigate.")

st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Go to:",
    ["Home", "EDA Visualizations", "Single Review Prediction", "Batch Prediction"]
)

if page == "Home":
    st.header("📌 Project Overview")
    st.write("""
    Welcome to the Steam Review Sentiment Analyzer Dashboard!

    **Features you can use:**
    - 📊 View EDA visualizations from your processed data
    - 🔮 Predict sentiment for a single text review
    - 📁 Upload TXT or CSV files for batch sentiment analysis
    - 💾 Download prediction results as CSV
    - 🔧 Choose which ML model to use (Logistic Regression or Naive Bayes)

    Navigate using the sidebar to explore each feature.
    """)

elif page == "EDA Visualizations":
    st.header("📊 EDA Visualizations")

    st.write("Below are the exploratory data analysis plots generated from your processed dataset.")

    # Define directories
    plot_dirs = [
    os.path.join(ROOT, "EDA", "plots"),
    os.path.join(ROOT, "EDA", "wordclouds")
    ]

    for directory in plot_dirs:
        st.subheader(f"📁 Folder: {directory}")

        if not os.path.exists(directory):
            st.warning(f"Directory not found: {directory}")
            continue

        images = [img for img in os.listdir(directory) if img.endswith((".png", ".jpg", ".jpeg"))]

        if not images:
            st.info("No images found in this directory.")
            continue

        for image in images:
            image_path = os.path.join(directory, image)
            st.image(image_path, caption=image, width="container")

        st.markdown("---")  # Divider between folders

elif page == "Single Review Prediction":
    st.header("🔮 Single Review Sentiment Prediction")

    st.write("Type a Steam review below and choose a model to classify the sentiment.")

    review_text = st.text_area("Enter review text:", height=200)

    model_choice = st.selectbox(
        "Select model:",
        ["logistic_regression", "naive_bayes"]
    )

    if st.button("Predict Sentiment"):
        if review_text.strip() == "":
            st.warning("Please enter a review before predicting.")
        else:
            result = predict_review(review_text, model_choice)

            if result is None:
                st.error("Prediction failed. Check logs.")
            else:
                st.success("Prediction complete!")

                st.write("### 📝 Cleaned Text")
                st.write(result["cleaned_text"])

                st.write("### 🎯 Sentiment")
                st.write(f"**{result['sentiment']}**")

                st.write("### 📊 Confidence Score")
                st.write(result["confidence"])

elif page == "Batch Prediction":
    st.header("📁 Batch Review Prediction")

    st.write("Upload a TXT or CSV file containing reviews. The model will classify each review and return a downloadable CSV.")

    uploaded_file = st.file_uploader("Upload your file", type=["txt", "csv"])

    model_choice = st.selectbox(
        "Select model:",
        ["logistic_regression", "naive_bayes"]
    )

    if uploaded_file is not None:
        st.info(f"Uploaded file: **{uploaded_file.name}**")

        # Save temporarily
        temp_ext = uploaded_file.name.split(".")[-1]
        temp_path = os.path.join(ROOT, "temp_upload." + temp_ext)

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # We will rewrite run_batch_prediction to accept a direct file path
        from ML.predict import run_batch_prediction

        st.write("Processing batch prediction...")

        result_path = run_batch_prediction(model_choice, override_input_file=temp_path)

        if result_path:
            st.success("Batch processing complete! 🎉")

            # Show results in a table
            df_results = pd.read_csv(result_path)
            st.dataframe(df_results)

            # Download button
            with open(result_path, "rb") as f:
                st.download_button(
                    label="Download Results CSV",
                    data=f,
                    file_name=os.path.basename(result_path),
                    mime="text/csv"
                )
        else:
            st.error("Batch prediction failed. Check logs for details.")
