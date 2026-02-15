import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import logging

# -----------------------------
# Suppress Streamlit warnings
# -----------------------------
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)

# -----------------------------
# Load the trained model
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "sentiment_model.joblib"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please upload 'sentiment_model.joblib' first.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Sentiment Rating App", page_icon="💬", layout="centered")

st.title("💬 Sentiment Rating App (XGBoost + TF-IDF)")
st.markdown("""
Enter a **Title** and **Body** to predict sentiment.
The color bar fills **based on sentiment only**:
- 🔴 Negative (Red)
- 🟠 Neutral (Orange)
- 🟢 Positive (Green)
""")

st.divider()

# -----------------------------
# Input Section
# -----------------------------
title_input = st.text_input("📰 Enter Subject:")
body_input = st.text_area("📝 Enter Description:", height=180)

# -----------------------------
# Prediction Section
# -----------------------------
if st.button("🔍 Predict Sentiment"):
    if not title_input or not body_input:
        st.warning("⚠️ Please enter both Title and Body.")
    else:
        # Prepare input
        input_df = pd.DataFrame({'title': [title_input], 'body': [body_input]})

        # Predict
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0] * 100

        sentiment_labels = {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}
        sentiment_text = sentiment_labels[prediction]
        confidence = probabilities[prediction]

        # -----------------------------
        # Fixed bar color and fill by sentiment
        # -----------------------------
        if prediction == 0:  # Negative
            fill_percent = 40
            bar_color = "#FF4B4B"  # Red
        elif prediction == 1:  # Neutral
            fill_percent = 75
            bar_color = "#FFA500"  # Orange
        else:  # Positive
            fill_percent = 100
            bar_color = "#4BB543"  # Green

        # -----------------------------
        # Display output
        # -----------------------------
        st.markdown(f"### 🎯 Predicted Sentiment: **{sentiment_text}**")
        st.markdown(f"**Model Confidence:** {confidence:.2f}%")

        # Sentiment-based color bar (no percentage text)
        bar_html = f"""
        <style>
        .bar-container {{
            background-color: #ddd;
            border-radius: 25px;
            width: 100%;
            height: 30px;
            margin-top: 10px;
        }}
        .bar-fill {{
            background-color: {bar_color};
            height: 30px;
            border-radius: 25px;
            transition: width 0.8s ease-in-out;
        }}
        </style>
        <div class='bar-container'>
            <div class='bar-fill'></div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

        # -----------------------------
        # Show class probabilities
        # -----------------------------
        st.markdown("### 🧾 Class Probabilities")
        prob_df = pd.DataFrame({
            "Sentiment": ["Negative 😞", "Neutral 😐", "Positive 😊"],
            "Confidence (%)": probabilities.round(2)
        })
        st.dataframe(prob_df.set_index("Sentiment"))

# -----------------------------
# Sidebar Info
# -----------------------------
st.sidebar.title("ℹ️ App Info")
st.sidebar.write("""
**Sentiment Color Mapping (Fixed):**
- 🔴 Negative → Red (40%)
- 🟠 Neutral → Orange (75%)
- 🟢 Positive → Green (100%)

**Model:** XGBoost Classifier
**Features:** TF-IDF (Title + Body)
""")
