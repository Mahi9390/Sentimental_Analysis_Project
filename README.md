Multilingual Sentiment Analysis and Rating Classification System


📌 Project Overview

This project implements an end-to-end multilingual sentiment analysis system that classifies customer reviews into Negative, Neutral, and Positive categories. The solution handles multilingual input through automated translation, applies advanced NLP preprocessing, and uses a machine learning pipeline with XGBoost for accurate sentiment prediction. The trained model is deployed using both Streamlit and Flask, enabling interactive UI-based and API-based access.

🗂️ Dataset

Source: Customer review dataset (Excel format)

Features:

title: Review title text

body: Review description text

rating: Numerical rating used to derive sentiment labels

Target:

sentiment (Negative, Neutral, Positive)

🔎 Exploratory Data Analysis (EDA)

Analyzed rating and sentiment distributions

Visualized patterns using histograms and bar plots

Generated word clouds to understand dominant terms in reviews

Identified class imbalance across sentiment categories

🧹 Text Preprocessing & Feature Engineering

Automated multilingual translation to English

Text cleaning: lowercasing, URL and noise removal, emoji handling

Tokenization, stopword removal, and lemmatization

Feature extraction using TF-IDF (unigrams & bigrams) for both title and body

Parallel text processing using ColumnTransformer

🤖 Model Development

Model: XGBoost multi-class classifier

Handled class imbalance using custom class weights

Integrated preprocessing and modeling using a scikit-learn Pipeline

Model evaluation:

Accuracy score

Classification report

Confusion matrix

🚀 Model Deployment
Streamlit Web Application

Interactive UI for real-time sentiment prediction

Displays predicted sentiment with confidence scores

Suitable for demonstrations and end users

Flask REST API

Exposes the trained model as an API endpoint

Accepts JSON input for title and body text

Returns sentiment predictions for integration with other applications

🛠️ Technologies Used

Programming: Python

NLP: NLTK, TF-IDF, WordCloud

ML Model: XGBoost

Data Analysis: pandas, NumPy, Matplotlib, Seaborn

Pipelines: scikit-learn

Deployment: Streamlit, Flask

Model Storage: joblib

▶️ How to Run the Project
1. Clone the Repository
git clone https://github.com/your-username/sentiment-analysis-project.git
cd sentiment-analysis-project

2. Install Dependencies
pip install -r requirements.txt

3. Run Streamlit App
streamlit run app.py

4. Run Flask API
python app.py
