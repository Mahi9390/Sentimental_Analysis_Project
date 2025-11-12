from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# -----------------------------
# Load the trained model
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'sentiment_model.joblib')
model = joblib.load(MODEL_PATH)

# Sentiment mapping (same as your Streamlit app)
sentiment_labels = {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}

# -----------------------------
# Home page
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -----------------------------
# Predict endpoint
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        title_input = request.form.get('title')
        body_input = request.form.get('body')

        if not title_input or not body_input:
            return jsonify({'error': 'Please provide both title and body.'}), 400

        # Prepare input as DataFrame (like in your Streamlit app)
        input_df = pd.DataFrame({'title': [title_input], 'body': [body_input]})

        # Make predictions
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0] * 100

        # Convert NumPy to Python types
        prediction = int(prediction)
        probabilities = [float(p) for p in probabilities]

        sentiment_text = sentiment_labels[prediction]
        confidence = probabilities[prediction]

        # Color bar logic
        if prediction == 0:  # Negative
            fill_percent = 40
            bar_color = "#FF4B4B"  # Red
        elif prediction == 1:  # Neutral
            fill_percent = 75
            bar_color = "#FFA500"  # Orange
        else:  # Positive
            fill_percent = 100
            bar_color = "#4BB543"  # Green

        return jsonify({
            'sentiment': sentiment_text,
            'confidence': round(confidence, 2),
            'probabilities': probabilities,
            'bar_color': bar_color,
            'fill_percent': fill_percent
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    from waitress import serve
    print("✅ Server running at http://127.0.0.1:5000")
    serve(app, host='0.0.0.0', port=5000)
