from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import numpy as np

# --- 1. Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# --- 2. Global Variables: Load Models ---
# Initialize variables to hold the loaded model and vectorizer
model = None
vectorizer = None

try:
    # Load the trained model and vectorizer files (These MUST be in the same directory)
    model = joblib.load('text_threat_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("AI Text Model assets loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR loading model assets: {e}. Ensure both .pkl files are in the root directory.")
    # Exit or disable the prediction route if models aren't found
    # For deployment, this is CRITICAL.

# --- 3. API Endpoint for Prediction ---
@app.route('/predict_threat', methods=['POST'])
def predict_threat():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not available. Server startup failed.'}), 503

    # Get JSON data from the request (sent by the frontend)
    data = request.get_json(force=True)
    report_text = data.get('text', '')

    if not report_text:
        return jsonify({'error': 'No text provided for analysis.'}), 400

    try:
        # Preprocess the incoming text using the saved vectorizer
        # The vectorizer expects an array of strings
        text_vector = vectorizer.transform([report_text])

        # Make the prediction
        prediction_label = model.predict(text_vector)[0]
        
        # Get the probability score
        probabilities = model.predict_proba(text_vector)[0]
        label_index = list(model.classes_).index(prediction_label)
        confidence_score = probabilities[label_index] * 100 # Convert to percentage

        # --- Format the API Response ---
        response = {
            'status': 'success',
            'report_text': report_text,
            'prediction': prediction_label,
            'confidence': f"{confidence_score:.2f}%"
        }

        # Set severity based on prediction
        if prediction_label == 'Threat':
            response['severity'] = 'CRITICAL'
        else:
            response['severity'] = 'LOW'

        return jsonify(response)
        
    except Exception as e:
        # Catch any errors during prediction (e.g., unexpected data type)
        return jsonify({'error': f'Prediction failed due to server error: {e}'}), 500

# --- 4. Run the API (For local testing) ---
if __name__ == '__main__':
    # When deployed, the production server (Gunicorn) will handle running the app
    # This block is mainly for running locally in Codespaces: python api_server.py
    app.run(debug=True, port=5000)