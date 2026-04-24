from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

# Create Flask app instance
app = Flask(__name__)
CORS(app)

# Load API Key
API_KEY = os.getenv("API_KEY", "v3ryscurkeis") 

# Load model
data = joblib.load("sms_spam_model_LR_TFIDF_V2.pkl")
model = data["model"] # Loads Model
threshold = data["threshold"] # Loads Model Threshold

# Display Message for Render Site
@app.route("/")
def home():
    return "Spam Detection API is running"

# API Post Endpoint
@app.route("/predict", methods=["POST"])

def predict():
    try:
        # API Key Authentication
        client_key = request.headers.get("x-api-key") # Reads API Key from Header
        if client_key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401 # Returns 401 if API Key is Invalid

        data = request.get_json()

        # Input Validation
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]

        if not isinstance(text, str):
            return jsonify({"error": "Input must be a string"}), 400

        text = text.strip()

        if len(text) == 0:
            return jsonify({"error": "Empty message"}), 400


        # Model prediction
        spam_prob = model.predict_proba([text])[0][1] # Calcs Prob of Spam
        ham_prob = 1 - spam_prob # Calcs Prob of Real Message

        prediction = "Spam" if spam_prob > threshold else "Not Spam" # If prediction above threshold, then Spam, else Not Spam

        # Returns JSON response with prediction and probabilities
        return jsonify({
            "prediction": prediction,
            "spam_probability": float(spam_prob),
            "ham_probability": float(ham_prob)
        })

    # Crash handling
    except Exception:
        return jsonify({"error": "Internal server error"}), 500

# Runs Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)