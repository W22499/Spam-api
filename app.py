from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load API Key
API_KEY = os.getenv("API_KEY", "v3ryscurkeis") 

# Load model
data = joblib.load("sms_spam_model_LR_TFIDF_V2.pkl")
model = data["model"]
threshold = data["threshold"]

@app.route("/")
def home():
    return "Spam Detection API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # API Key Authentication
        client_key = request.headers.get("x-api-key")
        if client_key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

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
        spam_prob = model.predict_proba([text])[0][1]
        ham_prob = 1 - spam_prob

        prediction = "Spam" if spam_prob > threshold else "Not Spam"

        return jsonify({
            "prediction": prediction,
            "spam_probability": float(spam_prob),
            "ham_probability": float(ham_prob)
        })

    except Exception:
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)