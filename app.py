from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import math

app = Flask(__name__)
CORS(app)

# Load model (with vectorizer inside)
data = joblib.load("sms_fraud_model_LSV_CV.pkl")

model = data["model"]
threshold = data["threshold"]

@app.route("/")
def home():
    return "Spam Detection API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate input
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]

        # Prediction
        score = model.decision_function([text])[0]
        prediction = 1 if score > threshold else 0

        # Confidence
        decision_score = model.decision_function([text])[0]

        # Convert to pseudo-confidence (0–1)
        confidence = 1 / (1 + math.exp(-score))

        # Convert to readable label
        result = "spam" if prediction == 1 else "Not Spam"

        return jsonify({
            "prediction": result,
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)