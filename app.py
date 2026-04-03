from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load model (with vectorizer inside)
model = joblib.load("sms_fraud_model_MNB_CV.pkl")

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
        prediction = model.predict([text])[0]

        # Confidence
        probabilities = model.predict_proba([text])[0]
        confidence = max(probabilities)

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