from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

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
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]

        # Get probability of SPAM (class = 1)
        spam_prob = model.predict_proba([text])[0][1]
        ham_prob = 1 - spam_prob

        prediction = "Spam" if spam_prob > threshold else "Not Spam"

        return jsonify({
            "prediction": prediction,
            "spam_probability": float(spam_prob),
            "ham_probability": float(ham_prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)