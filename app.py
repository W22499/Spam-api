from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load model (with vectorizer inside)
model = joblib.load("sms_fraud_model_MNB_CV.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data["text"]

        # Direct prediction (no manual vectorizing)
        prediction = model.predict([text])[0]

        result = "Spam" if prediction == 1 else "Not Spam"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)