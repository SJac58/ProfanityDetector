from flask import Flask, request, jsonify
from flask_cors import CORS
from model.predict import predict_toxicity

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        message = data.get("text", "")
        is_toxic = predict_toxicity(message)
        data={"toxic": is_toxic, "reason": "AI detected profanity/offensive language in this message"}
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
