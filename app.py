from flask import Flask, request, jsonify, render_template
import requests
import os

app = Flask(__name__)

# Get API key from environment
HF_API_KEY = os.getenv("HF_API_KEY")

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"summary": "Please enter some text."})

        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(API_URL, headers=headers, json={"inputs": text})

        if response.status_code != 200:
            return jsonify({"summary": "API Error: " + response.text})

        result = response.json()

        if isinstance(result, list) and "summary_text" in result[0]:
            summary = result[0]["summary_text"]
        else:
            summary = str(result)

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"summary": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)