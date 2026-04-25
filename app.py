from flask import Flask, request, jsonify, render_template
import requests
import os

app = Flask(__name__)

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

        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "inputs": text,
                "parameters": {
                    "max_length": 130,
                    "min_length": 30,
                    "do_sample": False
                },
                "options": {"wait_for_model": True}
            },
            timeout=60
        )

        print("STATUS:", response.status_code)
        print("RESPONSE:", response.text)

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
    app.run(host="0.0.0.0", port=5000)