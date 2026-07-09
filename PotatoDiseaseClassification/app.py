import os

from flask import Flask, jsonify, render_template_string, request

from src.predict import predict_image

app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Potato Disease Classifier</title>
    <style>
      body { font-family: Arial, sans-serif; padding: 20px; max-width: 700px; margin: auto; }
      h1 { margin-bottom: 0.2em; }
      .card { background: #f9f9f9; padding: 20px; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.1); }
      input[type=file] { margin-bottom: 1em; }
      .result { margin-top: 1em; padding: 15px; background: #fff; border: 1px solid #ddd; border-radius: 8px; }
      .error { color: red; }
      pre { white-space: pre-wrap; word-wrap: break-word; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Potato Disease Classifier</h1>
      <p>Upload a potato leaf image to identify whether it is Healthy, Early Blight, or Late Blight.</p>
      <form method="post" action="/predict" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required />
        <button type="submit">Predict</button>
      </form>
      {% if result %}
      <div class="result">
        <h2>Prediction</h2>
        <p><strong>Disease:</strong> {{ result.disease }}</p>
        <p><strong>Scientific name:</strong> {{ result.scientific_name }}</p>
        <p><strong>Description:</strong> {{ result.description }}</p>
        <p><strong>Treatment:</strong> {{ result.treatment }}</p>
      </div>
      {% endif %}
      {% if error %}
      <div class="error">{{ error }}</div>
      {% endif %}
    </div>
  </body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE, result=None, error=None)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template_string(HTML_TEMPLATE, result=None, error="No image file was provided."), 400

    uploaded_file = request.files["image"]
    image_data = uploaded_file.read()

    if not image_data:
        return render_template_string(HTML_TEMPLATE, result=None, error="Uploaded image is empty."), 400

    try:
        result = predict_image(image_bytes=image_data)
    except Exception as exc:
        return render_template_string(HTML_TEMPLATE, result=None, error=str(exc)), 400

    return render_template_string(HTML_TEMPLATE, result=result, error=None)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file was provided."}), 400

    image_data = request.files["image"].read()
    if not image_data:
        return jsonify({"error": "Uploaded image is empty."}), 400

    try:
        result = predict_image(image_bytes=image_data)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
