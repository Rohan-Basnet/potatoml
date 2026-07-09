# Potato Disease Classification

A Python web application that classifies potato leaf images into one of three categories:
- Early Blight
- Late Blight
- Healthy

The application exposes a browser interface and a JSON API for image uploads.

## Deploying to Render

1. Connect your repository to Render and choose `Web Service`.
2. Set the service root to `PotatoDiseaseClassification`.
3. Set the environment to `Python`.
4. Use the following commands:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. If you use the repository `render.yaml`, Render will automatically detect these settings.

## Running locally

From the `PotatoDiseaseClassification` directory:

```bash
python -m pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000`.

## API

- `POST /api/predict`: upload a file as `image` field and receive a JSON prediction.
- `GET /health`: service health check.

## Files

- `app.py`: Flask web application.
- `src/predict.py`: image feature extraction and model inference.
- `models/potato_classifier.pkl`: pre-trained SVM classifier.

## Notes

- Use `opencv-python-headless` in production to avoid GUI dependencies.
- The service loads the pretrained model from `models/potato_classifier.pkl`.
