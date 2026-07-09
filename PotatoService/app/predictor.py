import json
import joblib
import numpy as np

from app.config import MODEL_PATH, CLASS_MAPPING_PATH
from app.preprocess import preprocess_image


class PotatoDiseasePredictor:
    """
    Loads the trained model and predicts potato leaf diseases.
    """

    def __init__(self):
        # Load trained model
        self.model = joblib.load(MODEL_PATH)

        # Load class mapping
        with open(CLASS_MAPPING_PATH, "r") as file:
            self.class_mapping = json.load(file)

    def predict(self, image_path: str) -> dict:
        """
        Predict disease from an image.

        Returns:
            {
                "prediction": "...",
                "confidence": ...
            }
        """

        # Extract HOG features
        features = preprocess_image(image_path)

        # Reshape for sklearn
        features = features.reshape(1, -1)

        # Predict class index
        prediction_index = self.model.predict(features)[0]

        # Convert NumPy integer to normal Python int
        prediction_index = int(prediction_index)

        # Get disease name
        prediction = self.class_mapping[str(prediction_index)]

        # Check if model supports probabilities
        confidence = None

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(np.max(probabilities))

        return {
            "prediction": prediction,
            "confidence": confidence
        }


# Create a single predictor instance
predictor = PotatoDiseasePredictor()