import os
import cv2
import joblib
import numpy as np

from skimage.feature import hog
from src.disease_info import disease_info

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "potato_classifier.pkl")

model = joblib.load(MODEL_PATH)


def load_image(image_path=None, image_bytes=None):
    if image_bytes is not None:
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    elif image_path is not None:
        image = cv2.imread(image_path)
    else:
        raise ValueError("Either image_path or image_bytes must be provided.")

    if image is None:
        raise ValueError("Unable to read image. Please provide a valid image file.")

    return cv2.resize(image, (128, 128))


def get_feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
    )


def predict_image(image_path=None, image_bytes=None):
    image = load_image(image_path=image_path, image_bytes=image_bytes)
    feature = get_feature(image)
    prediction = model.predict([feature])[0]
    return disease_info[prediction]