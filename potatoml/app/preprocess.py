import cv2
import numpy as np
from skimage.feature import hog

from app.config import (
    IMAGE_SIZE,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
    HOG_VISUALIZE,
)


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess an image for potato disease prediction.

    Steps:
    1. Read image
    2. Resize to training image size
    3. Convert to grayscale
    4. Extract HOG features
    5. Return feature vector
    """

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Resize image
    image = cv2.resize(image, IMAGE_SIZE)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract HOG features
    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        visualize=HOG_VISUALIZE,
    )

    # Return as NumPy array
    return np.array(features)