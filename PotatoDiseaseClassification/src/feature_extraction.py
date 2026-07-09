import os
import cv2
import numpy as np

from skimage.feature import hog


def extract_features(base_path):

    X = []

    y = []

    classes = sorted(
    folder
    for folder in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, folder))
    )
    
    for label, cls in enumerate(classes):

        folder = os.path.join(base_path, cls)

        for image_name in os.listdir(folder):

            image_path = os.path.join(folder, image_name)

            image = cv2.imread(image_path)

            image = cv2.resize(image, (128,128))

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            feature = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8,8),
                cells_per_block=(2,2),
                visualize=False
            )

            X.append(feature)

            y.append(label)

    return np.array(X), np.array(y), classes