from src.feature_extraction import extract_features
from src.train import (
    split_data,
    train_logistic,
    train_knn,
    train_svm,
)
from src.evaluate import evaluate

import joblib
import os

# Dataset path
base_path = "/Users/rohanbasnet/Downloads/archive/potato_disease/"

# Verify dataset exists
if not os.path.exists(base_path):
    raise FileNotFoundError(f"Dataset folder not found: {base_path}")

print(f"Dataset Path: {base_path}")

# 1. Extract features
X, y, classes = extract_features(base_path)

print("Features:", X.shape)


# 2. Split dataset
X_train, X_test, y_train, y_test = split_data(X, y)


# 3. Train models

logistic = train_logistic(
    X_train,
    y_train
)

knn = train_knn(
    X_train,
    y_train
)

svm = train_svm(
    X_train,
    y_train
)


# 4. Evaluate

print(
    "Logistic:",
    evaluate(logistic, X_test, y_test)
)

print(
    "KNN:",
    evaluate(knn, X_test, y_test)
)

print(
    "SVM:",
    evaluate(svm, X_test, y_test)
)


# 5. Save SVM model

# 5. Save model

import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    svm,
    os.path.join(MODEL_DIR, "potato_classifier.pkl")
)

joblib.dump(
    classes,
    os.path.join(MODEL_DIR, "classes.pkl")
)

print("✅ Model saved")
print("✅ Classes saved")