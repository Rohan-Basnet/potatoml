from pathlib import Path

# ======================================================
# Project Directories
# ======================================================

# Root directory (PotatoMLService/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Application directories
MODEL_DIR = BASE_DIR / "model"
UPLOAD_DIR = BASE_DIR / "uploads"

# ======================================================
# Model Files
# ======================================================

MODEL_PATH = MODEL_DIR / "potato_model.pkl"
CLASS_MAPPING_PATH = MODEL_DIR / "class_mapping.json"

# ======================================================
# Image Preprocessing Configuration
# (Must match the training configuration)
# ======================================================

IMAGE_SIZE = (128, 128)

# ======================================================
# HOG Feature Extraction Configuration
# (Must match the training configuration)
# ======================================================

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_VISUALIZE = False