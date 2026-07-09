import os
import shutil
import uuid
import requests

from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class ImageRequest(BaseModel):
    image_url: str


@app.post("/predict")
async def predict(request: ImageRequest):
    """
    Predict potato leaf disease from a Cloudinary image URL.
    """

    filename = f"{uuid.uuid4()}.jpg"
    image_path = UPLOAD_DIR / filename

    try:
        # Download image
        response = requests.get(request.image_url, timeout=15)


        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail="Unable to download image from the provided URL."
            )

        content_type = response.headers.get("Content-Type", "")

        if not content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="The provided URL is not an image."
            )

        # Save image
        with open(image_path, "wb") as f:
            f.write(response.content)

        # Predict
        result = predictor.predict(str(image_path))

        return result

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if image_path.exists():
            os.remove(image_path)