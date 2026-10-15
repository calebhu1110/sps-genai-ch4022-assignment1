"""CNN image-classification endpoint (Assignment 2)."""
import io
from pathlib import Path

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from .model import SimpleCNN
from .preprocess import CLASSES, transform

MODEL_PATH = Path(__file__).parent / "model.pth"

# Load the trained weights once, when the app starts.
_model = SimpleCNN()
_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
_model.eval()

router = APIRouter(tags=["cnn"])


@router.post("/predict_cnn")
async def predict_cnn(file: UploadFile = File(...)):
    """Upload an image; returns the predicted CIFAR10 class."""
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    x = transform(image).unsqueeze(0)  # (3, 64, 64) -> (1, 3, 64, 64)
    with torch.no_grad():
        probs = torch.softmax(_model(x), dim=1)[0]
    idx = int(probs.argmax())
    return {
        "predicted_class": CLASSES[idx],
        "class_index": idx,
        "confidence": round(float(probs[idx]), 4),
    }
