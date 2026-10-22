from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.pt"

image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

app = FastAPI(
    title="Kidney CT Classifier API",
    description="Educational image-classification API. Not for clinical diagnosis.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def load_model():
    """Load the trained checkpoint once, when the first prediction is requested."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}. "
            "Run the training notebook or add the trained model/model.pt file."
        )

    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        classes = checkpoint["classes"]
        state_dict = checkpoint["state_dict"]
    except (OSError, KeyError, RuntimeError, TypeError) as exc:
        raise RuntimeError(
            f"Could not load the model checkpoint at {MODEL_PATH}: {exc}"
        ) from exc

    if not isinstance(classes, list) or not classes:
        raise RuntimeError("The model checkpoint does not contain a valid class list.")

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(state_dict)
    model.eval()
    return model, classes


@app.get("/")
def home():
    return {
        "message": "Kidney CT Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health")
def health():
    try:
        load_model()
    except (FileNotFoundError, RuntimeError) as exc:
        return {
            "status": "degraded",
            "model_available": False,
            "detail": str(exc),
        }

    return {"status": "ok", "model_available": True}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid JPG or PNG image.",
        ) from None
    finally:
        file.file.close()

    try:
        model, classes = load_model()
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    with torch.inference_mode():
        probabilities = model(image_transform(image).unsqueeze(0)).softmax(dim=1)[0]

    scores = probabilities.tolist()
    top_index = max(range(len(classes)), key=scores.__getitem__)

    return {
        "prediction": classes[top_index],
        "confidence": scores[top_index],
        "probabilities": dict(zip(classes, scores)),
    }
