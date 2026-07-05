"""
RadioScreen Backend — Educational Screening API

Author: Fateh

DISCLAIMER: Research/educational prototype only. NOT a certified medical
device. NOT for real diagnostic use. Always accompanied by a mandatory
disclaimer in every API response.

Run locally with:
    pip install fastapi uvicorn scikit-learn numpy pillow python-multipart
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DISCLAIMER = (
    "Educational research prototype only. NOT a certified medical device. "
    "NOT a diagnosis. Always consult a licensed physician or radiologist."
)

IMG_SIZE = 64

app = FastAPI(
    title="RadioScreen API (Educational Prototype)",
    description="Chest X-ray screening demo — NOT for clinical use. " + DISCLAIMER,
    version="0.1.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------
# Train the same lightweight demo model used in poc/radioscreen_poc.py,
# on synthetic data, once at startup.
# --------------------------------------------------------------------------
def _synthetic_xray(rng: np.random.Generator, abnormal: bool) -> np.ndarray:
    base = rng.normal(loc=120, scale=15, size=(IMG_SIZE, IMG_SIZE))
    if abnormal:
        cx, cy = rng.integers(15, IMG_SIZE - 15, size=2)
        radius = rng.integers(6, 14)
        yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        base[mask] += rng.normal(loc=55, scale=10, size=mask.sum())
    return np.clip(base, 0, 255)


def extract_features(image: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(image)
    edge_density = np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2))
    return np.array([image.mean(), image.std(), np.percentile(image, 90), edge_density])


def train_demo_model():
    rng = np.random.default_rng(42)
    samples, labels = [], []
    for _ in range(300):
        samples.append(extract_features(_synthetic_xray(rng, abnormal=False)))
        labels.append(0)
        samples.append(extract_features(_synthetic_xray(rng, abnormal=True)))
        labels.append(1)
    X, y = np.array(samples), np.array(labels)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=6)
    clf.fit(X_train, y_train)
    return clf


MODEL = train_demo_model()


def preprocess_upload(raw_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(raw_bytes)).convert("L")  # grayscale
    image = image.resize((IMG_SIZE, IMG_SIZE))
    return np.array(image, dtype=float)


@app.get("/health")
def health():
    return {"status": "ok", "disclaimer": DISCLAIMER}


@app.post("/screen")
async def screen_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image and returns a demo screening score.
    This endpoint NEVER returns a diagnosis — only an illustrative
    'abnormality likelihood' score from a model trained on synthetic data.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    raw = await file.read()
    try:
        image = preprocess_upload(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    features = extract_features(image).reshape(1, -1)
    proba = float(MODEL.predict_proba(features)[0, 1])

    return {
        "filename": file.filename,
        "abnormality_likelihood": round(proba, 4),
        "label": "possible_opacity_pattern" if proba > 0.5 else "no_notable_pattern_detected",
        "model": "RandomForest (demo, trained on synthetic data — NOT a real diagnostic model)",
        "disclaimer": DISCLAIMER,
    }
