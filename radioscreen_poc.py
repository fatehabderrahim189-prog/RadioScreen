"""
RadioScreen — Educational Chest X-ray Screening Assistant
Proof-of-Concept: Feature-Based Classification Pipeline

Author: Fateh

IMPORTANT / DISCLAIMER
----------------------
This is an EDUCATIONAL AND RESEARCH prototype only.
It is NOT a certified medical device and MUST NOT be used for real
clinical diagnosis. It does not replace a licensed radiologist or
physician. All images in this proof of concept are SYNTHETICALLY
GENERATED to demonstrate the pipeline end-to-end, since real patient
data (e.g., NIH ChestX-ray14, Kaggle RSNA Pneumonia) requires a signed
data-use agreement and cannot be bundled or downloaded automatically
in this environment.

For a real deployment, "reference/train_cnn_reference.py" shows how
this pipeline would be re-implemented with a proper deep convolutional
network (transfer learning on DenseNet-121) trained on a licensed
public dataset.
"""

import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


IMG_SIZE = 64  # small synthetic "images" for a fast, dependency-light demo


def _synthetic_xray(rng: np.random.Generator, abnormal: bool) -> np.ndarray:
    """
    Generates a synthetic grayscale image that loosely mimics coarse
    structural differences between a 'clear' and an 'opacity-present'
    chest X-ray region, purely for pipeline demonstration purposes.
    This is NOT derived from or intended to resemble real patient data.
    """
    base = rng.normal(loc=120, scale=15, size=(IMG_SIZE, IMG_SIZE))
    if abnormal:
        # Inject a localized brighter/denser "opacity-like" patch
        cx, cy = rng.integers(15, IMG_SIZE - 15, size=2)
        radius = rng.integers(6, 14)
        yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        base[mask] += rng.normal(loc=55, scale=10, size=mask.sum())
    return np.clip(base, 0, 255)


def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Simple, interpretable hand-crafted features standing in for what a
    CNN would learn automatically: intensity statistics + a crude
    'texture density' measure via the image gradient.
    """
    grad_y, grad_x = np.gradient(image)
    edge_density = np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2))
    return np.array([
        image.mean(),
        image.std(),
        np.percentile(image, 90),
        edge_density,
    ])


@dataclass
class Sample:
    image: np.ndarray
    label: int  # 0 = normal, 1 = abnormal (opacity present)


def build_dataset(n_per_class: int = 300, seed: int = 42) -> list[Sample]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_per_class):
        samples.append(Sample(_synthetic_xray(rng, abnormal=False), 0))
        samples.append(Sample(_synthetic_xray(rng, abnormal=True), 1))
    rng.shuffle(samples)
    return samples


def run_demo():
    print("=" * 72)
    print("RadioScreen PoC — Educational Chest X-ray Screening Pipeline")
    print("DISCLAIMER: synthetic data, NOT a diagnostic tool.")
    print("=" * 72)

    samples = build_dataset()
    X = np.array([extract_features(s.image) for s in samples])
    y = np.array([s.label for s in samples])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=6)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print(f"\nTest set size: {len(y_test)}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")

    print("\nFeature importances (mean intensity, std, p90, edge density):")
    for name, imp in zip(
        ["mean_intensity", "std_intensity", "p90_intensity", "edge_density"],
        clf.feature_importances_,
    ):
        print(f"  {name}: {imp:.3f}")

    print("\nReminder: this pipeline uses SYNTHETIC data and hand-crafted "
          "features for demonstration only. See reference/train_cnn_reference.py "
          "for the production-oriented deep-learning design.")


if __name__ == "__main__":
    run_demo()
