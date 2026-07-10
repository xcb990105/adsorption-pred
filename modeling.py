from __future__ import annotations

import json
from pathlib import Path

import joblib


MODEL_FILES = {
    "CatBoost": "CatBoost.pkl",
    "XGBoost": "XGBoost.pkl",
    "Random Forest": "RandomForest.pkl",
    "GPR": "GPR.pkl",
}


class ModelArtifactError(RuntimeError):
    pass


def model_names() -> list[str]:
    return list(MODEL_FILES.keys())


def load_model(model_name: str, models_dir: Path | str = "models"):
    if model_name not in MODEL_FILES:
        raise ModelArtifactError(f"Unknown model: {model_name}")

    path = Path(models_dir) / MODEL_FILES[model_name]
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")

    return joblib.load(path)


def load_metrics(models_dir: Path | str = "models") -> dict:
    path = Path(models_dir) / "model_metrics.json"
    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


def predict_capacity(model, feature_frame) -> float:
    prediction = model.predict(feature_frame)
    return float(prediction[0])
