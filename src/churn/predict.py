"""Load the trained model and score customers."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from churn import config


@lru_cache(maxsize=1)
def load_model(path: Path | str | None = None) -> dict:
    path = Path(path) if path is not None else config.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train the model with `make train`.")
    return joblib.load(path)


def risk_level(proba: float) -> str:
    if proba >= 0.7:
        return "high"
    if proba >= 0.4:
        return "medium"
    return "low"


def predict(customers: pd.DataFrame, artifact: dict | None = None) -> pd.DataFrame:
    """Return churn probability, decision and risk level for each customer."""
    artifact = artifact or load_model()
    proba = artifact["pipeline"].predict_proba(customers)[:, 1]
    out = pd.DataFrame(index=customers.index)
    out["churn_probability"] = proba.round(4)
    out["will_churn"] = proba >= artifact["threshold"]
    out["risk_level"] = [risk_level(p) for p in proba]
    return out
