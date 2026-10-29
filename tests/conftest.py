"""Shared fixtures: a small synthetic dataset with the Telco schema (tests only)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn import config  # noqa: E402


def make_fake_telco(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    tenure = rng.integers(0, 72, n)
    monthly = np.round(rng.uniform(20, 110, n), 2)
    yn = lambda: rng.choice(["Yes", "No"], n)  # noqa: E731
    svc = lambda: np.where(internet == "No", "No internet service", yn())  # noqa: E731
    logit = (-1.5 + 1.6 * (contract == "Month-to-month") + 0.8 * (internet == "Fiber optic")
             - 0.03 * tenure)
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    total = np.round(monthly * tenure, 2).astype(str)
    total[tenure == 0] = " "  # same quirk as the real dataset
    return pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": yn(), "Dependents": yn(), "tenure": tenure,
        "PhoneService": yn(), "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": internet, "OnlineSecurity": svc(), "OnlineBackup": svc(),
        "DeviceProtection": svc(), "TechSupport": svc(), "StreamingTV": svc(),
        "StreamingMovies": svc(), "Contract": contract, "PaperlessBilling": yn(),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check",
                                     "Bank transfer (automatic)", "Credit card (automatic)"], n),
        "MonthlyCharges": monthly, "TotalCharges": total,
        "Churn": np.where(churn, "Yes", "No"),
    })


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_fake_telco()


@pytest.fixture
def tmp_project(tmp_path, monkeypatch, raw_df):
    """Redirect every output path of the project to a temporary folder."""
    data_path = tmp_path / "data.csv"
    raw_df.to_csv(data_path, index=False)
    monkeypatch.setattr(config, "RAW_DATA_PATH", data_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "models" / "model.joblib")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "reports" / "metrics.json")
    monkeypatch.setattr(config, "CV_FOLDS", 3)
    return tmp_path
