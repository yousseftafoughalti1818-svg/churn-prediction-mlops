import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from churn import config  # noqa: E402
from churn.predict import load_model  # noqa: E402
from churn.train import train  # noqa: E402

CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 280.3,
}


@pytest.fixture
def client(tmp_project):
    load_model.cache_clear()
    train(config.RAW_DATA_PATH)
    from api.main import app

    yield TestClient(app)
    load_model.cache_clear()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict(client):
    body = client.post("/predict", json=CUSTOMER).json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in {"low", "medium", "high"}


def test_invalid_input_is_rejected(client):
    bad = {**CUSTOMER, "Contract": "Lifetime"}
    assert client.post("/predict", json=bad).status_code == 422
