"""REST API serving the churn model.

Run locally:
    uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn import __version__
from churn.explain import explain_customer
from churn.predict import load_model, predict

YesNo = Literal["Yes", "No"]
YesNoService = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoService
    OnlineBackup: YesNoService
    DeviceProtection: YesNoService
    TechSupport: YesNoService
    StreamingTV: YesNoService
    StreamingMovies: YesNoService
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
                "Dependents": "No", "tenure": 3, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "Yes",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.5, "TotalCharges": 280.3,
            }
        }
    }


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_level: str
    top_drivers: list[dict] = []


app = FastAPI(
    title="Customer Churn Prediction API",
    version=__version__,
    description="Scores telecom customers by churn risk and explains the prediction.",
)


@app.get("/health")
def health() -> dict:
    try:
        artifact = load_model()
    except FileNotFoundError:
        return {"status": "error", "detail": "model not trained"}
    return {"status": "ok", "model": artifact["model_name"], "version": __version__}


@app.post("/predict", response_model=Prediction)
def predict_one(customer: Customer) -> Prediction:
    try:
        artifact = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    df = pd.DataFrame([customer.model_dump()])
    row = predict(df, artifact).iloc[0]
    return Prediction(
        churn_probability=float(row["churn_probability"]),
        will_churn=bool(row["will_churn"]),
        risk_level=str(row["risk_level"]),
        top_drivers=explain_customer(artifact["pipeline"], df),
    )


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer]) -> list[Prediction]:
    try:
        artifact = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    df = pd.DataFrame([c.model_dump() for c in customers])
    preds = predict(df, artifact)
    return [
        Prediction(
            churn_probability=float(r.churn_probability),
            will_churn=bool(r.will_churn),
            risk_level=str(r.risk_level),
        )
        for r in preds.itertuples()
    ]
