"""Feature engineering and preprocessing pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn import config

TENURE_BINS = [-1, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-6m", "6-12m", "1-2y", "2-4y", "4y+"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create business-driven features from the raw columns.

    The function is pure (no fitted state), so it can be used inside the
    sklearn pipeline and applied identically at training and inference time.
    """
    df = df.copy()
    tenure = df["tenure"].astype(float)
    monthly = df["MonthlyCharges"].astype(float)
    total = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Average spend over the customer's lifetime (monthly charges for new customers)
    df["avg_monthly_spend"] = np.where(tenure > 0, total / tenure.replace(0, np.nan), monthly)
    df["avg_monthly_spend"] = df["avg_monthly_spend"].fillna(monthly)

    # Number of subscribed services
    services = df[config.SERVICE_COLS].astype(str).eq("Yes")
    internet = df["InternetService"].astype(str).ne("No")
    df["num_services"] = services.sum(axis=1) + internet.astype(int)
    df["charges_per_service"] = monthly / df["num_services"].clip(lower=1)

    # Contract, technology and payment signals
    df["is_month_to_month"] = df["Contract"].eq("Month-to-month").astype(int)
    df["is_fiber"] = df["InternetService"].eq("Fiber optic").astype(int)
    df["auto_payment"] = df["PaymentMethod"].astype(str).str.contains("automatic").astype(int)
    df["is_new_customer"] = (tenure <= 6).astype(int)

    df["tenure_group"] = pd.cut(tenure, bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    return df


def build_preprocessor() -> Pipeline:
    """Feature engineering + scaling of numeric columns + one-hot encoding."""
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(add_features, validate=False)),
            ("columns", column_transformer),
        ]
    )
