"""Data loading and cleaning for the IBM Telco Customer Churn dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from churn import config


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Read the raw CSV file."""
    path = Path(path) if path is not None else config.RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types and missing values.

    - TotalCharges is stored as text and contains blanks for new customers
      (tenure = 0): they are converted to numbers and filled with 0.
    - The target is mapped to 0/1.
    - Duplicated customers are removed.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip().replace("", np.nan),
        errors="coerce",
    )
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if config.TARGET in df.columns and not pd.api.types.is_numeric_dtype(df[config.TARGET]):
        df[config.TARGET] = df[config.TARGET].str.strip().map({"Yes": 1, "No": 0})

    if config.ID_COL in df.columns:
        df = df.drop_duplicates(subset=config.ID_COL)
    return df.reset_index(drop=True)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target (the customer id is dropped)."""
    X = df.drop(columns=[c for c in (config.TARGET, config.ID_COL) if c in df.columns])
    y = df[config.TARGET].astype(int)
    return X, y
