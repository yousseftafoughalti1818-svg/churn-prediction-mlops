"""Central configuration: paths, column names and training parameters."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Telco-Customer-Churn.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

ID_COL = "customerID"
TARGET = "Churn"

NUMERIC_RAW = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_RAW = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

# Features created by churn.features.add_features
ENGINEERED_NUMERIC = [
    "avg_monthly_spend", "num_services", "charges_per_service",
    "is_month_to_month", "is_fiber", "auto_payment", "is_new_customer",
]
ENGINEERED_CATEGORICAL = ["tenure_group"]

NUMERIC_FEATURES = NUMERIC_RAW + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = CATEGORICAL_RAW + ENGINEERED_CATEGORICAL

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
