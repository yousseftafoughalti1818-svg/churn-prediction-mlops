# 📉 Customer Churn Prediction - End-to-end ML project

![CI](https://github.com/yousseftafoughalti1818-svg/churn-prediction-mlops/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-boosting-green)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)

Predict which telecom customers are about to cancel their subscription, **explain why**, and
serve the model through an **API** and a **dashboard** for the marketing team.

The project covers the full data science life cycle: data cleaning → exploratory analysis →
feature engineering → model comparison → threshold tuning → explainability (SHAP) →
REST API → Docker → automated tests and CI.

---

## 🎯 Business problem

Acquiring a new customer costs much more than keeping an existing one. The goal is to give the
retention team a **ranked list of at-risk customers** so they can target loyalty offers where
they matter most, and to understand the **main drivers of churn**.

- **Data**: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) - 7,043 customers, 21 columns
  (contract, services, billing, tenure…), ~26% churners.
- **Target**: `Churn` (Yes/No).
- **Baseline ("existing approach")**: a simple business rule that flags every
  *month-to-month* customer as at risk. The model must beat it.

## 🧱 Architecture

```
churn-prediction-mlops/
├── src/churn/
│   ├── config.py        # paths, columns, parameters
│   ├── data.py          # loading and cleaning (TotalCharges fix, target encoding)
│   ├── features.py      # feature engineering + sklearn preprocessing pipeline
│   ├── train.py         # CV model comparison, threshold tuning, reports
│   ├── explain.py       # SHAP global and per-customer explanations
│   └── predict.py       # inference helpers
├── api/main.py          # FastAPI service (/health, /predict, /predict/batch)
├── app/streamlit_app.py # dashboard for the marketing team
├── notebooks/01_eda.ipynb
├── tests/               # pytest: data, features, training, API
├── scripts/download_data.py
├── Dockerfile
├── Makefile
└── .github/workflows/ci.yml   # lint + tests + training + Docker build
```

## 🔬 Methodology

1. **Cleaning**: `TotalCharges` is stored as text with blanks for new customers → converted
   to numeric and set to 0 when `tenure = 0`; duplicates removed.
2. **Feature engineering** (inside the sklearn pipeline, so training and inference use the exact
   same code):
   `avg_monthly_spend`, `num_services`, `charges_per_service`, `is_month_to_month`, `is_fiber`,
   `auto_payment`, `is_new_customer`, `tenure_group`.
3. **Models compared** with 5-fold stratified cross-validation (ROC-AUC):
   Logistic Regression, Random Forest, XGBoost. Class imbalance is handled with class weights
   (`class_weight="balanced"`, `scale_pos_weight`).
4. **Model selection on cross-validation only** (no test-set leakage), then the **decision
   threshold** is tuned on out-of-fold predictions to maximise F1.
5. **Evaluation** on a held-out test set (20%): ROC-AUC, PR-AUC, precision, recall, F1,
   lift in the top 10% riskiest customers, and comparison with the business-rule baseline.
6. **Explainability** with SHAP: global drivers of churn and the top reasons for each customer.

## 📊 Results

Trained on the full dataset (7,043 customers, 26.5% churners), evaluated on a held-out 20% test
set (1,409 customers). Full report: [`reports/metrics.md`](reports/metrics.md).

| Model | CV ROC-AUC (5 folds) | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.847 ± 0.011 | **0.845** | 0.651 |
| **Random Forest** (selected) | **0.848 ± 0.010** | 0.844 | **0.657** |
| XGBoost | 0.841 ± 0.011 | 0.839 | 0.646 |

Random Forest is selected on cross-validation; decision threshold tuned to **0.58** (max F1 on
out-of-fold predictions).

| Test set | Precision | Recall | F1 |
|---|---|---|---|
| **Random Forest** | **0.552** | 0.727 | **0.627** |
| Business rule (flag every month-to-month customer) | 0.426 | 0.880 | 0.574 |

- **+9.4% F1** compared with the business rule, with **+30% precision** (0.55 vs 0.43): far fewer
  retention offers sent to customers who would have stayed.
- **Lift ×2.9 in the top 10%** riskiest customers: contacting the 10% highest-scored customers
  reaches almost 3 times more churners than a random selection.
- The model detects **73% of churners** (272 of 374 in the test set).

<p align="center">
  <img src="reports/figures/roc_curves.png" width="45%">
  <img src="reports/figures/confusion_matrix.png" width="40%">
</p>

### What drives churn? (SHAP)

<p align="center">
  <img src="reports/figures/shap_summary.png" width="60%">
</p>

1. **Month-to-month contract** is by far the strongest driver; two-year contracts strongly reduce risk.
2. **Short tenure**: customers in their first 6 months are much more likely to leave.
3. **No online security / no tech support** increase churn risk.
4. **Fiber optic** internet and **electronic check** payment are associated with higher churn.

**Business recommendations**: offer new month-to-month customers an incentive to switch to a 1- or
2-year contract, bundle online security and tech support with fiber subscriptions, and promote
automatic payment methods.

**Limits and next steps**: `is_month_to_month` and `Contract_Month-to-month` carry the same
information (redundant feature to remove); the three models perform almost identically, which
suggests the signal is mostly linear; next steps are probability calibration, cost-based threshold
selection and monitoring of data drift.

## 🚀 Quick start

```bash
git clone https://github.com/yousseftafoughalti1818-svg/churn-prediction-mlops.git
cd churn-prediction-mlops
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install        # or: pip install -r requirements-dev.txt && pip install -e .
make train          # downloads the data, trains, writes reports/ and models/
make test           # runs the test suite
```

Without `make` (e.g. on Windows):

```bash
pip install -r requirements-dev.txt && pip install -e .
python scripts/download_data.py
python -m churn.train
pytest -q
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
```

### API

```bash
make api            # http://127.0.0.1:8000/docs
```

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
  "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
  "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 280.3}'
```

Example response (values depend on your trained model):

```json
{"churn_probability": 0.81, "will_churn": true, "risk_level": "high",
 "top_drivers": [{"feature": "is_month_to_month", "impact": 0.42}, ...]}
```

### Dashboard

```bash
make app            # Streamlit: portfolio view, single-customer scoring, model performance
```

### Docker

```bash
make docker-build
make docker-run     # API on http://localhost:8000
```

## 🧪 Quality

- **Tests** (`pytest`): data cleaning, feature engineering, end-to-end training on a synthetic
  dataset with the same schema, and API contract (valid and invalid inputs).
- **CI** (GitHub Actions): lint with `ruff`, tests, training on the real dataset (metrics
  uploaded as an artifact), Docker build and container health check.

## 🛠️ Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · Matplotlib · Seaborn ·
FastAPI · Pydantic · Streamlit · Docker · pytest · GitHub Actions

## 👤 Author

**Youssef Tafoughalti** - M2 Complex Systems Engineering (EILCO)
[LinkedIn](https://www.linkedin.com/in/youssef-tafoughalti-b1012a196/) ·
[GitHub](https://github.com/yousseftafoughalti1818-svg)
