"""Streamlit dashboard for the marketing team.

Run:
    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churn import config  # noqa: E402
from churn.data import clean, load_raw  # noqa: E402
from churn.explain import explain_customer  # noqa: E402
from churn.predict import load_model, predict  # noqa: E402

st.set_page_config(page_title="Churn Radar", page_icon="📉", layout="wide")
st.title("📉 Churn Radar - customer retention dashboard")

try:
    artifact = load_model()
except FileNotFoundError:
    st.error("Model not found. Run `make train` first.")
    st.stop()

tab_portfolio, tab_customer, tab_model = st.tabs(
    ["Customer portfolio", "Score a customer", "Model performance"]
)

# ---------------------------------------------------------------- portfolio
with tab_portfolio:
    uploaded = st.file_uploader("Upload a customer CSV (same columns as the Telco dataset)", type="csv")
    if uploaded is not None:
        df = clean(pd.read_csv(uploaded))
    elif config.RAW_DATA_PATH.exists():
        df = clean(load_raw())
        st.caption("Showing the Telco dataset. Upload your own file to score new customers.")
    else:
        st.info("Upload a CSV file to start.")
        st.stop()

    features = df.drop(columns=[c for c in (config.TARGET, config.ID_COL) if c in df.columns])
    scored = pd.concat([df, predict(features, artifact)], axis=1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(scored):,}")
    c2.metric("Flagged as likely to churn", f"{scored['will_churn'].mean():.1%}")
    c3.metric("Monthly revenue at risk",
              f"{scored.loc[scored['will_churn'], 'MonthlyCharges'].sum():,.0f} $")

    st.subheader("Churn risk by contract type")
    st.bar_chart(scored.groupby("Contract")["churn_probability"].mean())

    st.subheader("Top 50 customers to contact first")
    cols = [c for c in (config.ID_COL, "tenure", "Contract", "MonthlyCharges",
                        "churn_probability", "risk_level") if c in scored.columns]
    st.dataframe(scored.sort_values("churn_probability", ascending=False)[cols].head(50),
                 use_container_width=True)
    st.download_button("Download scored customers (CSV)",
                       scored.to_csv(index=False).encode("utf-8"), "scored_customers.csv")

# ---------------------------------------------------------------- one customer
with tab_customer:
    with st.form("customer"):
        a, b, c = st.columns(3)
        customer = {
            "gender": a.selectbox("Gender", ["Female", "Male"]),
            "SeniorCitizen": int(a.checkbox("Senior citizen")),
            "Partner": a.selectbox("Partner", ["Yes", "No"]),
            "Dependents": a.selectbox("Dependents", ["No", "Yes"]),
            "tenure": a.slider("Tenure (months)", 0, 72, 6),
            "PhoneService": b.selectbox("Phone service", ["Yes", "No"]),
            "MultipleLines": b.selectbox("Multiple lines", ["No", "Yes", "No phone service"]),
            "InternetService": b.selectbox("Internet", ["Fiber optic", "DSL", "No"]),
            "OnlineSecurity": b.selectbox("Online security", ["No", "Yes", "No internet service"]),
            "OnlineBackup": b.selectbox("Online backup", ["No", "Yes", "No internet service"]),
            "DeviceProtection": b.selectbox("Device protection", ["No", "Yes", "No internet service"]),
            "TechSupport": c.selectbox("Tech support", ["No", "Yes", "No internet service"]),
            "StreamingTV": c.selectbox("Streaming TV", ["No", "Yes", "No internet service"]),
            "StreamingMovies": c.selectbox("Streaming movies", ["No", "Yes", "No internet service"]),
            "Contract": c.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
            "PaperlessBilling": c.selectbox("Paperless billing", ["Yes", "No"]),
            "PaymentMethod": c.selectbox("Payment method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"]),
            "MonthlyCharges": c.number_input("Monthly charges ($)", 0.0, 200.0, 80.0),
        }
        customer["TotalCharges"] = customer["MonthlyCharges"] * max(customer["tenure"], 1)
        submitted = st.form_submit_button("Predict")

    if submitted:
        one = pd.DataFrame([customer])
        res = predict(one, artifact).iloc[0]
        st.metric("Churn probability", f"{res['churn_probability']:.1%}", res["risk_level"].upper())
        drivers = explain_customer(artifact["pipeline"], one)
        if drivers:
            st.write("Main drivers (SHAP, positive = increases churn risk):")
            st.bar_chart(pd.DataFrame(drivers).set_index("feature"))

# ---------------------------------------------------------------- model
with tab_model:
    if config.METRICS_PATH.exists():
        metrics = json.loads(config.METRICS_PATH.read_text())
        st.write(f"Best model: **{metrics['best_model']}** - threshold {metrics['threshold']:.2f}")
        st.dataframe(pd.DataFrame(metrics["models"]).T.round(3))
    for fig in ("roc_curves.png", "confusion_matrix.png", "shap_summary.png", "feature_importance.png"):
        path = config.FIGURES_DIR / fig
        if path.exists():
            st.image(str(path), caption=fig)
