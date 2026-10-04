from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from credit_risk.features.inference import build_inference_features
from credit_risk.utils.paths import METRICS_DIR, MODELS_DIR

st.set_page_config(page_title="Credit Risk Intelligence", page_icon="📊", layout="wide")
st.title("Credit Risk Intelligence")
st.caption("Explainable portfolio demonstration using a tuned XGBoost default-risk model.")

model = joblib.load(MODELS_DIR / "selected_probability_model.joblib")
threshold = float(
    json.loads((METRICS_DIR / "selected_threshold.json").read_text(encoding="utf-8"))[
        "selected_threshold"
    ]
)

final_path = METRICS_DIR / "final_test_metrics.json"
if final_path.exists():
    m = json.loads(final_path.read_text(encoding="utf-8"))["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Final ROC-AUC", f"{m['roc_auc']:.3f}")
    c2.metric("Final PR-AUC", f"{m['average_precision_pr_auc']:.3f}")
    c3.metric("Decision threshold", f"{threshold:.3f}")

st.subheader("Customer input")
with st.form("risk_form"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        limit_bal = st.number_input("Credit limit", min_value=0.0, value=200000.0)
        sex = st.selectbox("Sex code", [1, 2], index=1)
        education = st.selectbox("Education code", [0, 1, 2, 3, 4, 5, 6], index=2)
        marriage = st.selectbox("Marriage code", [0, 1, 2, 3], index=2)
        age = st.number_input("Age", min_value=18, max_value=120, value=35)
    with c2:
        pay = {
            field: st.number_input(field, min_value=-2, max_value=9, value=0)
            for field in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        }
    with c3:
        bills = {
            f"BILL_AMT{i}": st.number_input(f"BILL_AMT{i}", value=50000.0) for i in range(1, 7)
        }
    with c4:
        payments = {
            f"PAY_AMT{i}": st.number_input(f"PAY_AMT{i}", min_value=0.0, value=5000.0)
            for i in range(1, 7)
        }
    submitted = st.form_submit_button("Score default risk")

if submitted:
    raw = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        **pay,
        **bills,
        **payments,
    }
    features = build_inference_features(pd.DataFrame([raw]))
    probability = float(model.predict_proba(features)[:, 1][0])
    c1, c2 = st.columns(2)
    c1.metric("Estimated default probability", f"{probability:.2%}")
    c2.metric("Operating threshold", f"{threshold:.2%}")
    st.warning("Elevated-risk flag: YES") if probability >= threshold else st.success(
        "Elevated-risk flag: NO"
    )

importance_path = METRICS_DIR / "shap_semantic_feature_importance.csv"
if importance_path.exists():
    st.subheader("Global model drivers")
    importance = pd.read_csv(importance_path).head(12)
    st.bar_chart(importance.set_index("semantic_feature")["mean_abs_shap"], horizontal=True)

st.caption("Portfolio demonstration only. This model is not suitable for real lending decisions.")
