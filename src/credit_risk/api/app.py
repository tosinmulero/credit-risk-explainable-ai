from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI

from credit_risk.api.schemas import CreditRiskRequest, CreditRiskResponse
from credit_risk.features.inference import build_inference_features
from credit_risk.utils.paths import METRICS_DIR, MODELS_DIR

app = FastAPI(
    title="Credit Risk Intelligence API",
    version="1.0.0",
    description="Explainable default-risk scoring API for the portfolio project.",
)


@lru_cache(maxsize=1)
def load_runtime():
    model = joblib.load(MODELS_DIR / "selected_probability_model.joblib")
    payload = json.loads((METRICS_DIR / "selected_threshold.json").read_text(encoding="utf-8"))
    return model, float(payload["selected_threshold"])


@app.get("/health")
def health() -> dict[str, str]:
    load_runtime()
    return {"status": "ok", "model": "selected_probability_model"}


@app.post("/predict", response_model=CreditRiskResponse)
def predict(request: CreditRiskRequest) -> CreditRiskResponse:
    model, threshold = load_runtime()
    features = build_inference_features(pd.DataFrame([request.model_dump()]))
    probability = float(model.predict_proba(features)[:, 1][0])
    return CreditRiskResponse(
        default_probability=probability,
        decision_threshold=threshold,
        elevated_risk_flag=probability >= threshold,
        model_name="Tuned XGBoost",
    )
