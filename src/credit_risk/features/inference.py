from __future__ import annotations

import json

import pandas as pd

from credit_risk.features.build_features import add_behavioural_features, add_semantic_categories
from credit_risk.utils.paths import METRICS_DIR

RAW_INPUT_FEATURES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


def load_model_features() -> list[str]:
    payload = json.loads((METRICS_DIR / "primary_model_features.json").read_text(encoding="utf-8"))
    return payload["model_features"]


def build_inference_features(raw: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(RAW_INPUT_FEATURES).difference(raw.columns))
    if missing:
        raise ValueError(f"Missing raw input features: {missing}")
    work = raw[RAW_INPUT_FEATURES].copy()
    work = add_semantic_categories(work)
    work = add_behavioural_features(work)
    return work[load_model_features()].copy()
