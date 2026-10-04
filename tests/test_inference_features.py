import pandas as pd

from credit_risk.features.inference import build_inference_features, load_model_features


def sample_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "LIMIT_BAL": 200000.0,
                "SEX": 2,
                "EDUCATION": 2,
                "MARRIAGE": 2,
                "AGE": 35,
                "PAY_0": 0,
                "PAY_2": 0,
                "PAY_3": 0,
                "PAY_4": 0,
                "PAY_5": 0,
                "PAY_6": 0,
                "BILL_AMT1": 50000.0,
                "BILL_AMT2": 48000.0,
                "BILL_AMT3": 47000.0,
                "BILL_AMT4": 46000.0,
                "BILL_AMT5": 45000.0,
                "BILL_AMT6": 44000.0,
                "PAY_AMT1": 5000.0,
                "PAY_AMT2": 5000.0,
                "PAY_AMT3": 5000.0,
                "PAY_AMT4": 5000.0,
                "PAY_AMT5": 5000.0,
                "PAY_AMT6": 5000.0,
            }
        ]
    )


def test_inference_features_match_manifest():
    features = build_inference_features(sample_raw_frame())
    assert list(features.columns) == load_model_features()
    assert len(features) == 1
    assert features.isna().sum().sum() == 0
