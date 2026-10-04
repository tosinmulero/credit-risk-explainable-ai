import json

import joblib
import pytest

from credit_risk.features.inference import build_inference_features
from credit_risk.utils.paths import METRICS_DIR, MODELS_DIR
from tests.test_inference_features import sample_raw_frame


def test_selected_model_predicts_probability():
    model_path = MODELS_DIR / "selected_probability_model.joblib"
    threshold_path = METRICS_DIR / "selected_threshold.json"
    if not model_path.exists() or not threshold_path.exists():
        pytest.skip("Runtime model artifacts are not present in this environment.")
    model = joblib.load(model_path)
    p = float(model.predict_proba(build_inference_features(sample_raw_frame()))[:, 1][0])
    threshold = float(json.loads(threshold_path.read_text(encoding="utf-8"))["selected_threshold"])
    assert 0.0 <= p <= 1.0
    assert 0.0 < threshold < 1.0
