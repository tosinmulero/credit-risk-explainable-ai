from __future__ import annotations

import json
from pathlib import Path

from credit_risk.utils.paths import DOCS_DIR, METRICS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def read_json(name: str) -> dict:
    return json.loads((METRICS_DIR / name).read_text(encoding="utf-8"))


def main() -> None:
    print("=" * 72)
    print("STAGE 19 - MODEL CARD, ARCHITECTURE AND README")
    print("=" * 72)

    final_test = read_json("final_test_metrics.json")
    threshold = read_json("selected_threshold.json")
    probability = read_json("selected_probability_model.json")
    tuning = read_json("xgboost_optuna_best_params.json")
    m = final_test["metrics"]

    model_card = f"""# Model Card — Credit Risk Intelligence

## Model overview

- Model family: XGBoost
- Probability variant: {probability["selected_probability_variant"]}
- Optuna trials: {tuning["completed_trials"]}
- Operating threshold: {threshold["selected_threshold"]:.3f}
- Final holdout observations: {m["observations"]:,}

## Final holdout performance

| Metric | Value |
|---|---:|
| ROC-AUC | {m["roc_auc"]:.4f} |
| PR-AUC / Average Precision | {m["average_precision_pr_auc"]:.4f} |
| Precision | {m["precision"]:.4f} |
| Recall | {m["recall"]:.4f} |
| F1 | {m["f1"]:.4f} |
| Log loss | {m["log_loss"]:.4f} |
| Brier score | {m["brier_score"]:.4f} |
| ECE | {m["expected_calibration_error"]:.4f} |

## Intended use

Portfolio demonstration of a complete credit-risk machine-learning workflow.

## Limitations

This model is built on a public historical benchmark dataset and is not suitable for real lending or underwriting decisions. A production system would require representative current data, formal model-risk governance, legal and regulatory review, bias testing, security controls, and institution-specific cost assumptions.

## Responsible use

`SEX` and `AGE` are excluded from the primary model and retained for audit analysis. This design choice does not by itself establish fairness or legal compliance. SHAP explanations describe model behaviour and are not causal.
"""
    (DOCS_DIR / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")

    architecture = """# Architecture

```text
Raw benchmark data
    |
    v
Validation + semantic schema
    |
    v
Feature engineering
    |
    +--> Train 60%
    +--> Validation 20%
    +--> Sealed Test 20%
            |
            v
Logistic baseline
    |
    v
Random Forest / XGBoost / LightGBM
    |
    v
Optuna-tuned XGBoost
    |
    v
Probability calibration comparison
    |
    v
Cost-sensitive threshold policy
    |
    +--> SHAP explainability
    +--> Fairness audit
    |
    v
Final sealed test evaluation
    |
    +--> MLflow
    +--> FastAPI
    +--> Streamlit
    +--> Drift monitoring
    +--> Pytest + GitHub Actions
```
"""
    (DOCS_DIR / "ARCHITECTURE.md").write_text(architecture, encoding="utf-8")

    summary = f"""# Final Project Summary

Final holdout ROC-AUC: **{m["roc_auc"]:.4f}**  
Final holdout PR-AUC: **{m["average_precision_pr_auc"]:.4f}**  
Final Brier score: **{m["brier_score"]:.4f}**  
Frozen operating threshold: **{threshold["selected_threshold"]:.3f}**

See `docs/MODEL_CARD.md`, `docs/ARCHITECTURE.md`, `docs/FINAL_TEST_EVALUATION.md`, `docs/SHAP_EXPLAINABILITY.md`, `docs/FAIRNESS_AUDIT.md`, and `docs/DRIFT_MONITORING.md`.
"""
    (DOCS_DIR / "FINAL_PROJECT_SUMMARY.md").write_text(summary, encoding="utf-8")

    readme_path = PROJECT_ROOT / "README.md"
    readme = (
        readme_path.read_text(encoding="utf-8")
        if readme_path.exists()
        else "# Credit Risk Intelligence\n"
    )
    marker = "<!-- FINAL-PORTFOLIO-SUMMARY -->"
    section = f"""{marker}

## Final portfolio model

End-to-end explainable credit-risk workflow using Logistic Regression, Random Forest, XGBoost, LightGBM, Optuna, SHAP, MLflow, FastAPI, Streamlit, monitoring, tests, and CI.

**Final sealed-test ROC-AUC:** {m["roc_auc"]:.4f}  
**Final sealed-test PR-AUC:** {m["average_precision_pr_auc"]:.4f}  
**Final Brier score:** {m["brier_score"]:.4f}  
**Frozen operating threshold:** {threshold["selected_threshold"]:.3f}

### API

```powershell
uvicorn credit_risk.api.app:app --reload --port 8000
```

### Dashboard

```powershell
streamlit run src/credit_risk/dashboard/app.py
```

See `docs/MODEL_CARD.md` and `docs/ARCHITECTURE.md`.
"""
    if marker in readme:
        readme = readme.split(marker)[0].rstrip() + "\n\n" + section
    else:
        readme = readme.rstrip() + "\n\n" + section
    readme_path.write_text(readme, encoding="utf-8")

    print("[PASS] MODEL_CARD.md")
    print("[PASS] ARCHITECTURE.md")
    print("[PASS] FINAL_PROJECT_SUMMARY.md")
    print("[PASS] README.md final section")
    print("STAGE 19 DOCUMENTATION PASSED")


if __name__ == "__main__":
    main()
