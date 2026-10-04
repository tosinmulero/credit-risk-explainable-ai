# Architecture

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
