

<!-- FINAL-PORTFOLIO-SUMMARY -->

## Final portfolio model

End-to-end explainable credit-risk workflow using Logistic Regression, Random Forest, XGBoost, LightGBM, Optuna, SHAP, MLflow, FastAPI, Streamlit, monitoring, tests, and CI.

**Final sealed-test ROC-AUC:** 0.7834  
**Final sealed-test PR-AUC:** 0.5622  
**Final Brier score:** 0.1341  
**Frozen operating threshold:** 0.145

### API

```powershell
uvicorn credit_risk.api.app:app --reload --port 8000
```

### Dashboard

```powershell
streamlit run src/credit_risk/dashboard/app.py
```

See `docs/MODEL_CARD.md` and `docs/ARCHITECTURE.md`.
