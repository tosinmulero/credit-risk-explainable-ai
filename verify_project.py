from pathlib import Path

REQUIRED = [
    "models/selected_probability_model.joblib",
    "reports/metrics/final_test_metrics.json",
    "reports/metrics/final_model_manifest.json",
    "reports/metrics/mlflow_final_run.json",
    "reports/metrics/shap_semantic_feature_importance.csv",
    "reports/metrics/fairness_subgroup_metrics.csv",
    "reports/metrics/drift_summary.csv",
    "docs/MODEL_CARD.md",
    "docs/ARCHITECTURE.md",
    "docs/FINAL_PROJECT_SUMMARY.md",
    "src/credit_risk/api/app.py",
    "src/credit_risk/dashboard/app.py",
    ".github/workflows/ci.yml",
]

root = Path.cwd()
missing = []
print("=" * 72)
print("STAGE 20 - FINAL PROJECT VERIFICATION")
print("=" * 72)
for item in REQUIRED:
    exists = (root / item).exists()
    print(f"{'PASS' if exists else 'MISSING':7} {item}")
    if not exists:
        missing.append(item)
if missing:
    raise SystemExit(f"Verification failed: {len(missing)} required artifacts are missing.")
print("\nFINAL PROJECT VERIFICATION PASSED")
