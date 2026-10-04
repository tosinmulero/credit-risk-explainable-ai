from __future__ import annotations

import json
from pathlib import Path

import mlflow

from credit_risk.utils.paths import (
    DOCS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_NAME = "Credit Risk Intelligence"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    print("=" * 72)
    print("STAGE 14 - MLFLOW EXPERIMENT TRACKING")
    print("=" * 72)

    final_metrics = load_json(METRICS_DIR / "final_test_metrics.json")

    tuning = load_json(METRICS_DIR / "xgboost_optuna_best_params.json")

    threshold = load_json(METRICS_DIR / "selected_threshold.json")

    probability = load_json(METRICS_DIR / "selected_probability_model.json")

    database_path = PROJECT_ROOT / "mlflow.db"

    tracking_uri = "sqlite:///" + database_path.as_posix()

    print(f"MLflow tracking database: {database_path}")

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="final_selected_xgboost") as run:
        parameters = {f"xgb_{name}": value for name, value in tuning["best_params"].items()}

        parameters.update(
            {
                "selected_threshold": (threshold["selected_threshold"]),
                "probability_variant": (probability["selected_probability_variant"]),
                "optuna_trials": (tuning["completed_trials"]),
                "cv_folds": (tuning["cv_folds"]),
            }
        )

        mlflow.log_params(parameters)

        for name, value in final_metrics["metrics"].items():
            if isinstance(
                value,
                (int, float),
            ):
                mlflow.log_metric(
                    f"test_{name}",
                    float(value),
                )

        artifacts = [
            MODELS_DIR / "selected_probability_model.joblib",
            METRICS_DIR / "final_test_metrics.json",
            METRICS_DIR / "selected_threshold.json",
            METRICS_DIR / "xgboost_optuna_best_params.json",
            METRICS_DIR / "selected_probability_model.json",
            DOCS_DIR / "FINAL_TEST_EVALUATION.md",
            DOCS_DIR / "SHAP_EXPLAINABILITY.md",
            DOCS_DIR / "FAIRNESS_AUDIT.md",
            DOCS_DIR / "DRIFT_MONITORING.md",
            DOCS_DIR / "MODEL_CARD.md",
        ]

        for artifact in artifacts:
            if artifact.exists():
                mlflow.log_artifact(str(artifact))

        if FIGURES_DIR.exists():
            for figure in FIGURES_DIR.glob("*.png"):
                mlflow.log_artifact(
                    str(figure),
                    artifact_path="figures",
                )

        run_information = {
            "run_id": (run.info.run_id),
            "experiment_id": (run.info.experiment_id),
            "experiment_name": (EXPERIMENT_NAME),
            "tracking_uri": (tracking_uri),
            "backend": ("SQLite"),
            "final_test_logged": True,
        }

    (METRICS_DIR / "mlflow_final_run.json").write_text(
        json.dumps(
            run_information,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print(f"MLflow Run ID: {run_information['run_id']}")

    print(f"Experiment ID: {run_information['experiment_id']}")

    print(f"Backend: {run_information['backend']}")

    print(f"Tracking URI: {run_information['tracking_uri']}")

    print("")
    print("STAGE 14 MLFLOW TRACKING PASSED")

    print("=" * 72)


if __name__ == "__main__":
    main()
