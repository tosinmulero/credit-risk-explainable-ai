from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.importance import get_param_importances
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from credit_risk.utils.paths import (
    DOCS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)

RANDOM_STATE = 42
TARGET = "DEFAULT_NEXT_MONTH"
IDENTIFIER = "CLIENT_ID"

N_TRIALS = 30
N_SPLITS = 5
THRESHOLD = 0.50

STUDY_NAME = "xgboost_credit_risk_pr_auc"
STUDY_DB = METRICS_DIR / "xgboost_optuna.db"


def load_configuration() -> tuple[list[str], list[str]]:
    configuration = json.loads(
        (METRICS_DIR / "primary_model_features.json").read_text(encoding="utf-8")
    )

    return (
        configuration["model_features"],
        configuration["categorical_model_features"],
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet")

    validation = pd.read_parquet(PROCESSED_DATA_DIR / "validation.parquet")

    return train, validation


def create_preprocessor(
    model_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric_features = [
        feature for feature in model_features if feature not in categorical_features
    ]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_estimator(
    parameters: dict[str, float | int],
) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        **parameters,
    )


def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, float | int]:
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": THRESHOLD,
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "average_precision_pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def create_objective(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    model_features: list[str],
    categorical_features: list[str],
):
    cross_validation = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    def objective(
        trial: optuna.Trial,
    ) -> float:
        parameters = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                250,
                1000,
                step=50,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.20,
                log=True,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                8,
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                1,
                10,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.60,
                1.00,
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.60,
                1.00,
            ),
            "gamma": trial.suggest_float(
                "gamma",
                0.0,
                5.0,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-4,
                10.0,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-3,
                20.0,
                log=True,
            ),
        }

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor(
                        model_features,
                        categorical_features,
                    ),
                ),
                (
                    "model",
                    make_estimator(parameters),
                ),
            ]
        )

        scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            scoring="average_precision",
            cv=cross_validation,
            n_jobs=1,
        )

        mean_score = float(scores.mean())

        std_score = float(scores.std())

        trial.set_user_attr(
            "cv_ap_std",
            std_score,
        )

        return mean_score

    return objective


def load_benchmark_metrics() -> dict[str, float]:
    comparison = pd.read_csv(METRICS_DIR / "advanced_model_comparison.csv")

    row = comparison.loc[comparison["model"] == "XGBoost"].iloc[0]

    return {
        "roc_auc": float(row["roc_auc"]),
        "average_precision_pr_auc": float(row["average_precision_pr_auc"]),
        "accuracy": float(row["accuracy"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
        "log_loss": float(row["log_loss"]),
        "brier_score": float(row["brier_score"]),
    }


def save_trial_history(
    study: optuna.Study,
) -> pd.DataFrame:
    trials = study.trials_dataframe()

    trials.to_csv(
        METRICS_DIR / "xgboost_optuna_trials.csv",
        index=False,
    )

    completed = trials.loc[trials["state"] == "COMPLETE"].copy()

    if not completed.empty:
        completed = completed.sort_values("number")

        completed["running_best_pr_auc"] = completed["value"].cummax()

        figure, axis = plt.subplots(figsize=(9, 6))

        axis.plot(
            completed["number"],
            completed["value"],
            marker="o",
            linestyle="",
            label="Trial PR-AUC",
        )

        axis.plot(
            completed["number"],
            completed["running_best_pr_auc"],
            label="Running best",
        )

        axis.set_xlabel("Optuna trial")

        axis.set_ylabel("5-fold CV Average Precision")

        axis.set_title("XGBoost Hyperparameter Optimisation History")

        axis.legend()

        figure.tight_layout()

        figure.savefig(
            FIGURES_DIR / "14_xgboost_optuna_history.png",
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(figure)

    return trials


def save_parameter_importance(
    study: optuna.Study,
) -> dict[str, float]:
    try:
        importance = get_param_importances(study)
    except Exception:
        importance = {}

    if not importance:
        return {}

    frame = pd.DataFrame(
        {
            "parameter": list(importance.keys()),
            "importance": list(importance.values()),
        }
    )

    frame.to_csv(
        METRICS_DIR / "xgboost_optuna_parameter_importance.csv",
        index=False,
    )

    plot_frame = frame.sort_values(
        "importance",
        ascending=True,
    )

    figure, axis = plt.subplots(figsize=(9, 6))

    axis.barh(
        plot_frame["parameter"],
        plot_frame["importance"],
    )

    axis.set_xlabel("Optuna importance")

    axis.set_title("XGBoost Hyperparameter Importance")

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "15_xgboost_parameter_importance.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return {str(key): float(value) for key, value in importance.items()}


def save_documentation(
    study: optuna.Study,
    benchmark_metrics: dict[str, float],
    tuned_metrics: dict[str, float | int],
    selected_variant: str,
) -> None:
    lines = [
        "# XGBoost Hyperparameter Optimisation",
        "",
        "## Objective",
        "",
        "Optimise the provisional XGBoost leader using Optuna while "
        "preserving the untouched validation and test sets.",
        "",
        "## Optimisation Design",
        "",
        f"- Optuna trials: **{N_TRIALS}**",
        f"- Cross-validation folds: **{N_SPLITS}**",
        "- Cross-validation: stratified and shuffled",
        "- Primary optimisation metric: Average Precision / PR-AUC",
        "- Hyperparameter search data: training set only",
        "- Validation data excluded from Optuna search",
        "- Test data not accessed",
        "",
        "## Best Cross-Validated Result",
        "",
        (f"- Best training CV PR-AUC: **{study.best_value:.4f}**"),
        "",
        "## Best Hyperparameters",
        "",
        "| Parameter | Value |",
        "|---|---:|",
    ]

    for name, value in study.best_params.items():
        lines.append(f"| {name} | {value} |")

    lines.extend(
        [
            "",
            "## Validation Comparison",
            "",
            "| Metric | Benchmark XGBoost | Tuned XGBoost |",
            "|---|---:|---:|",
            (f"| ROC-AUC | {benchmark_metrics['roc_auc']:.4f} | {tuned_metrics['roc_auc']:.4f} |"),
            (
                "| PR-AUC | "
                f"{benchmark_metrics['average_precision_pr_auc']:.4f} | "
                f"{tuned_metrics['average_precision_pr_auc']:.4f} |"
            ),
            (
                "| Precision | "
                f"{benchmark_metrics['precision']:.4f} | "
                f"{tuned_metrics['precision']:.4f} |"
            ),
            (f"| Recall | {benchmark_metrics['recall']:.4f} | {tuned_metrics['recall']:.4f} |"),
            (f"| F1 | {benchmark_metrics['f1']:.4f} | {tuned_metrics['f1']:.4f} |"),
            (
                "| Log Loss | "
                f"{benchmark_metrics['log_loss']:.4f} | "
                f"{tuned_metrics['log_loss']:.4f} |"
            ),
            (
                "| Brier Score | "
                f"{benchmark_metrics['brier_score']:.4f} | "
                f"{tuned_metrics['brier_score']:.4f} |"
            ),
            "",
            "## Selected XGBoost Variant",
            "",
            f"**{selected_variant}**",
            "",
            "The selection uses validation PR-AUC first and ROC-AUC "
            "second. Threshold-specific metrics are not used as the "
            "primary selection criterion because the operating threshold "
            "will be optimised separately.",
            "",
            "## Governance",
            "",
            "The test set remains sealed. Hyperparameter optimisation "
            "does not use validation observations.",
        ]
    )

    (DOCS_DIR / "XGBOOST_OPTIMISATION.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 72)
    print("STAGE 8 - XGBOOST OPTUNA HYPERPARAMETER OPTIMISATION")
    print("=" * 72)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_features, categorical_features = load_configuration()

    train, validation = load_data()

    x_train = train[model_features].copy()

    y_train = train[TARGET].copy()

    x_validation = validation[model_features].copy()

    y_validation = validation[TARGET].copy()

    print(f"Training observations:   {len(train):,}")

    print(f"Validation observations: {len(validation):,}")

    print(f"Candidate features:      {len(model_features)}")

    print(f"Optuna target trials:    {N_TRIALS}")

    print(f"CV folds per trial:      {N_SPLITS}")

    storage = "sqlite:///" + STUDY_DB.as_posix()

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
    )

    completed_trials = sum(trial.state == TrialState.COMPLETE for trial in study.trials)

    remaining_trials = max(
        0,
        N_TRIALS - completed_trials,
    )

    print(f"Existing completed trials: {completed_trials}")

    print(f"Trials to run now:         {remaining_trials}")

    if remaining_trials > 0:
        objective = create_objective(
            x_train,
            y_train,
            model_features,
            categorical_features,
        )

        study.optimize(
            objective,
            n_trials=remaining_trials,
            show_progress_bar=True,
        )

    print("")
    print("OPTUNA BEST RESULT")
    print("-" * 72)

    print(f"Best 5-fold CV PR-AUC: {study.best_value:.6f}")

    print("")
    print("BEST PARAMETERS")
    print("-" * 72)

    for name, value in study.best_params.items():
        print(f"{name}: {value}")

    best_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(
                    model_features,
                    categorical_features,
                ),
            ),
            (
                "model",
                make_estimator(study.best_params),
            ),
        ]
    )

    print("")
    print("[INFO] Fitting tuned XGBoost on full training set...")

    best_pipeline.fit(
        x_train,
        y_train,
    )

    validation_probabilities = best_pipeline.predict_proba(x_validation)[:, 1]

    tuned_metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
    )

    benchmark_metrics = load_benchmark_metrics()

    model_path = MODELS_DIR / "xgboost_tuned.joblib"

    joblib.dump(
        best_pipeline,
        model_path,
    )

    predictions = pd.DataFrame(
        {
            IDENTIFIER: validation[IDENTIFIER].to_numpy(),
            "actual_default": (y_validation.to_numpy()),
            "predicted_probability": (validation_probabilities),
            "predicted_class_050": (validation_probabilities >= THRESHOLD).astype(int),
        }
    )

    predictions.to_csv(
        METRICS_DIR / "xgboost_tuned_validation_predictions.csv",
        index=False,
    )

    comparison = pd.DataFrame(
        [
            {
                "variant": "Benchmark XGBoost",
                **benchmark_metrics,
            },
            {
                "variant": "Tuned XGBoost",
                **{
                    key: value
                    for key, value in tuned_metrics.items()
                    if key
                    in {
                        "roc_auc",
                        "average_precision_pr_auc",
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "log_loss",
                        "brier_score",
                    }
                },
            },
        ]
    )

    comparison.to_csv(
        METRICS_DIR / "xgboost_tuning_comparison.csv",
        index=False,
    )

    tuned_ap = float(tuned_metrics["average_precision_pr_auc"])

    benchmark_ap = float(benchmark_metrics["average_precision_pr_auc"])

    tuned_roc = float(tuned_metrics["roc_auc"])

    benchmark_roc = float(benchmark_metrics["roc_auc"])

    tuned_wins = tuned_ap > benchmark_ap or (
        np.isclose(
            tuned_ap,
            benchmark_ap,
        )
        and tuned_roc > benchmark_roc
    )

    selected_variant = "Tuned XGBoost" if tuned_wins else "Benchmark XGBoost"

    parameter_importance = save_parameter_importance(study)

    save_trial_history(study)

    result_payload = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "study_name": STUDY_NAME,
        "completed_trials": int(sum(trial.state == TrialState.COMPLETE for trial in study.trials)),
        "cv_folds": N_SPLITS,
        "optimisation_metric": ("average_precision"),
        "best_cv_pr_auc": float(study.best_value),
        "best_params": study.best_params,
        "parameter_importance": (parameter_importance),
        "benchmark_validation_metrics": (benchmark_metrics),
        "tuned_validation_metrics": (tuned_metrics),
        "selected_xgboost_variant": (selected_variant),
        "test_set_accessed": False,
    }

    (METRICS_DIR / "xgboost_optuna_best_params.json").write_text(
        json.dumps(
            result_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_documentation(
        study,
        benchmark_metrics,
        tuned_metrics,
        selected_variant,
    )

    print("")
    print("=" * 72)
    print("VALIDATION COMPARISON")
    print("=" * 72)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("")
    print("TUNED MODEL CONFUSION MATRIX")
    print("-" * 72)

    print(f"True negatives:  {tuned_metrics['true_negatives']:,}")

    print(f"False positives: {tuned_metrics['false_positives']:,}")

    print(f"False negatives: {tuned_metrics['false_negatives']:,}")

    print(f"True positives:  {tuned_metrics['true_positives']:,}")

    print("")
    print("SELECTED XGBOOST VARIANT:")

    print(selected_variant)

    print("")
    print("TEST SET ACCESSED: False")

    print("")
    print("STAGE 8 XGBOOST OPTIMISATION PASSED")

    print("=" * 72)


if __name__ == "__main__":
    main()
