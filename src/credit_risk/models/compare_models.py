from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
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
THRESHOLD = 0.50


def load_configuration() -> tuple[list[str], list[str]]:
    config_path = METRICS_DIR / "primary_model_features.json"

    config = json.loads(config_path.read_text(encoding="utf-8"))

    return (
        config["model_features"],
        config["categorical_model_features"],
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


def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
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


def build_models() -> dict[str, object]:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


def load_logistic_baseline() -> dict[str, float | int | str]:
    metrics_path = METRICS_DIR / "logistic_baseline_validation_metrics.json"

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    metrics = payload["metrics"]

    return {
        "model": "Logistic Regression",
        "fit_seconds": np.nan,
        **metrics,
    }


def save_feature_importance(
    model_name: str,
    pipeline: Pipeline,
) -> None:
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return

    preprocessor = pipeline.named_steps["preprocessor"]

    names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame(
        {
            "feature": names,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    safe_name = model_name.lower().replace(" ", "_")

    importance.to_csv(
        METRICS_DIR / f"{safe_name}_feature_importance.csv",
        index=False,
    )


def save_auc_comparison(
    comparison: pd.DataFrame,
) -> None:
    plot_data = comparison.sort_values("average_precision_pr_auc")

    figure, axis = plt.subplots(figsize=(9, 6))

    positions = np.arange(len(plot_data))

    width = 0.35

    axis.bar(
        positions - width / 2,
        plot_data["roc_auc"],
        width,
        label="ROC-AUC",
    )

    axis.bar(
        positions + width / 2,
        plot_data["average_precision_pr_auc"],
        width,
        label="PR-AUC / AP",
    )

    axis.set_xticks(
        positions,
        plot_data["model"],
        rotation=20,
        ha="right",
    )

    axis.set_ylim(
        0,
        1,
    )

    axis.set_ylabel("Score")

    axis.set_title("Validation Model Discrimination Comparison")

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "11_model_auc_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_roc_comparison(
    probability_store: dict[str, np.ndarray],
    y_validation: pd.Series,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))

    for model_name, probabilities in probability_store.items():
        fpr, tpr, _ = roc_curve(
            y_validation,
            probabilities,
        )

        auc = roc_auc_score(
            y_validation,
            probabilities,
        )

        axis.plot(
            fpr,
            tpr,
            label=f"{model_name} ({auc:.3f})",
        )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    axis.set_xlabel("False positive rate")

    axis.set_ylabel("True positive rate")

    axis.set_title("Validation ROC Curves")

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "12_model_roc_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_pr_comparison(
    probability_store: dict[str, np.ndarray],
    y_validation: pd.Series,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))

    for model_name, probabilities in probability_store.items():
        precision, recall, _ = precision_recall_curve(
            y_validation,
            probabilities,
        )

        ap = average_precision_score(
            y_validation,
            probabilities,
        )

        axis.plot(
            recall,
            precision,
            label=f"{model_name} ({ap:.3f})",
        )

    prevalence = float(y_validation.mean())

    axis.axhline(
        prevalence,
        linestyle="--",
        label=f"Prevalence ({prevalence:.3f})",
    )

    axis.set_xlabel("Recall")

    axis.set_ylabel("Precision")

    axis.set_title("Validation Precision-Recall Curves")

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "13_model_pr_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def load_logistic_probabilities() -> np.ndarray:
    path = METRICS_DIR / "logistic_baseline_validation_predictions.csv"

    frame = pd.read_csv(path)

    return frame["predicted_probability"].to_numpy()


def save_documentation(
    comparison: pd.DataFrame,
    best_model: str,
) -> None:
    best_row = comparison.loc[comparison["model"] == best_model].iloc[0]

    lines = [
        "# Advanced Model Benchmarking",
        "",
        "## Objective",
        "",
        "Compare multiple supervised-learning model families under "
        "the same train/validation design before hyperparameter tuning.",
        "",
        "## Models",
        "",
        "- Logistic Regression",
        "- Random Forest",
        "- XGBoost",
        "- LightGBM",
        "",
        "## Evaluation Policy",
        "",
        "All advanced models are trained on the same 18,000-row "
        "training set and evaluated on the same 6,000-row validation set.",
        "",
        "The 6,000-row test set remains untouched.",
        "",
        "Model selection at this stage prioritises Average Precision "
        "(PR-AUC), followed by ROC-AUC. Probability-quality metrics are "
        "also retained for later calibration analysis.",
        "",
        "## Validation Comparison",
        "",
        "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Log Loss | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.model} | "
            f"{row.roc_auc:.4f} | "
            f"{row.average_precision_pr_auc:.4f} | "
            f"{row.precision:.4f} | "
            f"{row.recall:.4f} | "
            f"{row.f1:.4f} | "
            f"{row.log_loss:.4f} | "
            f"{row.brier_score:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Provisional Validation Leader",
            "",
            f"**{best_model}**",
            "",
            (f"Validation PR-AUC: **{best_row['average_precision_pr_auc']:.4f}**"),
            (f"Validation ROC-AUC: **{best_row['roc_auc']:.4f}**"),
            "",
            "This is not the final production model. The leading "
            "candidate will undergo hyperparameter optimisation, "
            "probability calibration, threshold optimisation, "
            "explainability and final test-set evaluation.",
        ]
    )

    (DOCS_DIR / "ADVANCED_MODEL_COMPARISON.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 72)
    print("STAGE 7 - ADVANCED MODEL BENCHMARKING")
    print("=" * 72)

    model_features, categorical_features = load_configuration()

    train, validation = load_data()

    x_train = train[model_features].copy()

    y_train = train[TARGET].copy()

    x_validation = validation[model_features].copy()

    y_validation = validation[TARGET].copy()

    models = build_models()

    result_rows = [load_logistic_baseline()]

    probability_store = {"Logistic Regression": (load_logistic_probabilities())}

    prediction_output = pd.DataFrame(
        {
            IDENTIFIER: validation[IDENTIFIER].to_numpy(),
            "actual_default": (y_validation.to_numpy()),
        }
    )

    for model_name, estimator in models.items():
        print("")
        print("-" * 72)
        print(f"Training: {model_name}")
        print("-" * 72)

        preprocessor = create_preprocessor(
            model_features,
            categorical_features,
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    estimator,
                ),
            ]
        )

        start = time.perf_counter()

        pipeline.fit(
            x_train,
            y_train,
        )

        fit_seconds = time.perf_counter() - start

        probabilities = pipeline.predict_proba(x_validation)[:, 1]

        predictions = (probabilities >= THRESHOLD).astype(int)

        metrics = calculate_metrics(
            y_validation,
            probabilities,
            predictions,
        )

        result_rows.append(
            {
                "model": model_name,
                "fit_seconds": fit_seconds,
                **metrics,
            }
        )

        probability_store[model_name] = probabilities

        safe_name = model_name.lower().replace(" ", "_")

        prediction_output[f"{safe_name}_probability"] = probabilities

        prediction_output[f"{safe_name}_class_050"] = predictions

        model_path = MODELS_DIR / f"{safe_name}_benchmark.joblib"

        joblib.dump(
            pipeline,
            model_path,
        )

        save_feature_importance(
            model_name,
            pipeline,
        )

        print(f"ROC-AUC:    {metrics['roc_auc']:.4f}")

        print(f"PR-AUC:     {metrics['average_precision_pr_auc']:.4f}")

        print(f"Precision:  {metrics['precision']:.4f}")

        print(f"Recall:     {metrics['recall']:.4f}")

        print(f"F1:         {metrics['f1']:.4f}")

        print(f"Log Loss:   {metrics['log_loss']:.4f}")

        print(f"Brier:      {metrics['brier_score']:.4f}")

        print(f"Fit time:   {fit_seconds:.2f}s")

    comparison = pd.DataFrame(result_rows)

    comparison = comparison.sort_values(
        [
            "average_precision_pr_auc",
            "roc_auc",
        ],
        ascending=False,
    ).reset_index(drop=True)

    comparison.to_csv(
        METRICS_DIR / "advanced_model_comparison.csv",
        index=False,
    )

    prediction_output.to_csv(
        METRICS_DIR / "advanced_model_validation_predictions.csv",
        index=False,
    )

    best_model = str(comparison.iloc[0]["model"])

    best_row = comparison.iloc[0]

    provisional = {
        "selected_at_utc": datetime.now(UTC).isoformat(),
        "selection_stage": ("untuned_validation_benchmark"),
        "primary_metric": ("average_precision_pr_auc"),
        "secondary_metric": ("roc_auc"),
        "provisional_best_model": (best_model),
        "validation_pr_auc": float(best_row["average_precision_pr_auc"]),
        "validation_roc_auc": float(best_row["roc_auc"]),
        "test_set_accessed": False,
    }

    (METRICS_DIR / "provisional_best_model.json").write_text(
        json.dumps(
            provisional,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_auc_comparison(comparison)

    save_roc_comparison(
        probability_store,
        y_validation,
    )

    save_pr_comparison(
        probability_store,
        y_validation,
    )

    save_documentation(
        comparison,
        best_model,
    )

    print("")
    print("=" * 72)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 72)

    display_columns = [
        "model",
        "roc_auc",
        "average_precision_pr_auc",
        "precision",
        "recall",
        "f1",
        "log_loss",
        "brier_score",
        "fit_seconds",
    ]

    print(
        comparison[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("")
    print("PROVISIONAL VALIDATION LEADER:")

    print(best_model)

    print("")
    print("IMPORTANT: TEST SET ACCESSED = False")

    print("")
    print("STAGE 7 ADVANCED MODEL BENCHMARKING PASSED")

    print("=" * 72)


if __name__ == "__main__":
    main()
