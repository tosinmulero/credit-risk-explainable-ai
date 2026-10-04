from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
)
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
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

CALIBRATION_FOLDS = 5
CALIBRATION_BINS = 10


def load_configuration() -> tuple[
    list[str],
    list[str],
    dict[str, float | int],
]:
    feature_config = json.loads(
        (METRICS_DIR / "primary_model_features.json").read_text(encoding="utf-8")
    )

    tuning_config = json.loads(
        (METRICS_DIR / "xgboost_optuna_best_params.json").read_text(encoding="utf-8")
    )

    return (
        feature_config["model_features"],
        feature_config["categorical_model_features"],
        tuning_config["best_params"],
    )


def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
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


def create_tuned_pipeline(
    model_features: list[str],
    categorical_features: list[str],
    best_params: dict[str, float | int],
) -> Pipeline:
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
        **best_params,
    )

    return Pipeline(
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
                classifier,
            ),
        ]
    )


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = CALIBRATION_BINS,
) -> float:
    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    bin_ids = np.digitize(
        probabilities,
        edges[1:-1],
        right=True,
    )

    total = len(probabilities)
    error = 0.0

    for bin_id in range(bins):
        mask = bin_ids == bin_id

        if not np.any(mask):
            continue

        predicted_mean = float(probabilities[mask].mean())

        observed_mean = float(y_true[mask].mean())

        weight = int(mask.sum()) / total

        error += weight * abs(predicted_mean - observed_mean)

    return float(error)


def maximum_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = CALIBRATION_BINS,
) -> float:
    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    bin_ids = np.digitize(
        probabilities,
        edges[1:-1],
        right=True,
    )

    gaps = []

    for bin_id in range(bins):
        mask = bin_ids == bin_id

        if not np.any(mask):
            continue

        predicted_mean = float(probabilities[mask].mean())

        observed_mean = float(y_true[mask].mean())

        gaps.append(abs(predicted_mean - observed_mean))

    return float(max(gaps) if gaps else 0.0)


def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, float]:
    y_array = y_true.to_numpy(dtype=int)

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
        "expected_calibration_error": (
            expected_calibration_error(
                y_array,
                probabilities,
            )
        ),
        "maximum_calibration_error": (
            maximum_calibration_error(
                y_array,
                probabilities,
            )
        ),
    }


def save_reliability_diagram(
    probabilities: dict[str, np.ndarray],
    y_validation: pd.Series,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    for name, values in probabilities.items():
        observed, predicted = calibration_curve(
            y_validation,
            values,
            n_bins=CALIBRATION_BINS,
            strategy="quantile",
        )

        axis.plot(
            predicted,
            observed,
            marker="o",
            label=name,
        )

    axis.set_xlabel("Mean predicted probability")

    axis.set_ylabel("Observed default rate")

    axis.set_title("Validation Reliability Diagram")

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "16_probability_calibration_reliability.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_probability_distribution(
    probabilities: dict[str, np.ndarray],
) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))

    for name, values in probabilities.items():
        axis.hist(
            values,
            bins=30,
            alpha=0.35,
            label=name,
        )

    axis.set_xlabel("Predicted default probability")

    axis.set_ylabel("Validation observations")

    axis.set_title("Predicted Probability Distributions")

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "17_probability_distribution_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_documentation(
    comparison: pd.DataFrame,
    selected_variant: str,
) -> None:
    selected = comparison.loc[comparison["variant"] == selected_variant].iloc[0]

    lines = [
        "# Probability Calibration",
        "",
        "## Objective",
        "",
        "Evaluate whether probability calibration improves the "
        "probabilistic reliability of the tuned XGBoost model.",
        "",
        "## Calibration Design",
        "",
        "- Base model: tuned XGBoost",
        "- Calibration methods: sigmoid and isotonic",
        "- Calibration folds: 5",
        "- Calibration fitted using training data only",
        "- Validation set used only for method comparison",
        "- Test set not accessed",
        "",
        "## Selection Policy",
        "",
        "The primary calibration metric is Brier score, followed "
        "by log loss. Lower values are better.",
        "",
        "ROC-AUC and PR-AUC are retained to ensure calibration does "
        "not materially damage discrimination.",
        "",
        "## Validation Comparison",
        "",
        "| Variant | ROC-AUC | PR-AUC | Log Loss | Brier | ECE | MCE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.variant} | "
            f"{row.roc_auc:.4f} | "
            f"{row.average_precision_pr_auc:.4f} | "
            f"{row.log_loss:.4f} | "
            f"{row.brier_score:.4f} | "
            f"{row.expected_calibration_error:.4f} | "
            f"{row.maximum_calibration_error:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Selected Probability Variant",
            "",
            f"**{selected_variant}**",
            "",
            f"- Brier score: **{selected['brier_score']:.4f}**",
            f"- Log loss: **{selected['log_loss']:.4f}**",
            (f"- Expected calibration error: **{selected['expected_calibration_error']:.4f}**"),
            "",
            "## Governance",
            "",
            "No validation or test observations are used to fit the calibration models.",
            "",
            "The selected probability model will be passed to the "
            "decision-threshold optimisation stage.",
        ]
    )

    (DOCS_DIR / "PROBABILITY_CALIBRATION.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 72)
    print("STAGE 9 - XGBOOST PROBABILITY CALIBRATION")
    print("=" * 72)

    model_features, categorical_features, best_params = load_configuration()

    train, validation = load_data()

    x_train = train[model_features].copy()

    y_train = train[TARGET].copy()

    x_validation = validation[model_features].copy()

    y_validation = validation[TARGET].copy()

    print(f"Training rows:            {len(train):,}")

    print(f"Validation rows:          {len(validation):,}")

    print(f"Calibration folds:        {CALIBRATION_FOLDS}")

    tuned_model = joblib.load(MODELS_DIR / "xgboost_tuned.joblib")

    uncalibrated_probabilities = tuned_model.predict_proba(x_validation)[:, 1]

    probability_store = {"Uncalibrated tuned XGBoost": (uncalibrated_probabilities)}

    model_store = {"Uncalibrated tuned XGBoost": (tuned_model)}

    cross_validation = StratifiedKFold(
        n_splits=CALIBRATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for method in (
        "sigmoid",
        "isotonic",
    ):
        print("")
        print(f"[INFO] Fitting {method} calibration...")

        base_pipeline = create_tuned_pipeline(
            model_features,
            categorical_features,
            best_params,
        )

        calibrated_model = CalibratedClassifierCV(
            estimator=base_pipeline,
            method=method,
            cv=cross_validation,
            n_jobs=-1,
        )

        calibrated_model.fit(
            x_train,
            y_train,
        )

        probabilities = calibrated_model.predict_proba(x_validation)[:, 1]

        display_name = (
            "Sigmoid calibrated XGBoost" if method == "sigmoid" else "Isotonic calibrated XGBoost"
        )

        probability_store[display_name] = probabilities

        model_store[display_name] = calibrated_model

        safe_name = method + "_calibrated_xgboost.joblib"

        joblib.dump(
            calibrated_model,
            MODELS_DIR / safe_name,
        )

    metric_rows = []

    for name, probabilities in probability_store.items():
        metrics = calculate_metrics(
            y_validation,
            probabilities,
        )

        metric_rows.append(
            {
                "variant": name,
                **metrics,
            }
        )

    comparison = pd.DataFrame(metric_rows)

    comparison = comparison.sort_values(
        [
            "brier_score",
            "log_loss",
            "average_precision_pr_auc",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    selected_variant = str(comparison.iloc[0]["variant"])

    selected_model = model_store[selected_variant]

    selected_model_path = MODELS_DIR / "selected_probability_model.joblib"

    joblib.dump(
        selected_model,
        selected_model_path,
    )

    comparison.to_csv(
        METRICS_DIR / "probability_calibration_comparison.csv",
        index=False,
    )

    predictions = pd.DataFrame(
        {
            IDENTIFIER: validation[IDENTIFIER].to_numpy(),
            "actual_default": (y_validation.to_numpy()),
        }
    )

    for name, probabilities in probability_store.items():
        safe_column = name.lower().replace(" ", "_").replace("-", "_")

        predictions[safe_column] = probabilities

    predictions.to_csv(
        METRICS_DIR / "probability_calibration_validation_predictions.csv",
        index=False,
    )

    selected_row = comparison.iloc[0]

    result = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "base_model": "Tuned XGBoost",
        "calibration_folds": CALIBRATION_FOLDS,
        "selection_primary_metric": ("brier_score"),
        "selection_secondary_metric": ("log_loss"),
        "selected_probability_variant": (selected_variant),
        "selected_metrics": {
            key: float(selected_row[key])
            for key in (
                "roc_auc",
                "average_precision_pr_auc",
                "log_loss",
                "brier_score",
                "expected_calibration_error",
                "maximum_calibration_error",
            )
        },
        "test_set_accessed": False,
    }

    (METRICS_DIR / "selected_probability_model.json").write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_reliability_diagram(
        probability_store,
        y_validation,
    )

    save_probability_distribution(probability_store)

    save_documentation(
        comparison,
        selected_variant,
    )

    print("")
    print("=" * 72)
    print("CALIBRATION VALIDATION COMPARISON")
    print("=" * 72)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("")
    print("SELECTED PROBABILITY VARIANT:")

    print(selected_variant)

    print("")
    print(f"Brier score: {selected_row['brier_score']:.4f}")

    print(f"Log loss:    {selected_row['log_loss']:.4f}")

    print(f"ECE:         {selected_row['expected_calibration_error']:.4f}")

    print(f"MCE:         {selected_row['maximum_calibration_error']:.4f}")

    print("")
    print("TEST SET ACCESSED: False")

    print("")
    print("STAGE 9 PROBABILITY CALIBRATION PASSED")

    print("=" * 72)


if __name__ == "__main__":
    main()
