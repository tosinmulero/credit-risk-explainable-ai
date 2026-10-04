from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def load_feature_configuration() -> tuple[list[str], list[str]]:
    path = METRICS_DIR / "primary_model_features.json"

    configuration = json.loads(path.read_text(encoding="utf-8"))

    model_features = configuration["model_features"]
    categorical_features = configuration["categorical_model_features"]

    return model_features, categorical_features


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = PROCESSED_DATA_DIR / "train.parquet"
    validation_path = PROCESSED_DATA_DIR / "validation.parquet"

    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)

    return train, validation


def create_pipeline(
    model_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    numeric_features = [
        feature for feature in model_features if feature not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
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

    classifier = LogisticRegression(
        solver="liblinear",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "threshold": THRESHOLD,
        "observations": int(len(y_true)),
        "default_prevalence": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision_pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
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
        "log_loss": float(log_loss(y_true, probabilities)),
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


def save_confusion_matrix(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> None:
    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Non-default",
            "Default",
        ],
    )

    figure, axis = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=axis,
        values_format=",d",
        colorbar=False,
    )

    axis.set_title("Logistic Regression Validation Confusion Matrix")

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "08_logistic_baseline_confusion_matrix.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_roc_curve(
    y_true: pd.Series,
    probabilities: np.ndarray,
    roc_auc: float,
) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true,
        probabilities,
    )

    figure, axis = plt.subplots(figsize=(7, 6))

    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"Logistic Regression (AUC = {roc_auc:.3f})",
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    axis.set_title("Validation ROC Curve")

    axis.set_xlabel("False positive rate")

    axis.set_ylabel("True positive rate")

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "09_logistic_baseline_roc_curve.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_precision_recall_curve(
    y_true: pd.Series,
    probabilities: np.ndarray,
    average_precision: float,
) -> None:
    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities,
    )

    prevalence = float(y_true.mean())

    figure, axis = plt.subplots(figsize=(7, 6))

    axis.plot(
        recall,
        precision,
        label=(f"Logistic Regression (AP = {average_precision:.3f})"),
    )

    axis.axhline(
        prevalence,
        linestyle="--",
        label=(f"Prevalence baseline ({prevalence:.3f})"),
    )

    axis.set_title("Validation Precision-Recall Curve")

    axis.set_xlabel("Recall")

    axis.set_ylabel("Precision")

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "10_logistic_baseline_precision_recall_curve.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_coefficients(
    pipeline: Pipeline,
) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    transformed_features = preprocessor.get_feature_names_out()

    coefficients = classifier.coef_[0]

    result = pd.DataFrame(
        {
            "feature": transformed_features,
            "coefficient": coefficients,
        }
    )

    result["absolute_coefficient"] = result["coefficient"].abs()

    result["odds_ratio"] = np.exp(result["coefficient"])

    result = result.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    result.to_csv(
        METRICS_DIR / "logistic_baseline_coefficients.csv",
        index=False,
    )

    return result


def save_documentation(
    metrics: dict[str, float | int],
    model_features: list[str],
    categorical_features: list[str],
) -> None:
    lines = [
        "# Logistic Regression Benchmark",
        "",
        "## Purpose",
        "",
        "This model establishes the first supervised-learning benchmark "
        "for the credit-risk project.",
        "",
        "It is intentionally interpretable and will be used as a reference "
        "when comparing more complex tree-based models.",
        "",
        "## Data Usage",
        "",
        "- Training data: 18,000 observations",
        "- Validation data: 6,000 observations",
        "- Test data: not accessed",
        "- Decision threshold: 0.50",
        "",
        "## Preprocessing",
        "",
        "- Numeric features are standardised using statistics fitted only on the training set.",
        "- Categorical variables are one-hot encoded using categories "
        "learned only from the training set.",
        "- Unknown validation categories are ignored safely.",
        "",
        "## Candidate Features",
        "",
        f"- Total candidate features before encoding: **{len(model_features)}**",
        f"- Categorical features: **{len(categorical_features)}**",
        "",
        "## Validation Performance",
        "",
        f"- ROC-AUC: **{metrics['roc_auc']:.4f}**",
        (f"- Average Precision / PR-AUC: **{metrics['average_precision_pr_auc']:.4f}**"),
        f"- Accuracy: **{metrics['accuracy']:.4f}**",
        f"- Precision: **{metrics['precision']:.4f}**",
        f"- Recall: **{metrics['recall']:.4f}**",
        f"- F1: **{metrics['f1']:.4f}**",
        f"- Log Loss: **{metrics['log_loss']:.4f}**",
        f"- Brier Score: **{metrics['brier_score']:.4f}**",
        "",
        "## Interpretation",
        "",
        "Accuracy is not treated as the primary metric because the target is imbalanced.",
        "",
        "ROC-AUC evaluates ranking discrimination across thresholds, while "
        "Average Precision gives greater visibility into performance on the "
        "default class.",
        "",
        "Log loss and Brier score evaluate probability quality and will "
        "be important when probability calibration is assessed later.",
        "",
        "## Governance",
        "",
        "The validation set is used for benchmark assessment. The test set "
        "remains sealed for final model evaluation.",
        "",
        "Coefficient magnitude is descriptive of this fitted model and should "
        "not be interpreted as causal evidence.",
    ]

    (DOCS_DIR / "LOGISTIC_REGRESSION_BASELINE.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 72)
    print("STAGE 6 - LOGISTIC REGRESSION BENCHMARK")
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

    model_features, categorical_features = load_feature_configuration()

    train, validation = load_splits()

    x_train = train[model_features].copy()

    y_train = train[TARGET].copy()

    x_validation = validation[model_features].copy()

    y_validation = validation[TARGET].copy()

    print(f"Training rows:          {len(train):,}")

    print(f"Validation rows:        {len(validation):,}")

    print(f"Candidate features:     {len(model_features)}")

    print(f"Categorical features:   {len(categorical_features)}")

    print(f"Training default rate:  {y_train.mean():.2%}")

    print(f"Validation default rate:{y_validation.mean():.2%}")

    print("")
    print("[INFO] Fitting training-only preprocessing and model...")

    pipeline = create_pipeline(
        model_features,
        categorical_features,
    )

    pipeline.fit(
        x_train,
        y_train,
    )

    probabilities = pipeline.predict_proba(x_validation)[:, 1]

    predictions = (probabilities >= THRESHOLD).astype(int)

    metrics = calculate_metrics(
        y_validation,
        probabilities,
        predictions,
    )

    metadata = {
        "model_name": "logistic_regression_baseline",
        "model_family": "LogisticRegression",
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "random_state": RANDOM_STATE,
        "threshold": THRESHOLD,
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "candidate_features": int(len(model_features)),
        "sklearn_version": sklearn.__version__,
        "test_set_accessed": False,
        "metrics": metrics,
    }

    model_path = MODELS_DIR / "logistic_regression_baseline.joblib"

    joblib.dump(
        pipeline,
        model_path,
    )

    (METRICS_DIR / "logistic_baseline_validation_metrics.json").write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    predictions_frame = pd.DataFrame(
        {
            IDENTIFIER: validation[IDENTIFIER].to_numpy(),
            "actual_default": y_validation.to_numpy(),
            "predicted_probability": probabilities,
            "predicted_class_050": predictions,
        }
    )

    predictions_frame.to_csv(
        METRICS_DIR / "logistic_baseline_validation_predictions.csv",
        index=False,
    )

    report = classification_report(
        y_validation,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).T.to_csv(METRICS_DIR / "logistic_baseline_classification_report.csv")

    coefficients = save_coefficients(pipeline)

    save_confusion_matrix(
        y_validation,
        predictions,
    )

    save_roc_curve(
        y_validation,
        probabilities,
        metrics["roc_auc"],
    )

    save_precision_recall_curve(
        y_validation,
        probabilities,
        metrics["average_precision_pr_auc"],
    )

    save_documentation(
        metrics,
        model_features,
        categorical_features,
    )

    print("")
    print("VALIDATION METRICS")
    print("-" * 72)

    print(f"ROC-AUC:          {metrics['roc_auc']:.4f}")

    print(f"PR-AUC / AP:      {metrics['average_precision_pr_auc']:.4f}")

    print(f"Accuracy:         {metrics['accuracy']:.4f}")

    print(f"Precision:        {metrics['precision']:.4f}")

    print(f"Recall:           {metrics['recall']:.4f}")

    print(f"F1:               {metrics['f1']:.4f}")

    print(f"Log Loss:         {metrics['log_loss']:.4f}")

    print(f"Brier Score:      {metrics['brier_score']:.4f}")

    print("")
    print("CONFUSION MATRIX")
    print("-" * 72)

    print(f"True negatives:   {metrics['true_negatives']:,}")

    print(f"False positives:  {metrics['false_positives']:,}")

    print(f"False negatives:  {metrics['false_negatives']:,}")

    print(f"True positives:   {metrics['true_positives']:,}")

    print("")
    print("TOP ABSOLUTE COEFFICIENTS")
    print("-" * 72)

    print(
        coefficients[
            [
                "feature",
                "coefficient",
                "odds_ratio",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )

    print("")
    print(f"[PASS] Model saved: {model_path}")
    print("[PASS] Validation predictions saved.")
    print("[PASS] Model coefficients saved.")
    print("[PASS] Confusion matrix generated.")
    print("[PASS] ROC curve generated.")
    print("[PASS] Precision-recall curve generated.")
    print("[PASS] Benchmark documentation generated.")
    print("[PASS] Test set remained untouched.")

    print("")
    print("STAGE 6 LOGISTIC REGRESSION BENCHMARK PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
