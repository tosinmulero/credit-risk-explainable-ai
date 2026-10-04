from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
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

from credit_risk.utils.paths import (
    DOCS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)

TARGET = "DEFAULT_NEXT_MONTH"
IDENTIFIER = "CLIENT_ID"


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ids = np.digitize(p, edges[1:-1], right=True)
    total = 0.0
    for i in range(bins):
        mask = ids == i
        if np.any(mask):
            total += mask.mean() * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total)


def main() -> None:
    print("=" * 72)
    print("STAGE 13 - FINAL SEALED TEST EVALUATION")
    print("=" * 72)

    output = METRICS_DIR / "final_test_metrics.json"
    if output.exists():
        print("[SAFEGUARD] Final test metrics already exist; test set will not be evaluated again.")
        print(output.read_text(encoding="utf-8"))
        return

    feature_cfg = json.loads(
        (METRICS_DIR / "primary_model_features.json").read_text(encoding="utf-8")
    )
    threshold_cfg = json.loads(
        (METRICS_DIR / "selected_threshold.json").read_text(encoding="utf-8")
    )
    probability_cfg = json.loads(
        (METRICS_DIR / "selected_probability_model.json").read_text(encoding="utf-8")
    )
    features = feature_cfg["model_features"]
    threshold = float(threshold_cfg["selected_threshold"])

    test = pd.read_parquet(PROCESSED_DATA_DIR / "test.parquet")
    y = test[TARGET].astype(int)
    model = joblib.load(MODELS_DIR / "selected_probability_model.joblib")
    p = model.predict_proba(test[features])[:, 1]
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    metrics = {
        "observations": int(len(test)),
        "default_prevalence": float(y.mean()),
        "selected_threshold": threshold,
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision_pr_auc": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "log_loss": float(log_loss(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "expected_calibration_error": ece(y.to_numpy(), p),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    payload = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_policy": "single_final_holdout_evaluation",
        "selected_probability_variant": probability_cfg["selected_probability_variant"],
        "metrics": metrics,
        "test_set_accessed": True,
        "model_selection_changed_after_test": False,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    pd.DataFrame(
        {
            IDENTIFIER: test[IDENTIFIER].to_numpy(),
            "actual_default": y.to_numpy(),
            "predicted_probability": p,
            "predicted_class": pred,
        }
    ).to_csv(METRICS_DIR / "final_test_predictions.csv", index=False)

    matrix = confusion_matrix(y, pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(matrix, display_labels=["Non-default", "Default"])
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, values_format=",d", colorbar=False)
    ax.set_title("Final Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "25_final_test_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("Final Test ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "26_final_test_roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"AP = {metrics['average_precision_pr_auc']:.3f}")
    ax.axhline(float(y.mean()), linestyle="--")
    ax.set_title("Final Test Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "27_final_test_pr_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(predicted, observed, marker="o")
    ax.set_title("Final Test Reliability Diagram")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "28_final_test_reliability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    (DOCS_DIR / "FINAL_TEST_EVALUATION.md").write_text(
        "# Final Sealed Test Evaluation\n\n"
        f"ROC-AUC: **{metrics['roc_auc']:.4f}**  \n"
        f"PR-AUC: **{metrics['average_precision_pr_auc']:.4f}**  \n"
        f"Precision: **{metrics['precision']:.4f}**  \n"
        f"Recall: **{metrics['recall']:.4f}**  \n"
        f"F1: **{metrics['f1']:.4f}**  \n"
        f"Log loss: **{metrics['log_loss']:.4f}**  \n"
        f"Brier score: **{metrics['brier_score']:.4f}**  \n"
        f"ECE: **{metrics['expected_calibration_error']:.4f}**  \n"
        f"Frozen threshold: **{threshold:.3f}**\n\n"
        "No model or threshold selection is changed after this evaluation.\n",
        encoding="utf-8",
    )

    (METRICS_DIR / "final_model_manifest.json").write_text(
        json.dumps(
            {
                "model_artifact": "models/selected_probability_model.joblib",
                "threshold_artifact": "reports/metrics/selected_threshold.json",
                "feature_config": "reports/metrics/primary_model_features.json",
                "final_test_metrics": "reports/metrics/final_test_metrics.json",
                "selected_threshold": threshold,
                "selected_probability_variant": probability_cfg["selected_probability_variant"],
                "test_set_accessed": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nFINAL TEST METRICS")
    print("-" * 72)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("\nSTAGE 13 FINAL TEST EVALUATION PASSED")


if __name__ == "__main__":
    main()
