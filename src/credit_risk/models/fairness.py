from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
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
AUDIT_GROUPS = ["SEX_GROUP", "AGE_BAND"]


def _safe_auc(y: pd.Series, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if y.nunique() > 1 else float("nan")


def _safe_ap(y: pd.Series, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if y.nunique() > 1 else float("nan")


def subgroup_metrics(
    frame: pd.DataFrame, probabilities: np.ndarray, threshold: float, group_column: str
) -> pd.DataFrame:
    work = frame[[IDENTIFIER, TARGET, group_column]].copy()
    work["probability"] = probabilities
    work["prediction"] = (work["probability"] >= threshold).astype(int)
    rows = []
    for value, group in work.groupby(group_column, dropna=False):
        y = group[TARGET]
        pred = group["prediction"]
        p = group["probability"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "audit_dimension": group_column,
                "group": str(value),
                "n": int(len(group)),
                "default_rate": float(y.mean()),
                "selection_rate": float(pred.mean()),
                "roc_auc": _safe_auc(y, p),
                "pr_auc": _safe_ap(y, p),
                "brier_score": float(brier_score_loss(y, p)),
                "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
                "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
                "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 72)
    print("STAGE 12 - FAIRNESS AND SUBGROUP PERFORMANCE AUDIT")
    print("=" * 72)

    feature_cfg = json.loads(
        (METRICS_DIR / "primary_model_features.json").read_text(encoding="utf-8")
    )
    threshold_cfg = json.loads(
        (METRICS_DIR / "selected_threshold.json").read_text(encoding="utf-8")
    )
    model_features = feature_cfg["model_features"]
    threshold = float(threshold_cfg["selected_threshold"])

    validation = pd.read_parquet(PROCESSED_DATA_DIR / "validation.parquet")
    model = joblib.load(MODELS_DIR / "selected_probability_model.joblib")
    probabilities = model.predict_proba(validation[model_features])[:, 1]

    metrics = pd.concat(
        [subgroup_metrics(validation, probabilities, threshold, column) for column in AUDIT_GROUPS],
        ignore_index=True,
    )
    metrics.to_csv(METRICS_DIR / "fairness_subgroup_metrics.csv", index=False)

    gap_rows = []
    for dimension, group in metrics.groupby("audit_dimension"):
        for metric in [
            "selection_rate",
            "recall",
            "false_positive_rate",
            "false_negative_rate",
            "brier_score",
            "roc_auc",
            "pr_auc",
        ]:
            valid = group[metric].dropna()
            gap_rows.append(
                {
                    "audit_dimension": dimension,
                    "metric": metric,
                    "min_value": float(valid.min()) if not valid.empty else np.nan,
                    "max_value": float(valid.max()) if not valid.empty else np.nan,
                    "max_minus_min_gap": float(valid.max() - valid.min())
                    if not valid.empty
                    else np.nan,
                }
            )
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(METRICS_DIR / "fairness_metric_gaps.csv", index=False)

    recall = metrics.pivot(index="group", columns="audit_dimension", values="recall")
    ax = recall.plot(kind="bar", figsize=(10, 6))
    ax.set_ylabel("Recall")
    ax.set_title("Validation Recall by Audit Subgroup")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "24_fairness_subgroup_recall.png", dpi=180, bbox_inches="tight")
    plt.close()

    lines = [
        "# Fairness and Subgroup Performance Audit",
        "",
        "This audit compares validation-set performance across sex and age groups.",
        "",
        "`SEX`, `AGE`, `SEX_GROUP`, and `AGE_BAND` are excluded from the primary model and retained for auditing.",
        "",
        "This analysis does not certify legal or regulatory fairness.",
        "",
        f"Selected threshold: **{threshold:.3f}**.",
        "",
        "The test set was not accessed in this stage.",
    ]
    (DOCS_DIR / "FAIRNESS_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (METRICS_DIR / "fairness_audit_metadata.json").write_text(
        json.dumps(
            {
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "audit_dimensions": AUDIT_GROUPS,
                "selected_threshold": threshold,
                "test_set_accessed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSUBGROUP PERFORMANCE")
    print("-" * 72)
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nMAX-MIN GAPS")
    print("-" * 72)
    print(gaps.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nSTAGE 12 FAIRNESS AUDIT PASSED")


if __name__ == "__main__":
    main()
