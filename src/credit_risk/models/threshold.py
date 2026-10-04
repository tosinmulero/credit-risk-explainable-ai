from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from credit_risk.utils.paths import (
    DOCS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)

TARGET = "DEFAULT_NEXT_MONTH"
IDENTIFIER = "CLIENT_ID"

PRIMARY_FALSE_POSITIVE_COST = 1.0
PRIMARY_FALSE_NEGATIVE_COST = 5.0

THRESHOLDS = np.round(
    np.arange(
        0.01,
        0.991,
        0.005,
    ),
    3,
)


def load_configuration() -> list[str]:
    payload = json.loads((METRICS_DIR / "primary_model_features.json").read_text(encoding="utf-8"))

    return payload["model_features"]


def load_validation() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DATA_DIR / "validation.parquet")


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0

    recall = tp / (tp + fn) if (tp + fn) else 0.0

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    predicted_positive_rate = float(predictions.mean())

    cost = false_positive_cost * fp + false_negative_cost * fn

    cost_per_1000 = cost / len(y_true) * 1000

    return {
        "threshold": float(threshold),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "predicted_positive_rate": (predicted_positive_rate),
        "false_positive_cost": float(false_positive_cost),
        "false_negative_cost": float(false_negative_cost),
        "total_cost_units": float(cost),
        "cost_per_1000": float(cost_per_1000),
    }


def evaluate_grid(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
) -> pd.DataFrame:
    rows = [
        classification_metrics(
            y_true,
            probabilities,
            float(threshold),
            false_positive_cost,
            false_negative_cost,
        )
        for threshold in THRESHOLDS
    ]

    return pd.DataFrame(rows)


def select_threshold(
    grid: pd.DataFrame,
) -> pd.Series:
    ranked = grid.sort_values(
        [
            "total_cost_units",
            "f1",
            "recall",
            "threshold",
        ],
        ascending=[
            True,
            False,
            False,
            False,
        ],
    )

    return ranked.iloc[0]


def save_tradeoff_chart(
    grid: pd.DataFrame,
    selected_threshold: float,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        grid["threshold"],
        grid["precision"],
        label="Precision",
    )

    axis.plot(
        grid["threshold"],
        grid["recall"],
        label="Recall",
    )

    axis.plot(
        grid["threshold"],
        grid["f1"],
        label="F1",
    )

    axis.axvline(
        selected_threshold,
        linestyle="--",
        label=(f"Selected threshold ({selected_threshold:.3f})"),
    )

    axis.axvline(
        0.50,
        linestyle=":",
        label="Default threshold (0.50)",
    )

    axis.set_xlabel("Decision threshold")

    axis.set_ylabel("Metric value")

    axis.set_ylim(
        0,
        1,
    )

    axis.set_title("Validation Threshold Trade-offs")

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "18_threshold_metric_tradeoff.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_cost_curve(
    grid: pd.DataFrame,
    selected_threshold: float,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        grid["threshold"],
        grid["total_cost_units"],
    )

    axis.axvline(
        selected_threshold,
        linestyle="--",
        label=(f"Minimum-cost threshold ({selected_threshold:.3f})"),
    )

    axis.axvline(
        0.50,
        linestyle=":",
        label="Default threshold (0.50)",
    )

    axis.set_xlabel("Decision threshold")

    axis.set_ylabel("Illustrative cost units")

    axis.set_title("Validation Cost Curve — FN:FP Cost Ratio 5:1")

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "19_threshold_cost_curve.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_confusion_comparison(
    default_metrics: dict[str, float | int],
    selected_metrics: dict[str, float | int],
) -> None:
    labels = [
        "TN",
        "FP",
        "FN",
        "TP",
    ]

    default_values = [
        default_metrics["true_negatives"],
        default_metrics["false_positives"],
        default_metrics["false_negatives"],
        default_metrics["true_positives"],
    ]

    selected_values = [
        selected_metrics["true_negatives"],
        selected_metrics["false_positives"],
        selected_metrics["false_negatives"],
        selected_metrics["true_positives"],
    ]

    positions = np.arange(len(labels))

    width = 0.35

    figure, axis = plt.subplots(figsize=(9, 6))

    axis.bar(
        positions - width / 2,
        default_values,
        width,
        label="Threshold 0.50",
    )

    axis.bar(
        positions + width / 2,
        selected_values,
        width,
        label=("Cost-optimised threshold"),
    )

    axis.set_xticks(
        positions,
        labels,
    )

    axis.set_ylabel("Validation observations")

    axis.set_title("Confusion Matrix Component Comparison")

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "20_threshold_confusion_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def cost_sensitivity(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    scenarios = [
        (1.0, 2.0),
        (1.0, 5.0),
        (1.0, 10.0),
    ]

    rows = []

    for fp_cost, fn_cost in scenarios:
        grid = evaluate_grid(
            y_true,
            probabilities,
            false_positive_cost=fp_cost,
            false_negative_cost=fn_cost,
        )

        selected = select_threshold(grid)

        rows.append(
            {
                "false_positive_cost": (fp_cost),
                "false_negative_cost": (fn_cost),
                "fn_to_fp_ratio": (fn_cost / fp_cost),
                "optimal_threshold": float(selected["threshold"]),
                "precision": float(selected["precision"]),
                "recall": float(selected["recall"]),
                "f1": float(selected["f1"]),
                "false_positives": int(selected["false_positives"]),
                "false_negatives": int(selected["false_negatives"]),
                "cost_units": float(selected["total_cost_units"]),
            }
        )

    return pd.DataFrame(rows)


def save_documentation(
    default_metrics: dict[str, float | int],
    selected_metrics: dict[str, float | int],
    sensitivity: pd.DataFrame,
) -> None:
    default_cost = float(default_metrics["total_cost_units"])

    selected_cost = float(selected_metrics["total_cost_units"])

    cost_reduction = default_cost - selected_cost

    cost_reduction_pct = cost_reduction / default_cost if default_cost else 0.0

    lines = [
        "# Decision Threshold Optimisation",
        "",
        "## Objective",
        "",
        "Replace the arbitrary 0.50 classification cutoff with an "
        "explicit, validation-derived operating threshold.",
        "",
        "## Primary Cost Scenario",
        "",
        "- False-positive cost: **1 illustrative cost unit**",
        "- False-negative cost: **5 illustrative cost units**",
        "",
        "These are scenario weights for portfolio modelling only. "
        "They are not observed financial losses and do not represent "
        "a real lender's underwriting policy.",
        "",
        "## Selected Threshold",
        "",
        (f"- Cost-optimised validation threshold: **{selected_metrics['threshold']:.3f}**"),
        "",
        "## Comparison with 0.50",
        "",
        "| Metric | Threshold 0.50 | Optimised threshold |",
        "|---|---:|---:|",
        (
            "| Precision | "
            f"{default_metrics['precision']:.4f} | "
            f"{selected_metrics['precision']:.4f} |"
        ),
        (f"| Recall | {default_metrics['recall']:.4f} | {selected_metrics['recall']:.4f} |"),
        (f"| F1 | {default_metrics['f1']:.4f} | {selected_metrics['f1']:.4f} |"),
        (
            "| False positives | "
            f"{default_metrics['false_positives']} | "
            f"{selected_metrics['false_positives']} |"
        ),
        (
            "| False negatives | "
            f"{default_metrics['false_negatives']} | "
            f"{selected_metrics['false_negatives']} |"
        ),
        (f"| Cost units | {default_cost:.0f} | {selected_cost:.0f} |"),
        "",
        (
            "- Illustrative cost reduction: "
            f"**{cost_reduction:.0f} units "
            f"({cost_reduction_pct:.2%})**"
        ),
        "",
        "## Sensitivity Analysis",
        "",
        "| FN:FP Ratio | Optimal threshold | Precision | Recall | "
        "False positives | False negatives |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for row in sensitivity.itertuples(index=False):
        lines.append(
            f"| {row.fn_to_fp_ratio:.0f}:1 | "
            f"{row.optimal_threshold:.3f} | "
            f"{row.precision:.4f} | "
            f"{row.recall:.4f} | "
            f"{row.false_positives} | "
            f"{row.false_negatives} |"
        )

    lines.extend(
        [
            "",
            "## Governance",
            "",
            "Threshold optimisation uses the validation set only.",
            "",
            "The test set remains sealed and will be used only once "
            "the modelling, calibration and operating-policy choices "
            "have been frozen.",
            "",
            "A production credit decision system would require "
            "institution-specific financial costs, regulatory review, "
            "fair-lending analysis and policy approval.",
        ]
    )

    (DOCS_DIR / "THRESHOLD_OPTIMISATION.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 72)
    print("STAGE 10 - COST-SENSITIVE DECISION THRESHOLD OPTIMISATION")
    print("=" * 72)

    model_features = load_configuration()

    validation = load_validation()

    x_validation = validation[model_features].copy()

    y_validation = validation[TARGET].to_numpy(dtype=int)

    model = joblib.load(MODELS_DIR / "selected_probability_model.joblib")

    probabilities = model.predict_proba(x_validation)[:, 1]

    print(f"Validation observations: {len(validation):,}")

    print(
        "Primary FN:FP cost ratio: "
        f"{PRIMARY_FALSE_NEGATIVE_COST:.0f}:"
        f"{PRIMARY_FALSE_POSITIVE_COST:.0f}"
    )

    grid = evaluate_grid(
        y_validation,
        probabilities,
        false_positive_cost=(PRIMARY_FALSE_POSITIVE_COST),
        false_negative_cost=(PRIMARY_FALSE_NEGATIVE_COST),
    )

    grid.to_csv(
        METRICS_DIR / "threshold_optimisation_grid.csv",
        index=False,
    )

    selected = select_threshold(grid)

    selected_threshold = float(selected["threshold"])

    selected_metrics = classification_metrics(
        y_validation,
        probabilities,
        selected_threshold,
        PRIMARY_FALSE_POSITIVE_COST,
        PRIMARY_FALSE_NEGATIVE_COST,
    )

    default_metrics = classification_metrics(
        y_validation,
        probabilities,
        0.50,
        PRIMARY_FALSE_POSITIVE_COST,
        PRIMARY_FALSE_NEGATIVE_COST,
    )

    sensitivity = cost_sensitivity(
        y_validation,
        probabilities,
    )

    sensitivity.to_csv(
        METRICS_DIR / "threshold_cost_sensitivity.csv",
        index=False,
    )

    predictions = pd.DataFrame(
        {
            IDENTIFIER: validation[IDENTIFIER].to_numpy(),
            "actual_default": (y_validation),
            "predicted_probability": (probabilities),
            "predicted_class_050": (probabilities >= 0.50).astype(int),
            "predicted_class_selected": (probabilities >= selected_threshold).astype(int),
        }
    )

    predictions.to_csv(
        METRICS_DIR / "threshold_validation_predictions.csv",
        index=False,
    )

    default_cost = float(default_metrics["total_cost_units"])

    selected_cost = float(selected_metrics["total_cost_units"])

    result = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "model": ("selected_probability_model"),
        "selection_dataset": ("validation"),
        "false_positive_cost": (PRIMARY_FALSE_POSITIVE_COST),
        "false_negative_cost": (PRIMARY_FALSE_NEGATIVE_COST),
        "selected_threshold": (selected_threshold),
        "threshold_050_metrics": (default_metrics),
        "selected_threshold_metrics": (selected_metrics),
        "illustrative_cost_reduction": (default_cost - selected_cost),
        "illustrative_cost_reduction_pct": (
            (default_cost - selected_cost) / default_cost if default_cost else 0.0
        ),
        "test_set_accessed": False,
    }

    (METRICS_DIR / "selected_threshold.json").write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_tradeoff_chart(
        grid,
        selected_threshold,
    )

    save_cost_curve(
        grid,
        selected_threshold,
    )

    save_confusion_comparison(
        default_metrics,
        selected_metrics,
    )

    save_documentation(
        default_metrics,
        selected_metrics,
        sensitivity,
    )

    print("")
    print("=" * 72)
    print("THRESHOLD 0.50")
    print("=" * 72)

    for key in (
        "precision",
        "recall",
        "specificity",
        "f1",
        "false_positives",
        "false_negatives",
        "true_positives",
        "true_negatives",
        "total_cost_units",
    ):
        print(f"{key}: {default_metrics[key]}")

    print("")
    print("=" * 72)
    print("SELECTED COST-SENSITIVE THRESHOLD")
    print("=" * 72)

    print(f"Selected threshold: {selected_threshold:.3f}")

    for key in (
        "precision",
        "recall",
        "specificity",
        "f1",
        "false_positives",
        "false_negatives",
        "true_positives",
        "true_negatives",
        "total_cost_units",
    ):
        print(f"{key}: {selected_metrics[key]}")

    print("")
    print(f"Illustrative cost reduction: {default_cost - selected_cost:.0f}")

    print(f"Illustrative cost reduction %: {((default_cost - selected_cost) / default_cost):.2%}")

    print("")
    print("=" * 72)
    print("COST-RATIO SENSITIVITY")
    print("=" * 72)

    print(
        sensitivity.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("")
    print("TEST SET ACCESSED: False")

    print("")
    print("STAGE 10 THRESHOLD OPTIMISATION PASSED")

    print("=" * 72)


if __name__ == "__main__":
    main()
