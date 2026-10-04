from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from credit_risk.utils.paths import (
    DOCS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)

TARGET = "DEFAULT_NEXT_MONTH"
IDENTIFIER = "CLIENT_ID"
TOP_FEATURES = 20


def _config() -> tuple[list[str], float]:
    features = json.loads((METRICS_DIR / "primary_model_features.json").read_text(encoding="utf-8"))
    threshold = json.loads((METRICS_DIR / "selected_threshold.json").read_text(encoding="utf-8"))
    return features["model_features"], float(threshold["selected_threshold"])


def _semantic(name: str) -> str:
    if name.startswith("EDUCATION_GROUP_"):
        return "EDUCATION_GROUP"
    if name.startswith("MARRIAGE_GROUP_"):
        return "MARRIAGE_GROUP"
    return name


def main() -> None:
    print("=" * 72)
    print("STAGE 11 - SHAP EXPLAINABLE AI")
    print("=" * 72)

    model_features, threshold = _config()
    validation = pd.read_parquet(PROCESSED_DATA_DIR / "validation.parquet")
    x = validation[model_features].copy()

    pipeline = joblib.load(MODELS_DIR / "xgboost_tuned.joblib")
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    transformed = preprocessor.transform(x)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=float)
    names = list(preprocessor.get_feature_names_out())
    probabilities = pipeline.predict_proba(x)[:, 1]

    print(f"Validation observations: {len(validation):,}")
    print(f"Transformed features:    {len(names)}")
    print(f"Selected threshold:      {threshold:.3f}")

    explainer = shap.TreeExplainer(model)
    explanation = explainer(transformed)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, -1]
    base_values = np.asarray(explanation.base_values)
    if base_values.ndim == 2:
        base_values = base_values[:, -1]
    if base_values.ndim == 0:
        base_values = np.repeat(float(base_values), len(validation))

    global_imp = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    global_imp.to_csv(METRICS_DIR / "shap_global_feature_importance.csv", index=False)

    semantic = global_imp.copy()
    semantic["semantic_feature"] = semantic["feature"].map(_semantic)
    semantic = (
        semantic.groupby("semantic_feature", as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"), mean_shap=("mean_shap", "sum"))
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    semantic.to_csv(METRICS_DIR / "shap_semantic_feature_importance.csv", index=False)

    top = semantic.head(TOP_FEATURES).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["semantic_feature"], top["mean_abs_shap"])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Global Credit-Risk Feature Importance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "21_shap_global_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    shap_exp = shap.Explanation(
        values=values, base_values=base_values, data=transformed, feature_names=names
    )
    shap.plots.beeswarm(shap_exp, max_display=TOP_FEATURES, show=False)
    plt.title("SHAP Distribution of Validation Risk Drivers")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "22_shap_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()

    cases = pd.DataFrame(
        {
            "row_index": np.arange(len(validation)),
            IDENTIFIER: validation[IDENTIFIER].to_numpy(),
            "actual_default": validation[TARGET].to_numpy(),
            "predicted_probability": probabilities,
        }
    )
    median_probability = float(np.median(probabilities))
    selected = (
        pd.concat(
            [
                cases.nlargest(2, "predicted_probability"),
                cases.assign(distance=np.abs(cases["predicted_probability"] - median_probability))
                .nsmallest(2, "distance")
                .drop(columns="distance"),
                cases.nsmallest(2, "predicted_probability"),
            ],
            ignore_index=True,
        )
        .drop_duplicates("row_index")
        .head(6)
    )
    selected.to_csv(METRICS_DIR / "shap_selected_cases.csv", index=False)

    local_rows = []
    for case_number, case in enumerate(selected.itertuples(index=False), start=1):
        idx = int(case.row_index)
        local = values[idx]
        ranking = np.argsort(np.abs(local))[::-1][:10]
        for rank, j in enumerate(ranking, start=1):
            local_rows.append(
                {
                    "case_number": case_number,
                    IDENTIFIER: getattr(case, IDENTIFIER),
                    "actual_default": int(case.actual_default),
                    "predicted_probability": float(case.predicted_probability),
                    "selected_threshold": threshold,
                    "predicted_risk_flag": int(case.predicted_probability >= threshold),
                    "rank": rank,
                    "feature": names[j],
                    "feature_value": float(transformed[idx, j]),
                    "shap_value": float(local[j]),
                    "direction": "increases risk" if local[j] > 0 else "decreases risk",
                }
            )
        local_exp = shap.Explanation(
            values=local, base_values=base_values[idx], data=transformed[idx], feature_names=names
        )
        shap.plots.waterfall(local_exp, max_display=12, show=False)
        plt.title(f"Local Risk Explanation - Case {case_number}")
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"23_shap_local_case_{case_number:02d}.png", dpi=180, bbox_inches="tight"
        )
        plt.close()

    pd.DataFrame(local_rows).to_csv(METRICS_DIR / "shap_local_explanations.csv", index=False)

    lines = [
        "# SHAP Explainability",
        "",
        "TreeSHAP is used to explain the tuned XGBoost model.",
        "",
        "Positive SHAP values push predictions toward higher estimated default risk; negative values push them lower.",
        "",
        "SHAP explains model behaviour, not causality.",
        "",
        "## Top global features",
        "",
        "| Rank | Feature | Mean absolute SHAP |",
        "|---:|---|---:|",
    ]
    for rank, row in enumerate(semantic.head(15).itertuples(index=False), start=1):
        lines.append(f"| {rank} | {row.semantic_feature} | {row.mean_abs_shap:.6f} |")
    lines += [
        "",
        f"Selected threshold: **{threshold:.3f}**.",
        "",
        "The test set was not accessed in this stage.",
    ]
    (DOCS_DIR / "SHAP_EXPLAINABILITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    metadata = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "model": "Tuned XGBoost",
        "method": "TreeSHAP",
        "validation_rows_explained": int(len(validation)),
        "transformed_feature_count": int(len(names)),
        "semantic_feature_count": int(len(semantic)),
        "selected_threshold": threshold,
        "test_set_accessed": False,
    }
    (METRICS_DIR / "shap_explainability_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nTOP GLOBAL SHAP FEATURES")
    print("-" * 72)
    print(semantic.head(15).to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\nSTAGE 11 SHAP EXPLAINABILITY PASSED")


if __name__ == "__main__":
    main()
