from __future__ import annotations

import json
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_risk.utils.paths import DOCS_DIR, FIGURES_DIR, METRICS_DIR, PROCESSED_DATA_DIR

EPS = 1e-6


def numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    rp = (
        pd.cut(ref, bins=edges, include_lowest=True)
        .value_counts(normalize=True, sort=False)
        .to_numpy()
    )
    cp = (
        pd.cut(cur, bins=edges, include_lowest=True)
        .value_counts(normalize=True, sort=False)
        .to_numpy()
    )
    rp, cp = np.clip(rp, EPS, None), np.clip(cp, EPS, None)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.astype("string").fillna("<missing>")
    cur = current.astype("string").fillna("<missing>")
    cats = sorted(set(ref.unique()).union(set(cur.unique())))
    rp = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    cp = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    rp, cp = np.clip(rp, EPS, None), np.clip(cp, EPS, None)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def band(psi: float) -> str:
    if np.isnan(psi):
        return "unavailable"
    if psi < 0.10:
        return "low"
    if psi < 0.25:
        return "moderate"
    return "high"


def main() -> None:
    print("=" * 72)
    print("STAGE 17 - DRIFT MONITORING SIMULATION")
    print("=" * 72)
    config = json.loads((METRICS_DIR / "primary_model_features.json").read_text(encoding="utf-8"))
    features = config["model_features"]
    train = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet")
    validation = pd.read_parquet(PROCESSED_DATA_DIR / "validation.parquet")

    rows = []
    for feature in features:
        if pd.api.types.is_numeric_dtype(train[feature]):
            psi, kind = numeric_psi(train[feature], validation[feature]), "numeric"
        else:
            psi, kind = categorical_psi(train[feature], validation[feature]), "categorical"
        rows.append({"feature": feature, "feature_type": kind, "psi": psi, "drift_band": band(psi)})

    summary = pd.DataFrame(rows).sort_values("psi", ascending=False)
    summary.to_csv(METRICS_DIR / "drift_summary.csv", index=False)
    top = summary.dropna(subset=["psi"]).head(15).sort_values("psi")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["psi"])
    ax.axvline(0.10, linestyle="--", label="Moderate")
    ax.axvline(0.25, linestyle=":", label="High")
    ax.set_xlabel("PSI")
    ax.set_title("Pre-deployment Drift Simulation: Train vs Validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "29_drift_monitoring_psi.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    (METRICS_DIR / "drift_metadata.json").write_text(
        json.dumps(
            {
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "reference_dataset": "train",
                "comparison_dataset": "validation",
                "metric": "PSI",
                "high_drift_feature_count": int((summary["drift_band"] == "high").sum()),
                "moderate_drift_feature_count": int((summary["drift_band"] == "moderate").sum()),
                "test_set_used_for_drift": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (DOCS_DIR / "DRIFT_MONITORING.md").write_text(
        "# Drift Monitoring\n\nPre-deployment PSI simulation comparing train and validation distributions.\n\n"
        "- PSI < 0.10: low shift\n- PSI 0.10 to < 0.25: moderate shift\n- PSI >= 0.25: high shift\n\n"
        "These are monitoring heuristics, not universal regulatory limits. The test set is not used.\n",
        encoding="utf-8",
    )
    print(summary.head(20).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nSTAGE 17 DRIFT MONITORING PASSED")


if __name__ == "__main__":
    main()
