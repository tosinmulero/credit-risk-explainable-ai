from __future__ import annotations

import json

import numpy as np
import pandas as pd

from credit_risk.utils.paths import METRICS_DIR, RAW_DATA_DIR

EXPECTED_ROWS = 30_000
EXPECTED_FEATURES = 23


def main() -> None:
    print("=" * 72)
    print("RAW CREDIT DEFAULT DATA VALIDATION")
    print("=" * 72)

    raw_path = RAW_DATA_DIR / "credit_default_raw.parquet"
    provenance_path = RAW_DATA_DIR / "dataset_provenance.json"

    if not raw_path.exists():
        raise FileNotFoundError("Raw Parquet dataset was not found. Run download_data.py first.")

    if not provenance_path.exists():
        raise FileNotFoundError("Dataset provenance file was not found.")

    data = pd.read_parquet(raw_path)

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    feature_names = provenance["feature_names"]
    target_names = provenance["target_names"]
    id_columns = provenance["id_columns"]

    if len(target_names) != 1:
        raise ValueError(f"Expected one target column, found {len(target_names)}.")

    target_name = target_names[0]

    features = data[feature_names].copy()
    target = data[target_name].copy()

    target_values = sorted(int(value) for value in target.dropna().unique().tolist())

    class_counts_raw = target.value_counts(dropna=False).sort_index().to_dict()

    class_counts = {str(key): int(value) for key, value in class_counts_raw.items()}

    default_count = int((target == 1).sum())
    non_default_count = int((target == 0).sum())

    default_rate = default_count / len(target) if len(target) else 0.0

    missing_feature_values = int(features.isna().sum().sum())

    missing_target_values = int(target.isna().sum())

    duplicate_feature_rows = int(features.duplicated().sum())

    duplicate_full_rows = int(data.duplicated().sum())

    all_features_numeric = all(
        pd.api.types.is_numeric_dtype(features[column]) for column in features.columns
    )

    numeric_matrix = features.to_numpy(dtype=float)

    all_numeric_values_finite = bool(np.isfinite(numeric_matrix).all())

    if id_columns:
        duplicate_identifier_rows = int(data[id_columns].duplicated().sum())
    else:
        duplicate_identifier_rows = 0

    checks = {
        "expected_row_count": len(data) == EXPECTED_ROWS,
        "expected_feature_count": len(feature_names) == EXPECTED_FEATURES,
        "single_target_column": len(target_names) == 1,
        "binary_target_0_1": target_values == [0, 1],
        "no_missing_feature_values": missing_feature_values == 0,
        "no_missing_target_values": missing_target_values == 0,
        "all_features_numeric": all_features_numeric,
        "all_numeric_values_finite": all_numeric_values_finite,
        "identifier_values_unique": duplicate_identifier_rows == 0,
    }

    validation_passed = all(checks.values())

    imbalance_ratio = non_default_count / default_count if default_count else None

    report = {
        "validation_passed": validation_passed,
        "dataset": {
            "rows": int(len(data)),
            "columns": int(data.shape[1]),
            "feature_count": int(len(feature_names)),
            "target_name": target_name,
            "target_values": target_values,
        },
        "data_quality": {
            "missing_feature_values": missing_feature_values,
            "missing_target_values": missing_target_values,
            "duplicate_feature_rows": duplicate_feature_rows,
            "duplicate_full_rows": duplicate_full_rows,
            "duplicate_identifier_rows": duplicate_identifier_rows,
        },
        "target_distribution": {
            "class_counts": class_counts,
            "non_default_count": non_default_count,
            "default_count": default_count,
            "default_rate": round(default_rate, 6),
            "non_default_to_default_ratio": (
                round(imbalance_ratio, 4) if imbalance_ratio is not None else None
            ),
        },
        "checks": checks,
    }

    report_path = METRICS_DIR / "raw_data_validation.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "rows",
                "feature_count",
                "missing_feature_values",
                "missing_target_values",
                "duplicate_feature_rows",
                "duplicate_identifier_rows",
                "non_default_count",
                "default_count",
                "default_rate",
                "non_default_to_default_ratio",
            ],
            "value": [
                len(data),
                len(feature_names),
                missing_feature_values,
                missing_target_values,
                duplicate_feature_rows,
                duplicate_identifier_rows,
                non_default_count,
                default_count,
                default_rate,
                imbalance_ratio,
            ],
        }
    )

    summary_path = METRICS_DIR / "raw_data_summary.csv"

    summary.to_csv(summary_path, index=False)

    numeric_summary_path = METRICS_DIR / "raw_numeric_summary.csv"

    features.describe().T.to_csv(numeric_summary_path)

    print(f"Rows:                       {len(data):,}")
    print(f"Predictive features:        {len(feature_names)}")
    print(f"Target:                     {target_name}")
    print(f"Target values:              {target_values}")
    print(f"Missing feature values:     {missing_feature_values:,}")
    print(f"Missing target values:      {missing_target_values:,}")
    print(f"Duplicate feature profiles: {duplicate_feature_rows:,}")
    print(f"Duplicate identifiers:      {duplicate_identifier_rows:,}")
    print(f"Non-default observations:   {non_default_count:,}")
    print(f"Default observations:       {default_count:,}")
    print(f"Default rate:               {default_rate:.2%}")
    print("")

    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {check_name}")

    print("")
    print(f"[PASS] Validation JSON: {report_path}")
    print(f"[PASS] Summary CSV:     {summary_path}")
    print(f"[PASS] Numeric summary: {numeric_summary_path}")
    print("=" * 72)

    if not validation_passed:
        raise SystemExit("ERROR: One or more critical raw-data checks failed.")

    print("STAGE 3 DATA VALIDATION PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
