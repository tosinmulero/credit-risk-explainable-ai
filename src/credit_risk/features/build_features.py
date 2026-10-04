from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.utils.paths import (
    DOCS_DIR,
    INTERIM_DATA_DIR,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

RANDOM_STATE = 42
TARGET = "DEFAULT_NEXT_MONTH"
IDENTIFIER = "CLIENT_ID"

COLUMN_MAP = {
    "ID": IDENTIFIER,
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
    "Y": TARGET,
}

PAY_STATUS = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]

BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]


def load_data() -> pd.DataFrame:
    path = RAW_DATA_DIR / "credit_default_raw.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    data = pd.read_parquet(path)
    data = data.rename(columns=COLUMN_MAP)

    if IDENTIFIER not in data.columns:
        data.insert(
            0,
            IDENTIFIER,
            np.arange(1, len(data) + 1),
        )

    required = {
        IDENTIFIER,
        TARGET,
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        *PAY_STATUS,
        *BILLS,
        *PAYMENTS,
    }

    missing = sorted(required.difference(data.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return data


def add_semantic_categories(
    data: pd.DataFrame,
) -> pd.DataFrame:
    result = data.copy()

    result["SEX_GROUP"] = (
        result["SEX"]
        .map(
            {
                1: "male",
                2: "female",
            }
        )
        .fillna("undocumented")
        .astype("string")
    )

    result["EDUCATION_GROUP"] = (
        result["EDUCATION"]
        .map(
            {
                1: "graduate_school",
                2: "university",
                3: "high_school",
                4: "other_documented",
            }
        )
        .fillna("undocumented")
        .astype("string")
    )

    result["MARRIAGE_GROUP"] = (
        result["MARRIAGE"]
        .map(
            {
                1: "married",
                2: "single",
                3: "other_documented",
            }
        )
        .fillna("undocumented")
        .astype("string")
    )

    result["AGE_BAND"] = pd.cut(
        result["AGE"],
        bins=[
            0,
            24,
            34,
            44,
            54,
            64,
            np.inf,
        ],
        labels=[
            "<=24",
            "25-34",
            "35-44",
            "45-54",
            "55-64",
            "65+",
        ],
        include_lowest=True,
    ).astype("string")

    return result


def add_behavioural_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    result = data.copy()

    delays = result[PAY_STATUS].clip(lower=0)

    result["DELINQUENT_MONTHS_6M"] = delays.gt(0).sum(axis=1)

    result["SEVERE_DELINQUENT_MONTHS_6M"] = delays.ge(2).sum(axis=1)

    result["MAX_DELAY_MONTHS_6M"] = delays.max(axis=1)

    result["MEAN_DELAY_MONTHS_6M"] = delays.mean(axis=1)

    result["RECENT_DELAY_MONTHS"] = delays["PAY_0"]

    result["BILL_MEAN_6M"] = result[BILLS].mean(axis=1)

    result["BILL_STD_6M"] = result[BILLS].std(axis=1, ddof=0)

    result["BILL_MAX_6M"] = result[BILLS].max(axis=1)

    result["BILL_MIN_6M"] = result[BILLS].min(axis=1)

    result["BILL_CHANGE_RECENT_VS_OLDEST"] = result["BILL_AMT1"] - result["BILL_AMT6"]

    result["PAYMENT_MEAN_6M"] = result[PAYMENTS].mean(axis=1)

    result["PAYMENT_SUM_6M"] = result[PAYMENTS].sum(axis=1)

    result["PAYMENT_MAX_6M"] = result[PAYMENTS].max(axis=1)

    result["ZERO_PAYMENT_MONTHS_6M"] = result[PAYMENTS].eq(0).sum(axis=1)

    limit = result["LIMIT_BAL"].replace(0, np.nan)

    result["LATEST_UTILIZATION"] = result["BILL_AMT1"] / limit

    result["MEAN_UTILIZATION_6M"] = result["BILL_MEAN_6M"] / limit

    result["MAX_UTILIZATION_6M"] = result["BILL_MAX_6M"] / limit

    result["PAYMENT_TO_LIMIT_6M"] = result["PAYMENT_SUM_6M"] / (limit * 6)

    result["ZERO_LIMIT_FLAG"] = result["LIMIT_BAL"].eq(0).astype("int8")

    ratio_columns = [
        "LATEST_UTILIZATION",
        "MEAN_UTILIZATION_6M",
        "MAX_UTILIZATION_6M",
        "PAYMENT_TO_LIMIT_6M",
    ]

    result[ratio_columns] = result[ratio_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return result


def split_data(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    train_validation, test = train_test_split(
        data,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data[TARGET],
    )

    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=train_validation[TARGET],
    )

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def validate_splits(
    full_data: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    if len(train) != 18_000:
        raise AssertionError(f"Expected 18,000 training rows, found {len(train)}.")

    if len(validation) != 6_000:
        raise AssertionError(f"Expected 6,000 validation rows, found {len(validation)}.")

    if len(test) != 6_000:
        raise AssertionError(f"Expected 6,000 test rows, found {len(test)}.")

    if len(train) + len(validation) + len(test) != len(full_data):
        raise AssertionError("Split row count does not match full dataset.")

    train_ids = set(train[IDENTIFIER])
    validation_ids = set(validation[IDENTIFIER])
    test_ids = set(test[IDENTIFIER])

    if not train_ids.isdisjoint(validation_ids):
        raise AssertionError("Train/validation overlap detected.")

    if not train_ids.isdisjoint(test_ids):
        raise AssertionError("Train/test overlap detected.")

    if not validation_ids.isdisjoint(test_ids):
        raise AssertionError("Validation/test overlap detected.")


def build_manifest(
    data: pd.DataFrame,
) -> pd.DataFrame:
    engineered = {
        "DELINQUENT_MONTHS_6M",
        "SEVERE_DELINQUENT_MONTHS_6M",
        "MAX_DELAY_MONTHS_6M",
        "MEAN_DELAY_MONTHS_6M",
        "RECENT_DELAY_MONTHS",
        "BILL_MEAN_6M",
        "BILL_STD_6M",
        "BILL_MAX_6M",
        "BILL_MIN_6M",
        "BILL_CHANGE_RECENT_VS_OLDEST",
        "PAYMENT_MEAN_6M",
        "PAYMENT_SUM_6M",
        "PAYMENT_MAX_6M",
        "ZERO_PAYMENT_MONTHS_6M",
        "LATEST_UTILIZATION",
        "MEAN_UTILIZATION_6M",
        "MAX_UTILIZATION_6M",
        "PAYMENT_TO_LIMIT_6M",
        "ZERO_LIMIT_FLAG",
    }

    audit_only = {
        "SEX",
        "AGE",
        "SEX_GROUP",
        "AGE_BAND",
    }

    reference_only = {
        "EDUCATION",
        "MARRIAGE",
    }

    categorical = {
        "EDUCATION_GROUP",
        "MARRIAGE_GROUP",
    }

    rows = []

    for column in data.columns:
        if column == IDENTIFIER:
            role = "identifier"

        elif column == TARGET:
            role = "target"

        elif column in audit_only:
            role = "audit_only"

        elif column in reference_only:
            role = "reference_only"

        elif column in categorical:
            role = "model_categorical"

        elif column in engineered:
            role = "model_engineered_numeric"

        else:
            role = "model_numeric"

        rows.append(
            {
                "feature": column,
                "role": role,
                "dtype": str(data[column].dtype),
                "unique_values": int(data[column].nunique(dropna=False)),
            }
        )

    return pd.DataFrame(rows)


def save_documentation(
    manifest: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    model_features = manifest.loc[
        manifest["role"].str.startswith("model_"),
        "feature",
    ].tolist()

    categorical = manifest.loc[
        manifest["role"].eq("model_categorical"),
        "feature",
    ].tolist()

    payload = {
        "identifier": IDENTIFIER,
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "audit_only_features": [
            "SEX",
            "AGE",
            "SEX_GROUP",
            "AGE_BAND",
        ],
        "categorical_model_features": categorical,
        "model_features": model_features,
    }

    (METRICS_DIR / "primary_model_features.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Feature Engineering and Leakage-Safe Splitting",
        "",
        "## Split Strategy",
        "",
        "- Training: 60%",
        "- Validation: 20%",
        "- Test: 20%",
        "- Stratified by `DEFAULT_NEXT_MONTH`",
        "- Random state: 42",
        "",
        "The test set is sealed from model selection.",
        "",
        "## Leakage Control",
        "",
        "All features created in this stage are deterministic row-level "
        "transformations. No encoder, scaler, resampling method or model "
        "is fitted before the train/validation/test split.",
        "",
        "Learned preprocessing will be fitted on training data only.",
        "",
        "## Responsible Feature Policy",
        "",
        "`SEX` and `AGE` are retained for audit and subgroup analysis "
        "but excluded from the primary modelling feature set.",
        "",
        "Raw education and marital-status codes are preserved for lineage. "
        "Grouped categorical representations are used as model candidates.",
        "",
        "## Engineered Feature Families",
        "",
        "- delinquency frequency and severity",
        "- bill-level summaries",
        "- payment summaries",
        "- utilisation ratios",
        "- payment-to-limit behaviour",
        "- zero-payment frequency",
        "",
        "## Split Summary",
        "",
        "| Split | Rows | Defaults | Default rate |",
        "|---|---:|---:|---:|",
    ]

    for row in summary.itertuples(index=False):
        lines.append(f"| {row.split} | {row.rows:,} | {row.defaults:,} | {row.default_rate:.2%} |")

    (DOCS_DIR / "FEATURE_ENGINEERING.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 72)
    print("STAGE 5 - FEATURE ENGINEERING AND LEAKAGE-SAFE SPLITTING")
    print("=" * 72)

    data = load_data()
    data = add_semantic_categories(data)
    data = add_behavioural_features(data)

    INTERIM_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(
        INTERIM_DATA_DIR / "credit_default_engineered.parquet",
        index=False,
    )

    train, validation, test = split_data(data)

    validate_splits(
        data,
        train,
        validation,
        test,
    )

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    for name, frame in splits.items():
        frame.to_parquet(
            PROCESSED_DATA_DIR / f"{name}.parquet",
            index=False,
        )

        audit_columns = [
            IDENTIFIER,
            TARGET,
            "SEX",
            "SEX_GROUP",
            "AGE",
            "AGE_BAND",
        ]

        frame[audit_columns].to_parquet(
            PROCESSED_DATA_DIR / f"audit_{name}.parquet",
            index=False,
        )

    summary = pd.DataFrame(
        [
            {
                "split": name,
                "rows": int(len(frame)),
                "defaults": int(frame[TARGET].sum()),
                "non_defaults": int(frame[TARGET].eq(0).sum()),
                "default_rate": float(frame[TARGET].mean()),
            }
            for name, frame in splits.items()
        ]
    )

    summary.to_csv(
        METRICS_DIR / "split_summary.csv",
        index=False,
    )

    manifest = build_manifest(data)

    manifest.to_csv(
        METRICS_DIR / "feature_manifest.csv",
        index=False,
    )

    assignments = pd.concat(
        [frame[[IDENTIFIER, TARGET]].assign(split=name) for name, frame in splits.items()],
        ignore_index=True,
    )

    assignments.to_csv(
        METRICS_DIR / "split_assignments.csv",
        index=False,
    )

    save_documentation(
        manifest,
        summary,
    )

    model_count = int(manifest["role"].str.startswith("model_").sum())

    report = {
        "rows": int(len(data)),
        "columns_after_engineering": int(data.shape[1]),
        "candidate_model_features": model_count,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "random_state": RANDOM_STATE,
        "validation_passed": True,
    }

    (METRICS_DIR / "feature_engineering_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("[PASS] Semantic dataset created.")
    print("[PASS] Grouped categories created.")
    print("[PASS] Behavioural features created.")
    print("[PASS] Stratified 60/20/20 split created.")
    print("[PASS] Split overlap checks passed.")
    print("[PASS] Audit-only demographic datasets created.")
    print("[PASS] Feature manifest created.")
    print("[PASS] Documentation created.")

    print("")
    print("SPLIT SUMMARY")
    print("-" * 72)
    print(summary.to_string(index=False))

    print("")
    print(f"Candidate model features: {model_count}")
    print(f"Columns after engineering: {data.shape[1]}")

    print("")
    print("STAGE 5 FEATURE ENGINEERING PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
