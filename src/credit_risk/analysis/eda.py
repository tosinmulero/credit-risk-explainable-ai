from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from credit_risk.data.schema import COLUMN_NAME_MAP
from credit_risk.utils.paths import (
    DOCS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
)

DOCUMENTED_CATEGORY_CODES = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
    "PAY_0": {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    "PAY_2": {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    "PAY_3": {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    "PAY_4": {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    "PAY_5": {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    "PAY_6": {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9},
}


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def load_project_data() -> tuple[
    pd.DataFrame,
    list[str],
    str,
    list[str],
]:
    data_path = RAW_DATA_DIR / "credit_default_raw.parquet"
    provenance_path = RAW_DATA_DIR / "dataset_provenance.json"

    if not data_path.exists():
        raise FileNotFoundError("Raw Parquet dataset not found.")

    if not provenance_path.exists():
        raise FileNotFoundError("Dataset provenance file not found.")

    data = pd.read_parquet(data_path)
    data = data.rename(columns=COLUMN_NAME_MAP)

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    feature_names = [COLUMN_NAME_MAP.get(name, name) for name in provenance["feature_names"]]
    target_names = [COLUMN_NAME_MAP.get(name, name) for name in provenance["target_names"]]
    id_columns = [COLUMN_NAME_MAP.get(name, name) for name in provenance["id_columns"]]

    if len(target_names) != 1:
        raise ValueError("EDA expects exactly one target column.")

    target_name = target_names[0]

    return (
        data,
        feature_names,
        target_name,
        id_columns,
    )


def build_feature_profile(
    features: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for column in features.columns:
        series = features[column]

        rows.append(
            {
                "feature": column,
                "dtype": str(series.dtype),
                "rows": int(len(series)),
                "non_null": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean() * 100),
                "unique_values": int(series.nunique()),
                "minimum": (float(series.min()) if pd.api.types.is_numeric_dtype(series) else None),
                "maximum": (float(series.max()) if pd.api.types.is_numeric_dtype(series) else None),
                "mean": (float(series.mean()) if pd.api.types.is_numeric_dtype(series) else None),
                "median": (
                    float(series.median()) if pd.api.types.is_numeric_dtype(series) else None
                ),
                "standard_deviation": (
                    float(series.std()) if pd.api.types.is_numeric_dtype(series) else None
                ),
            }
        )

    return pd.DataFrame(rows)


def detect_undocumented_codes(
    features: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for column, documented_codes in DOCUMENTED_CATEGORY_CODES.items():
        if column not in features.columns:
            continue

        values = features[column].dropna().astype(int).value_counts().sort_index()

        for value, count in values.items():
            rows.append(
                {
                    "feature": column,
                    "code": int(value),
                    "count": int(count),
                    "percentage": float(count / len(features) * 100),
                    "documented": int(value) in documented_codes,
                }
            )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(["feature", "code"]).reset_index(drop=True)

    return result


def create_class_balance_chart(
    target: pd.Series,
    target_name: str,
) -> dict[str, float]:
    counts = target.value_counts().sort_index()

    non_default = int(counts.get(0, 0))
    default = int(counts.get(1, 0))
    total = int(len(target))

    default_rate = default / total if total else 0.0

    labels = [
        "Non-default",
        "Default",
    ]

    values = [
        non_default,
        default,
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)

    plt.title("Credit Default Class Distribution")
    plt.ylabel("Customers")

    for bar, value in zip(bars, values, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
        )

    save_figure(FIGURES_DIR / "01_class_distribution.png")

    return {
        "target_name": target_name,
        "total": total,
        "non_default": non_default,
        "default": default,
        "default_rate": default_rate,
    }


def create_limit_balance_chart(
    features: pd.DataFrame,
) -> None:
    if "LIMIT_BAL" not in features.columns:
        return

    plt.figure(figsize=(9, 5))

    plt.hist(
        features["LIMIT_BAL"],
        bins=40,
    )

    plt.title("Distribution of Credit Limit")
    plt.xlabel("Credit limit")
    plt.ylabel("Customers")

    save_figure(FIGURES_DIR / "02_credit_limit_distribution.png")


def create_age_chart(
    features: pd.DataFrame,
) -> None:
    if "AGE" not in features.columns:
        return

    plt.figure(figsize=(9, 5))

    plt.hist(
        features["AGE"],
        bins=30,
    )

    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Customers")

    save_figure(FIGURES_DIR / "03_age_distribution.png")


def create_target_correlation_analysis(
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    analysis = features.copy()

    target_temp_name = "__target__"
    analysis[target_temp_name] = target.to_numpy()

    correlations = (
        analysis.corr(numeric_only=True)[target_temp_name]
        .drop(target_temp_name)
        .sort_values(
            key=lambda values: values.abs(),
            ascending=False,
        )
    )

    result = correlations.rename("correlation_with_target").reset_index()

    result.columns = [
        "feature",
        "correlation_with_target",
    ]

    result.to_csv(
        METRICS_DIR / "target_correlations.csv",
        index=False,
    )

    top = result.head(12).copy()
    top = top.sort_values("correlation_with_target")

    plt.figure(figsize=(9, 7))

    plt.barh(
        top["feature"],
        top["correlation_with_target"],
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.title("Features with Strongest Linear Association with Default")
    plt.xlabel("Pearson correlation")

    save_figure(FIGURES_DIR / "04_target_correlations.png")

    return result


def create_repayment_status_analysis(
    data: pd.DataFrame,
    target_name: str,
) -> pd.DataFrame:
    candidate_columns = [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]

    available = [column for column in candidate_columns if column in data.columns]

    if not available:
        return pd.DataFrame()

    results: list[pd.DataFrame] = []

    for column in available:
        grouped = (
            data.groupby(column, dropna=False)[target_name]
            .agg(
                customers="size",
                defaults="sum",
                default_rate="mean",
            )
            .reset_index()
        )

        grouped.insert(
            0,
            "repayment_feature",
            column,
        )

        results.append(grouped)

    combined = pd.concat(
        results,
        ignore_index=True,
    )

    combined.to_csv(
        METRICS_DIR / "repayment_status_default_rates.csv",
        index=False,
    )

    latest = combined[combined["repayment_feature"] == available[0]].copy()

    latest = latest.sort_values(available[0])

    plt.figure(figsize=(10, 5))

    plt.bar(
        latest[available[0]].astype(str),
        latest["default_rate"] * 100,
    )

    plt.title(f"Default Rate by {available[0]} Repayment Status")
    plt.xlabel("Repayment status code")
    plt.ylabel("Default rate (%)")

    save_figure(FIGURES_DIR / "05_repayment_status_default_rate.png")

    return combined


def create_bill_payment_summary(
    features: pd.DataFrame,
) -> pd.DataFrame:
    bill_columns = [column for column in features.columns if column.startswith("BILL_AMT")]

    payment_columns = [column for column in features.columns if column.startswith("PAY_AMT")]

    rows: list[dict[str, object]] = []

    for group_name, columns in (
        ("bill_amount", bill_columns),
        ("payment_amount", payment_columns),
    ):
        for column in columns:
            series = features[column]

            rows.append(
                {
                    "feature_group": group_name,
                    "feature": column,
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "minimum": float(series.min()),
                    "maximum": float(series.max()),
                    "zero_pct": float((series == 0).mean() * 100),
                    "negative_pct": float((series < 0).mean() * 100),
                }
            )

    result = pd.DataFrame(rows)

    result.to_csv(
        METRICS_DIR / "bill_payment_summary.csv",
        index=False,
    )

    if bill_columns:
        bill_means = features[bill_columns].mean()

        plt.figure(figsize=(10, 5))

        plt.plot(
            bill_columns,
            bill_means.values,
            marker="o",
        )

        plt.title("Mean Monthly Bill Amount")
        plt.xlabel("Bill feature")
        plt.ylabel("Mean bill amount")
        plt.xticks(rotation=45)

        save_figure(FIGURES_DIR / "06_mean_bill_amounts.png")

    if payment_columns:
        payment_means = features[payment_columns].mean()

        plt.figure(figsize=(10, 5))

        plt.plot(
            payment_columns,
            payment_means.values,
            marker="o",
        )

        plt.title("Mean Previous Payment Amount")
        plt.xlabel("Payment feature")
        plt.ylabel("Mean payment amount")
        plt.xticks(rotation=45)

        save_figure(FIGURES_DIR / "07_mean_payment_amounts.png")

    return result


def create_category_frequency_tables(
    features: pd.DataFrame,
) -> pd.DataFrame:
    candidate_columns = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
    ]

    rows: list[dict[str, object]] = []

    for column in candidate_columns:
        if column not in features.columns:
            continue

        counts = features[column].value_counts(dropna=False).sort_index()

        for value, count in counts.items():
            rows.append(
                {
                    "feature": column,
                    "value": value,
                    "count": int(count),
                    "percentage": float(count / len(features) * 100),
                }
            )

    result = pd.DataFrame(rows)

    result.to_csv(
        METRICS_DIR / "categorical_frequency_summary.csv",
        index=False,
    )

    return result


def write_data_dictionary_markdown(
    features: pd.DataFrame,
    target_name: str,
) -> None:
    descriptions = {
        "LIMIT_BAL": "Amount of given credit.",
        "SEX": "Sex code supplied by the source dataset.",
        "EDUCATION": "Education category code supplied by the source dataset.",
        "MARRIAGE": "Marital-status category code supplied by the source dataset.",
        "AGE": "Age in years.",
        "PAY_0": "Most recent repayment-status code.",
        "PAY_2": "Repayment-status code for the prior month.",
        "PAY_3": "Repayment-status code.",
        "PAY_4": "Repayment-status code.",
        "PAY_5": "Repayment-status code.",
        "PAY_6": "Repayment-status code.",
    }

    for number in range(1, 7):
        descriptions[f"BILL_AMT{number}"] = "Historical bill-statement amount."

        descriptions[f"PAY_AMT{number}"] = "Historical payment amount."

    lines = [
        "# Data Dictionary",
        "",
        "## Predictive Features",
        "",
        "| Feature | Data type | Unique values | Description |",
        "|---|---|---:|---|",
    ]

    for column in features.columns:
        description = descriptions.get(
            column,
            "Predictive feature supplied by the UCI source dataset.",
        )

        lines.append(
            f"| {column} | {features[column].dtype} | "
            f"{features[column].nunique()} | {description} |"
        )

    lines.extend(
        [
            "",
            "## Target",
            "",
            f"`{target_name}` is the binary credit-default response.",
            "",
            "## Important coding note",
            "",
            "Categorical variables are encoded numerically. "
            "Some observed category codes may fall outside the "
            "documented source definitions. These are treated as "
            "undocumented source values rather than silently corrected.",
            "",
            "Repayment-status codes are ordinal operational codes and "
            "must not be interpreted as continuous monetary quantities.",
        ]
    )

    path = DOCS_DIR / "DATA_DICTIONARY.md"

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_eda_report(
    features: pd.DataFrame,
    target_name: str,
    class_summary: dict[str, float],
    correlations: pd.DataFrame,
    undocumented: pd.DataFrame,
    repayment: pd.DataFrame,
) -> None:
    top_correlations = correlations.head(8)

    lines = [
        "# Exploratory Data Analysis Report",
        "",
        "## Dataset Overview",
        "",
        f"- Observations: **{len(features):,}**",
        f"- Predictive features: **{features.shape[1]}**",
        f"- Target: **{target_name}**",
        (f"- Default observations: **{int(class_summary['default']):,}**"),
        (f"- Non-default observations: **{int(class_summary['non_default']):,}**"),
        (f"- Default rate: **{class_summary['default_rate']:.2%}**"),
        "",
        "## Data Quality",
        "",
        (f"- Missing predictive values: **{int(features.isna().sum().sum()):,}**"),
        (f"- Duplicate predictive profiles: **{int(features.duplicated().sum()):,}**"),
        "",
        "Duplicate predictive profiles are not automatically removed. "
        "Different clients can legitimately share the same model features.",
        "",
        "## Strongest Linear Associations with Default",
        "",
        "| Feature | Correlation |",
        "|---|---:|",
    ]

    for row in top_correlations.itertuples(index=False):
        lines.append(f"| {row.feature} | {row.correlation_with_target:.4f} |")

    lines.extend(
        [
            "",
            "Correlation is descriptive only and does not establish "
            "causality or feature importance in the final model.",
            "",
            "## Encoded Category Review",
            "",
        ]
    )

    if undocumented.empty:
        lines.append("No reviewed category variables contained undocumented codes.")
    else:
        undocumented_only = undocumented[
            undocumented["documented"] == False  # noqa: E712
        ]

        if undocumented_only.empty:
            lines.append("No reviewed category variables contained undocumented codes.")
        else:
            lines.append(
                "The following source values fall outside the documented "
                "category definitions and require explicit preprocessing:"
            )
            lines.append("")
            lines.append("| Feature | Code | Count | Percentage |")
            lines.append("|---|---:|---:|---:|")

            for row in undocumented_only.itertuples(index=False):
                lines.append(
                    f"| {row.feature} | {row.code} | {row.count:,} | {row.percentage:.2f}% |"
                )

    lines.extend(
        [
            "",
            "These values are not silently deleted or relabelled during EDA.",
            "",
            "## Repayment Behaviour",
            "",
        ]
    )

    if repayment.empty:
        lines.append("Repayment-status fields were not detected.")
    else:
        highest = repayment.loc[repayment["default_rate"].idxmax()]

        lines.append(
            "Repayment-status variables show material variation in "
            "observed default rates across status codes."
        )

        lines.append(
            f"The highest observed grouped default rate in this "
            f"summary occurs for `{highest['repayment_feature']}` "
            f"status `{highest.iloc[1]}` at "
            f"**{highest['default_rate']:.2%}**."
        )

    lines.extend(
        [
            "",
            "## Modelling Implications",
            "",
            "1. The target is imbalanced, so plain accuracy will not be "
            "treated as the primary evaluation metric.",
            "2. Average precision / PR-AUC, ROC-AUC, recall, precision, "
            "F1, log loss and Brier score will be considered.",
            "3. Probability calibration will be evaluated because credit "
            "risk decisions depend on estimated probability, not only class.",
            "4. Decision thresholds will be separated from model training "
            "and optimised using explicit business-cost assumptions.",
            "5. Encoded categorical variables require deliberate treatment.",
            "6. Demographic variables will receive subgroup/fairness analysis.",
            "7. Predictive associations will not be presented as causal effects.",
            "",
            "## Leakage Review",
            "",
            "The modelling pipeline will use only variables available in the "
            "source feature matrix and will keep the target isolated throughout "
            "training. Train/validation/test splitting will occur before any "
            "learned preprocessing.",
            "",
            "## Next Stage",
            "",
            "Feature engineering and reproducible train/validation/test construction.",
        ]
    )

    report_path = REPORTS_DIR / "EDA_REPORT.md"

    report_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CREDIT RISK PROFESSIONAL EXPLORATORY DATA ANALYSIS")
    print("=" * 72)

    (
        data,
        feature_names,
        target_name,
        id_columns,
    ) = load_project_data()

    features = data[feature_names].copy()
    target = data[target_name].copy()

    print(f"Rows:                  {len(data):,}")
    print(f"Predictive features:   {len(feature_names)}")
    print(f"Identifier columns:    {len(id_columns)}")
    print(f"Target:                {target_name}")
    print("")

    feature_profile = build_feature_profile(features)

    feature_profile.to_csv(
        METRICS_DIR / "feature_profile.csv",
        index=False,
    )

    print("[PASS] Feature profile created.")

    undocumented = detect_undocumented_codes(features)

    undocumented.to_csv(
        METRICS_DIR / "category_code_review.csv",
        index=False,
    )

    print("[PASS] Encoded-category review completed.")

    class_summary = create_class_balance_chart(
        target,
        target_name,
    )

    print("[PASS] Class-balance analysis completed.")

    create_limit_balance_chart(features)
    create_age_chart(features)

    print("[PASS] Core distributions created.")

    correlations = create_target_correlation_analysis(
        features,
        target,
    )

    print("[PASS] Target-correlation analysis completed.")

    repayment = create_repayment_status_analysis(
        data,
        target_name,
    )

    print("[PASS] Repayment-status analysis completed.")

    create_bill_payment_summary(features)

    print("[PASS] Bill/payment analysis completed.")

    create_category_frequency_tables(features)

    print("[PASS] Category-frequency tables created.")

    write_data_dictionary_markdown(
        features,
        target_name,
    )

    print("[PASS] Professional data dictionary created.")

    write_eda_report(
        features,
        target_name,
        class_summary,
        correlations,
        undocumented,
        repayment,
    )

    print("[PASS] EDA report created.")

    print("")
    print("TARGET SUMMARY")
    print("-" * 72)
    print(f"Non-default: {int(class_summary['non_default']):,}")
    print(f"Default:     {int(class_summary['default']):,}")
    print(f"Default rate: {class_summary['default_rate']:.2%}")

    print("")
    print("TOP TARGET CORRELATIONS")
    print("-" * 72)

    print(
        correlations.head(10).to_string(
            index=False,
        )
    )

    print("")
    print("UNDOCUMENTED CATEGORY CODES")
    print("-" * 72)

    if undocumented.empty:
        print("None detected in reviewed category fields.")
    else:
        undocumented_only = undocumented[
            undocumented["documented"] == False  # noqa: E712
        ]

        if undocumented_only.empty:
            print("None detected in reviewed category fields.")
        else:
            print(
                undocumented_only.to_string(
                    index=False,
                )
            )

    print("")
    print("=" * 72)
    print("STAGE 4 EDA COMPLETE")
    print("=" * 72)
    print("[PASS] Dataset profiled")
    print("[PASS] Class imbalance quantified")
    print("[PASS] Encoded categories reviewed")
    print("[PASS] Credit-limit distribution analysed")
    print("[PASS] Age distribution analysed")
    print("[PASS] Target correlations calculated")
    print("[PASS] Repayment behaviour analysed")
    print("[PASS] Bill/payment behaviour analysed")
    print("[PASS] Professional figures generated")
    print("[PASS] Machine-readable EDA outputs generated")
    print("[PASS] EDA_REPORT.md generated")
    print("[PASS] DATA_DICTIONARY.md generated")
    print("=" * 72)


if __name__ == "__main__":
    main()
