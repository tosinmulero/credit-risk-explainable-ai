from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

from credit_risk.utils.paths import DOCS_DIR, RAW_DATA_DIR

DATASET_ID = 350
DATASET_NAME = "Default of Credit Card Clients"
DATASET_DOI = "10.24432/C55S3H"
DATASET_URL = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def normalise_dataframe(value: object, fallback_name: str) -> pd.DataFrame:
    if value is None:
        raise ValueError(f"{fallback_name} data was not returned by UCI.")

    if isinstance(value, pd.Series):
        return value.to_frame()

    if isinstance(value, pd.DataFrame):
        return value.copy()

    return pd.DataFrame(value)


def main() -> None:
    print("=" * 72)
    print("UCI CREDIT DEFAULT DATA ACQUISITION")
    print("=" * 72)

    print(f"[INFO] Fetching UCI dataset ID {DATASET_ID}...")

    dataset = fetch_ucirepo(id=DATASET_ID)

    features = normalise_dataframe(
        dataset.data.features,
        "Feature",
    )

    targets = normalise_dataframe(
        dataset.data.targets,
        "Target",
    )

    ids_raw = getattr(dataset.data, "ids", None)

    if ids_raw is None:
        identifiers = pd.DataFrame({"row_id": range(1, len(features) + 1)})
        generated_identifier = True
    else:
        identifiers = normalise_dataframe(
            ids_raw,
            "Identifier",
        )
        generated_identifier = False

    if len(features) != len(targets):
        raise ValueError("Feature and target row counts do not match.")

    if len(identifiers) != len(features):
        raise ValueError("Identifier and feature row counts do not match.")

    identifiers = identifiers.reset_index(drop=True)
    features = features.reset_index(drop=True)
    targets = targets.reset_index(drop=True)

    combined = pd.concat(
        [identifiers, features, targets],
        axis=1,
    )

    csv_path = RAW_DATA_DIR / "credit_default_raw.csv"
    parquet_path = RAW_DATA_DIR / "credit_default_raw.parquet"
    provenance_path = RAW_DATA_DIR / "dataset_provenance.json"
    dictionary_path = DOCS_DIR / "data_dictionary.csv"

    combined.to_csv(csv_path, index=False)
    combined.to_parquet(parquet_path, index=False)

    variables = getattr(dataset, "variables", None)

    if isinstance(variables, pd.DataFrame):
        variables.to_csv(dictionary_path, index=False)
        variable_rows = len(variables)
    else:
        pd.DataFrame().to_csv(dictionary_path, index=False)
        variable_rows = 0

    provenance = {
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "source": "UCI Machine Learning Repository",
        "source_url": DATASET_URL,
        "doi": DATASET_DOI,
        "license": "CC BY 4.0",
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "rows": int(len(combined)),
        "feature_count": int(features.shape[1]),
        "feature_names": [str(column) for column in features.columns],
        "target_count": int(targets.shape[1]),
        "target_names": [str(column) for column in targets.columns],
        "id_columns": [str(column) for column in identifiers.columns],
        "generated_identifier": generated_identifier,
        "variable_metadata_rows": int(variable_rows),
        "raw_csv": str(csv_path.relative_to(csv_path.parents[2])),
        "raw_parquet": str(parquet_path.relative_to(parquet_path.parents[2])),
        "sha256": {
            "credit_default_raw.csv": sha256_file(csv_path),
            "credit_default_raw.parquet": sha256_file(parquet_path),
        },
    }

    provenance_path.write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )

    print(f"[PASS] Rows retrieved:       {len(combined):,}")
    print(f"[PASS] Predictive features:  {features.shape[1]}")
    print(f"[PASS] Target columns:       {targets.shape[1]}")
    print(f"[PASS] Identifier columns:   {identifiers.shape[1]}")
    print(f"[PASS] Raw CSV:              {csv_path}")
    print(f"[PASS] Raw Parquet:          {parquet_path}")
    print(f"[PASS] Data dictionary:      {dictionary_path}")
    print(f"[PASS] Provenance metadata:  {provenance_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
