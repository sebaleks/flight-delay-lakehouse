"""Create the BigQuery external table over the BTS bronze CSVs.

Builds an explicit all-STRING schema from the header of a landed bronze CSV
(bronze stays raw; silver casts types) and creates
``$BQ_BRONZE_DATASET.bts_on_time_performance`` as a hive-partitioned external
table over ``gs://$GCS_BUCKET/bronze/bts_on_time_performance/*.csv``.

Two BTS quirks are handled here:
- every data line ends with a trailing comma, i.e. a 110th unnamed column
  (exposed as ``trailing_empty``);
- the hive partition keys ``year``/``month`` collide case-insensitively with
  the file's own ``Year``/``Month`` columns, so those file columns are exposed
  as ``Year_file``/``Month_file`` (redundant with FlightDate + partition keys).

Idempotent: no-op if the table exists (``--replace`` recreates it). New months
landing under the wildcard appear automatically — no re-run needed.

Run:
    uv run --extra ingestion python -m ingestion.bts_external_table
"""

from __future__ import annotations

import argparse
import csv
import logging

from google.cloud import bigquery, storage

from ingestion.bts import SOURCE_PREFIX
from ingestion.config import require_env
from ingestion.util import setup_logging

log = logging.getLogger("ingestion.bts_external_table")

TABLE_NAME = "bts_on_time_performance"
COLUMN_RENAMES = {"Year": "Year_file", "Month": "Month_file", "": "trailing_empty"}


def read_landed_header(gcs: storage.Client, bucket_name: str) -> list[str]:
    """Read the header row of the first landed bronze CSV (first 64 KiB)."""
    bucket = gcs.bucket(bucket_name)
    blob = next(
        (b for b in gcs.list_blobs(bucket, prefix=f"{SOURCE_PREFIX}/") if b.name.endswith(".csv")),
        None,
    )
    if blob is None:
        raise SystemExit(
            f"no CSV under gs://{bucket_name}/{SOURCE_PREFIX}/ — run ingestion.bts first"
        )
    head = blob.download_as_bytes(start=0, end=65535).decode("utf-8-sig", errors="replace")
    return next(csv.reader([head.splitlines()[0]]))


def ensure_external_table(replace: bool = False) -> None:
    """Idempotent entry point (also wrapped by orchestration): create the
    external table if missing; no-op when present unless replace=True."""
    project = require_env("GCP_PROJECT_ID")
    bucket_name = require_env("GCS_BUCKET")
    dataset = require_env("BQ_BRONZE_DATASET")

    header = read_landed_header(storage.Client(project=project), bucket_name)
    names = [COLUMN_RENAMES.get(c, c) for c in header]
    if len(set(n.lower() for n in names)) != len(names):
        raise SystemExit(f"duplicate column names after rename: {names}")

    bq = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.{TABLE_NAME}"
    if replace:
        bq.delete_table(table_id, not_found_ok=True)
    elif any(t.table_id == TABLE_NAME for t in bq.list_tables(dataset)):
        log.info("%s already exists, nothing to do (use --replace to recreate)", table_id)
        return

    config = bigquery.ExternalConfig("CSV")
    config.source_uris = [f"gs://{bucket_name}/{SOURCE_PREFIX}/*.csv"]
    config.options.skip_leading_rows = 1
    config.schema = [bigquery.SchemaField(n, "STRING") for n in names]
    hive = bigquery.HivePartitioningOptions()
    hive.mode = "AUTO"
    hive.source_uri_prefix = f"gs://{bucket_name}/{SOURCE_PREFIX}"
    config.hive_partitioning = hive

    table = bigquery.Table(table_id)
    table.external_data_configuration = config
    bq.create_table(table)
    log.info("created external table %s (%d file columns + year/month)", table_id, len(names))


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="drop and recreate")
    args = parser.parse_args()
    ensure_external_table(replace=args.replace)


if __name__ == "__main__":
    main()
