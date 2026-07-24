"""Expose ISD hourly bronze to BigQuery: NDJSON access layer + external table.

WHY AN ACCESS LAYER EXISTS (deliberate, documented deviation): ISD station-year
CSVs are HETEROGENEOUS — only the data groups a station ever reported become
columns, so files range ~30 to ~104 columns with differing positions after the
16 fixed control/mandatory fields (verified on samples). BigQuery CSV external
tables are positional and cannot span that. The fix:

  bronze/isd_hourly/        raw CSVs   — the immutable record (never rewritten)
  bronze/isd_hourly_jsonl/  NDJSON.gz  — derived ACCESS representation for the
                                         external table: a per-row projection to
                                         the fields silver consumes, values
                                         byte-verbatim, plus one injected key
                                         `_station_id` ("USAF-WBAN", from the
                                         file identity — the STATION column can
                                         drop leading zeros)

The JSONL is regenerable from raw at any time and carries no parsing/typing
logic — silver does ALL decoding in SQL (CLAUDE.md §5).

Run (conversion is idempotent per file; table creation is a no-op when present):
    uv run --extra ingestion python -m ingestion.isd_external_table
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.api_core.exceptions import PreconditionFailed
from google.cloud import bigquery, storage

from ingestion.config import require_env
from ingestion.isd import SOURCE_PREFIX, YEARS
from ingestion.util import setup_logging

log = logging.getLogger("ingestion.isd_external_table")

JSONL_PREFIX = f"{SOURCE_PREFIX}_jsonl"
TABLE_NAME = "isd_hourly"

# The projection silver consumes: identity/time/QC, the mandatory packed
# elements, gust (OC1), precip accumulation groups (AA1-AA4, period-coded),
# and present-weather codes (MW manual, AW automated). Values pass verbatim.
KEEP_FIELDS = (
    "STATION",
    "DATE",
    "REPORT_TYPE",
    "QUALITY_CONTROL",
    "WND",
    "CIG",
    "VIS",
    "TMP",
    "DEW",
    "SLP",
    "OC1",
    "AA1",
    "AA2",
    "AA3",
    "AA4",
    "MW1",
    "MW2",
    "MW3",
    "MW4",
    "MW5",
    "MW6",
    "MW7",
    "AW1",
    "AW2",
    "AW3",
    "AW4",
    "AW5",
    "AW6",
)


def convert_station_year(bucket: storage.Bucket, station: str, year: int, force: bool) -> str:
    src = bucket.get_blob(f"{SOURCE_PREFIX}/year={year}/isd_{station}.csv")
    if src is None:
        raise FileNotFoundError(
            f"bronze CSV missing: {SOURCE_PREFIX}/year={year}/isd_{station}.csv"
        )
    dst = bucket.get_blob(f"{JSONL_PREFIX}/year={year}/isd_{station}.jsonl.gz")
    dst_name = f"{JSONL_PREFIX}/year={year}/isd_{station}.jsonl.gz"
    # skip only when the existing JSONL was derived from THIS raw generation
    # AND this projection: a --force repair of the raw CSV changes
    # src.generation, and a KEEP_FIELDS edit changes the projection stamp —
    # either way the stale conversion reconverts automatically instead of
    # silently serving pre-repair data or a mixed-projection lake
    projection = hashlib.sha256(",".join(KEEP_FIELDS).encode()).hexdigest()[:12]
    if (
        not force
        and dst is not None
        and (dst.metadata or {}).get("src_generation") == str(src.generation)
        and (dst.metadata or {}).get("projection") == projection
    ):
        return "skipped"

    raw = src.download_as_bytes()
    buf = io.BytesIO()
    n_out = 0
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        out = io.TextIOWrapper(gz, encoding="utf-8")
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8", errors="replace")))
        for row in reader:
            rec = {"_station_id": station}
            for k in KEEP_FIELDS:
                v = row.get(k)
                if v:  # sparse NDJSON: absent/empty fields are omitted
                    rec[k] = v
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            n_out += 1
        out.flush()

    # row parity against the ingest manifest (written by ingestion.isd from an
    # independent newline count of the validated download)
    manifest_blob = bucket.get_blob(f"{SOURCE_PREFIX}/year={year}/isd_{station}_manifest.json")
    if manifest_blob is not None:
        expected = json.loads(manifest_blob.download_as_bytes()).get("data_rows")
        if expected is not None and expected != n_out:
            raise RuntimeError(f"{station}/{year}: wrote {n_out} rows, manifest says {expected}")
    elif n_out == 0:
        raise RuntimeError(f"{station}/{year}: no manifest and zero rows converted")

    out_blob = bucket.blob(dst_name)
    out_blob.metadata = {"src_generation": str(src.generation), "projection": projection}
    try:
        if dst is None:
            # fresh object: precondition guards the concurrent-writer race
            out_blob.upload_from_string(
                buf.getvalue(), content_type="application/gzip", if_generation_match=0
            )
        else:  # stale or --force reconversion: overwrite deliberately
            out_blob.upload_from_string(buf.getvalue(), content_type="application/gzip")
    except PreconditionFailed:
        log.info("%s %d converted by a concurrent run", station, year)
        return "skipped"
    return "converted"


def run_conversion(force: bool = False, workers: int = 8) -> dict[str, int]:
    project = require_env("GCP_PROJECT_ID")
    gcs = storage.Client(project=project)
    bucket = gcs.bucket(require_env("GCS_BUCKET"))
    tasks = []
    name_re = re.compile(rf"{re.escape(SOURCE_PREFIX)}/year=(\d{{4}})/isd_([0-9A-Z]+-\d+)\.csv$")
    for year in YEARS:
        for blob in gcs.list_blobs(bucket, prefix=f"{SOURCE_PREFIX}/year={year}/"):
            m = name_re.match(blob.name)
            if m:
                tasks.append((m.group(2), year))
            elif blob.name.endswith(".csv"):
                log.warning("skipping unrecognized object %s", blob.name)
    log.info("converting %d station-year files to NDJSON", len(tasks))
    counts = {"converted": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(convert_station_year, bucket, s, y, force): (s, y) for s, y in tasks}
        for done, fut in enumerate(as_completed(futures), 1):
            s, y = futures[fut]
            try:
                counts[fut.result()] += 1
            except Exception:
                log.exception("%s/%d conversion FAILED", s, y)
                counts["failed"] += 1
            if done % 500 == 0:
                log.info("progress: %d/%d (%s)", done, len(tasks), counts)
    log.info("conversion done: %s", counts)
    return counts


def ensure_external_table(replace: bool = False) -> None:
    """Hive-partitioned (year) external table over the NDJSON access layer.
    All fields STRING — silver casts and QC-filters (CLAUDE.md §5)."""
    project = require_env("GCP_PROJECT_ID")
    bucket_name = require_env("GCS_BUCKET")
    dataset = require_env("BQ_BRONZE_DATASET")

    bq = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.{TABLE_NAME}"
    if replace:
        bq.delete_table(table_id, not_found_ok=True)
    elif any(t.table_id == TABLE_NAME for t in bq.list_tables(dataset)):
        log.info("%s already exists, nothing to do (use --replace to recreate)", table_id)
        return

    config = bigquery.ExternalConfig("NEWLINE_DELIMITED_JSON")
    config.source_uris = [f"gs://{bucket_name}/{JSONL_PREFIX}/*.jsonl.gz"]
    config.schema = [bigquery.SchemaField(n, "STRING") for n in ("_station_id", *KEEP_FIELDS)]
    config.ignore_unknown_values = True
    config.compression = "GZIP"  # explicit — never rely on extension sniffing
    hive = bigquery.HivePartitioningOptions()
    hive.mode = "AUTO"
    hive.source_uri_prefix = f"gs://{bucket_name}/{JSONL_PREFIX}"
    config.hive_partitioning = hive

    table = bigquery.Table(table_id)
    table.external_data_configuration = config
    bq.create_table(table)
    log.info("created external table %s (%d fields + year)", table_id, len(KEEP_FIELDS) + 1)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="reconvert existing JSONL files")
    parser.add_argument("--replace", action="store_true", help="drop and recreate the table")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    # each worker holds a whole file + its JSONL in memory (~3-4x file size)
    if not 1 <= args.workers <= 16:
        raise ValueError(f"--workers must be in [1, 16], got {args.workers}")
    counts = run_conversion(force=args.force, workers=args.workers)
    if counts["failed"]:
        raise SystemExit(1)
    ensure_external_table(replace=args.replace)


if __name__ == "__main__":
    main()
