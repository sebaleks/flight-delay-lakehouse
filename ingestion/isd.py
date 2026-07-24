"""Ingest NOAA ISD Global Hourly station-year CSVs into GCS bronze.

Source: https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{USAF}{WBAN}.csv
— the hourly observations GSOD daily summaries are derived from, so station
ids are the same USAF-WBAN pairs as airport_station_map (inventory verified:
all 1,165 mapped stations have 2022-2024 data).

Bronze layout: gs://$GCS_BUCKET/bronze/isd_hourly/year=YYYY/isd_<station>.csv
+ a manifest JSON per file. NOTE — deliberate deviation from the CLAUDE.md §3
year/month layout: the source's natural file grain is station-YEAR; splitting
rows into month files would rewrite the raw payload, which bronze never does.
Partitioning is by year only, documented here and in the PR.

Same guarantees as the BTS ingester: idempotent skip on csv+manifest, retries
with backoff, and payload identity checks (the first data row's STATION and
DATE year must match the requested file) so a wrong payload can never land in
the wrong partition.

Run:
    uv run --extra ingestion python -m ingestion.isd            # stations from BigQuery map
    uv run --extra ingestion python -m ingestion.isd --stations-file stations.txt
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage

from ingestion.config import require_env
from ingestion.util import download_with_retries, setup_logging

log = logging.getLogger("ingestion.isd")

# Base URL is overridable: NOAA/NESDIS has announced direct ISD CSV access
# via the NCEI website is planned to move to NODD (notice: nesdis.noaa.gov,
# "Service Location Change – Integrated Surface Data Global Hourly"). If a
# fresh bootstrap 404s here, point ISD_BASE_URL at the NODD CSV base — the
# {year}/{usaf}{wban}.csv suffix layout is the same.
BASE_URL = os.environ.get("ISD_BASE_URL", "https://www.ncei.noaa.gov/data/global-hourly/access")
URL_TEMPLATE = BASE_URL.rstrip("/") + "/{year}/{usaf}{wban}.csv"
SOURCE_PREFIX = "bronze/isd_hourly"
YEARS = (2022, 2023, 2024)

REQUIRED_COLUMNS = {"STATION", "DATE", "REPORT_TYPE", "WND", "TMP", "DEW", "VIS"}
# station-years in our set span ~2.7k (sparse arctic) to ~105.5k obs
# (5-minute-cadence ASOS: 288/day * 365 = 105,120 + specials — observed
# ceiling 105,534 in the 2022-2024 backfill); bounds catch truncated or
# wrong-content payloads
ROW_COUNT_BOUNDS = (500, 120_000)


class IsdIngestError(RuntimeError):
    pass


def _validate(csv_path: Path, station: str, year: int) -> int:
    usaf, wban = station.split("-")
    with open(csv_path, "rb") as fh:
        head = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
        reader = csv.reader(head)
        header = next(reader)
        first = next(reader, None)
    missing = REQUIRED_COLUMNS - set(header)
    if missing:
        raise IsdIngestError(f"{csv_path.name}: missing columns {sorted(missing)}")
    if first is None:
        raise IsdIngestError(f"{csv_path.name}: no data rows")
    row = dict(zip(header, first, strict=False))
    # payload identity: STATION is USAF+WBAN concatenated; DATE must be in-year
    if row["STATION"].lstrip("0") != f"{usaf}{wban}".lstrip("0"):
        raise IsdIngestError(f"{csv_path.name}: STATION {row['STATION']!r} != requested {station}")
    if not row["DATE"].startswith(str(year)):
        raise IsdIngestError(f"{csv_path.name}: first DATE {row['DATE']!r} not in {year}")

    newlines = 0
    last_byte = b"\n"
    with open(csv_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            newlines += chunk.count(b"\n")
            last_byte = chunk[-1:]
    # count is exact even without a trailing newline (same fix as bts.py) —
    # the NDJSON converter checks parity against this number
    rows = newlines + (0 if last_byte == b"\n" else 1) - 1
    low, high = ROW_COUNT_BOUNDS
    if not low <= rows <= high:
        raise IsdIngestError(f"{csv_path.name}: {rows:,} rows outside [{low:,}, {high:,}]")
    return rows


def ingest_station_year(bucket: storage.Bucket, station: str, year: int, force: bool) -> str:
    usaf, wban = station.split("-")
    prefix = f"{SOURCE_PREFIX}/year={year}"
    blob = bucket.blob(f"{prefix}/isd_{station}.csv")
    manifest_blob = bucket.blob(f"{prefix}/isd_{station}_manifest.json")
    if not force and blob.exists() and manifest_blob.exists():
        return "skipped"

    url = URL_TEMPLATE.format(year=year, usaf=usaf, wban=wban)
    with tempfile.TemporaryDirectory(prefix=f"isd_{station}_") as tmp:
        csv_path = Path(tmp) / "station.csv"
        download_with_retries(url, csv_path)
        rows = _validate(csv_path, station, year)
        size = csv_path.stat().st_size
        manifest_fields = {
            "source_url": url,
            "gcs_object": f"gs://{bucket.name}/{blob.name}",
            "data_rows": rows,
            "csv_bytes": size,
            "ingested_at": datetime.now(UTC).isoformat(),
        }
        try:
            if force:
                # delete the manifest FIRST (BTS pattern): if this run dies
                # between the CSV overwrite and the manifest write, the next
                # non-force run re-ingests instead of skipping on a stale
                # csv+manifest pair describing the pre-repair payload
                try:
                    manifest_blob.delete()
                except NotFound:
                    pass
                blob.upload_from_filename(str(csv_path), content_type="text/csv", timeout=300)
            else:
                blob.upload_from_filename(
                    str(csv_path), content_type="text/csv", timeout=300, if_generation_match=0
                )
        except PreconditionFailed:
            log.info("%s %d landed by a concurrent run; writing manifest only", station, year)
            # stats below describe THIS run's download; the landed blob came
            # from a concurrent run of the same deterministic source file
            manifest_fields["note"] = "payload landed by concurrent run; stats from this download"
        manifest_blob.upload_from_string(
            json.dumps(manifest_fields, indent=2), content_type="application/json"
        )
    log.info("%s %d landed (%s rows, %.1f MB)", station, year, f"{rows:,}", size / 1e6)
    return "landed"


def mapped_stations() -> list[str]:
    """The relevance set is defined by silver's airport_station_map — a mild,
    deliberate inversion (bronze reading a silver table) kept because the
    alternative is duplicating the nearest-station logic here."""
    from google.cloud import bigquery

    bq = bigquery.Client(project=require_env("GCP_PROJECT_ID"))
    silver = require_env("BQ_SILVER_DATASET")
    rows = bq.query(
        f"select distinct station_id from `{bq.project}.{silver}.airport_station_map`"
    ).result()
    return sorted(r["station_id"] for r in rows)


def run_isd_ingestion(
    stations: list[str], years: tuple[int, ...] = YEARS, force: bool = False, workers: int = 6
) -> dict[str, list[str]]:
    project = require_env("GCP_PROJECT_ID")
    bucket = storage.Client(project=project).bucket(require_env("GCS_BUCKET"))
    tasks = [(s, y) for s in stations for y in years]
    results: dict[str, list[str]] = {"landed": [], "skipped": [], "failed": []}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ingest_station_year, bucket, s, y, force): (s, y) for s, y in tasks}
        for fut in as_completed(futures):
            s, y = futures[fut]
            label = f"{s}/{y}"
            try:
                results[fut.result()].append(label)
            except Exception:
                log.exception("%s FAILED", label)
                results["failed"].append(label)
    log.info(
        "done: %d landed, %d skipped, %d failed",
        len(results["landed"]),
        len(results["skipped"]),
        len(results["failed"]),
    )
    return results


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations-file", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError(f"--workers must be in [1, 16], got {args.workers}")
    if args.force:
        log.warning(
            "--force re-lands station-years IN PLACE, deviating from bronze "
            "immutability (CLAUDE.md §3) — use only to repair a bad landing"
        )
    stations = args.stations_file.read_text().split() if args.stations_file else mapped_stations()
    log.info("ingesting %d stations x %d years", len(stations), len(YEARS))
    results = run_isd_ingestion(stations, force=args.force, workers=args.workers)
    if results["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
