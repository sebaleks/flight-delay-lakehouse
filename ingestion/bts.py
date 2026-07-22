"""Ingest BTS Reporting Carrier On-Time Performance into GCS bronze.

Downloads one zip per month from the BTS PREZIP endpoint (URL pattern verified
working 2026-07-08), validates the contained CSV (required columns + rough row
count), and uploads the CSV **byte-for-byte unmodified** to
``gs://$GCS_BUCKET/bronze/bts_on_time_performance/year=YYYY/month=MM/``.

Idempotent + resumable: a month is skipped only when BOTH its CSV and its
``_ingest_manifest.json`` exist in GCS, so re-running after an interruption
picks up where it left off and heals a partition that lost its manifest.
Uploads use ``if_generation_match=0`` so two concurrent runs cannot
double-write a partition. Each month is identity-checked (zip member name and
first data row must match the requested year/month) so a wrong-month payload
from the server can never land in the wrong partition.

``--force`` re-lands a partition IN PLACE, which deviates from the bronze
immutability rule (CLAUDE.md §3) — reserve it for repairing a corrupted or
incomplete landing, never for routine updates.

Run:
    uv run --extra ingestion python -m ingestion.bts --start 2022-01 --end 2024-12
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage

from ingestion.config import require_env
from ingestion.util import download_with_retries, setup_logging

log = logging.getLogger("ingestion.bts")

URL_TEMPLATE = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
SOURCE_PREFIX = "bronze/bts_on_time_performance"

# Columns silver/gold/ml actually depend on (subset of the ~109 in the file).
# Validation fails a month whose header is missing any of these.
REQUIRED_COLUMNS = frozenset(
    {
        "Year",
        "Quarter",
        "Month",
        "DayofMonth",
        "DayOfWeek",
        "FlightDate",
        "Reporting_Airline",
        "Tail_Number",
        "Flight_Number_Reporting_Airline",
        "Origin",
        "OriginAirportID",
        "Dest",
        "DestAirportID",
        "CRSDepTime",
        "DepTime",
        "DepDelay",
        "DepDelayMinutes",
        "DepDel15",
        "CRSArrTime",
        "ArrTime",
        "ArrDelay",
        "ArrDelayMinutes",
        "ArrDel15",
        "Cancelled",
        "Diverted",
        "CRSElapsedTime",
        "AirTime",
        "Distance",
        "CarrierDelay",
        "WeatherDelay",
        "NASDelay",
        "SecurityDelay",
        "LateAircraftDelay",
    }
)

# US domestic months in 2022-2024 run ~537k-650k rows; a file far outside
# these bounds is truncated or the wrong dataset.
ROW_COUNT_BOUNDS = (300_000, 1_000_000)


class IngestError(RuntimeError):
    pass


def parse_month(value: str) -> tuple[int, int]:
    try:
        year, month = value.split("-")
        parsed = (int(year), int(month))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM, got {value!r}") from exc
    if not 1 <= parsed[1] <= 12:
        raise argparse.ArgumentTypeError(f"month out of range in {value!r}")
    return parsed


def iter_months(start: tuple[int, int], end: tuple[int, int]):
    year, month = start
    while (year, month) <= end:
        yield year, month
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def extract_single_csv(zip_path: Path, dest_dir: Path) -> Path:
    """Extract the one .csv member of the BTS zip (alongside readme.html)."""
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise IngestError(f"corrupt zip member {bad!r} in {zip_path.name}")
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if len(members) != 1:
            raise IngestError(f"expected exactly 1 CSV in {zip_path.name}, found {members!r}")
        zf.extract(members[0], dest_dir)
    return dest_dir / members[0]


def validate_csv(csv_path: Path, year: int, month: int) -> int:
    """Check columns, payload identity, and rough row count; return row count.

    The identity check (first data row's Year/Month must equal the requested
    partition) guards against the server handing back a stale or mixed-up
    archive for the URL — column names and row counts look identical across
    months, so only content can prove we got the month we asked for.
    """
    with open(csv_path, "rb") as f:
        header_line = f.readline().decode("utf-8-sig")
        first_data_line = f.readline().decode("utf-8", errors="replace")
    header = next(csv.reader([header_line]))
    missing = REQUIRED_COLUMNS - set(header)
    if missing:
        raise IngestError(f"{csv_path.name} is missing columns: {sorted(missing)}")

    if not first_data_line.strip():
        raise IngestError(f"{csv_path.name} has a header but no data rows")
    try:
        row = next(csv.reader([first_data_line]))
        idx = {c: i for i, c in enumerate(header)}
        got = (int(row[idx["Year"]]), int(row[idx["Month"]]))
    except (StopIteration, IndexError, ValueError) as exc:
        raise IngestError(f"{csv_path.name}: cannot parse first data row: {exc}") from exc
    if got != (year, month):
        raise IngestError(
            f"{csv_path.name} contains data for {got[0]}-{got[1]:02d}, "
            f"not the requested {year}-{month:02d} — refusing to land it"
        )

    newlines = 0
    last_byte = b"\n"
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            newlines += chunk.count(b"\n")
            last_byte = chunk[-1:]
    lines = newlines + (0 if last_byte == b"\n" else 1)
    rows = lines - 1  # minus header

    low, high = ROW_COUNT_BOUNDS
    if not low <= rows <= high:
        raise IngestError(
            f"{csv_path.name} has {rows:,} data rows, outside sane bounds [{low:,}, {high:,}]"
        )
    return rows


def ingest_month(bucket: storage.Bucket, year: int, month: int, force: bool) -> str:
    prefix = f"{SOURCE_PREFIX}/year={year}/month={month:02d}"
    blob = bucket.blob(f"{prefix}/bts_otp_{year}_{month:02d}.csv")
    manifest_blob = bucket.blob(f"{prefix}/_ingest_manifest.json")
    # skip requires BOTH objects: a partition whose manifest upload was
    # interrupted gets healed on the next run instead of skipped forever
    if not force and blob.exists() and manifest_blob.exists():
        log.info("%d-%02d already in GCS, skipping", year, month)
        return "skipped"

    url = URL_TEMPLATE.format(year=year, month=month)  # month NOT zero-padded
    with tempfile.TemporaryDirectory(prefix=f"bts_{year}_{month:02d}_") as tmp:
        tmp_dir = Path(tmp)
        zip_path = tmp_dir / "otp.zip"
        log.info("%d-%02d downloading %s", year, month, url)
        download_with_retries(url, zip_path)

        csv_path = extract_single_csv(zip_path, tmp_dir)
        # zip member name embeds the true period, month unpadded like the URL
        if not csv_path.name.endswith(f"_{year}_{month}.csv"):
            raise IngestError(
                f"zip member {csv_path.name!r} is not the requested {year}-{month:02d}"
            )
        rows = validate_csv(csv_path, year, month)
        size = csv_path.stat().st_size
        log.info("%d-%02d validated: %s rows, %.1f MB", year, month, f"{rows:,}", size / 1e6)

        manifest = json.dumps(
            {
                "source_url": url,
                "zip_member": csv_path.name,
                "gcs_object": f"gs://{bucket.name}/{blob.name}",
                "data_rows": rows,
                "csv_bytes": size,
                "ingested_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )

        status = "landed"
        try:
            if force:
                # drop the old manifest first so a crash mid-re-land leaves a
                # healable csv-without-manifest state, never a stale manifest
                # describing the previous CSV
                try:
                    manifest_blob.delete()
                except NotFound:
                    pass
                blob.upload_from_filename(str(csv_path), content_type="text/csv", timeout=600)
            else:
                blob.upload_from_filename(
                    str(csv_path),
                    content_type="text/csv",
                    timeout=600,
                    if_generation_match=0,
                )
        except PreconditionFailed:
            # CSV already landed (concurrent run, or an earlier run that died
            # before its manifest) — still (re)write the manifest to heal
            log.info("%d-%02d CSV already present; writing manifest only", year, month)
            status = "skipped"

        manifest_blob.upload_from_string(manifest, content_type="application/json")
    if status == "landed":
        log.info("%d-%02d landed at gs://%s/%s", year, month, bucket.name, blob.name)
    return status


def run_ingestion(
    start: tuple[int, int] = (2022, 1),
    end: tuple[int, int] = (2024, 12),
    force: bool = False,
    workers: int = 3,
) -> dict[str, list[str]]:
    """The ingestion entry point (also wrapped by orchestration): land every
    month in [start, end], idempotently. Returns landed/skipped/failed labels;
    callers decide how to fail."""
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    project = require_env("GCP_PROJECT_ID")
    bucket = storage.Client(project=project).bucket(require_env("GCS_BUCKET"))

    months = list(iter_months(start, end))
    results: dict[str, list[str]] = {"landed": [], "skipped": [], "failed": []}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ingest_month, bucket, y, m, force): (y, m) for y, m in months}
        try:
            for fut in as_completed(futures):
                year, month = futures[fut]
                label = f"{year}-{month:02d}"
                try:
                    results[fut.result()].append(label)
                except Exception:
                    log.exception("%s FAILED", label)
                    results["failed"].append(label)
        except KeyboardInterrupt:
            log.warning("interrupted — cancelling queued months; re-run to resume")
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    log.info(
        "done: %d landed, %d skipped, %d failed%s",
        len(results["landed"]),
        len(results["skipped"]),
        len(results["failed"]),
        f" ({sorted(results['failed'])})" if results["failed"] else "",
    )
    return results


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=parse_month, default=(2022, 1), metavar="YYYY-MM")
    parser.add_argument("--end", type=parse_month, default=(2024, 12), metavar="YYYY-MM")
    parser.add_argument("--force", action="store_true", help="re-land existing months")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start is after --end")
    if args.force:
        log.warning(
            "--force re-lands partitions IN PLACE, deviating from bronze "
            "immutability (CLAUDE.md §3) — use only to repair a bad landing"
        )
    results = run_ingestion(args.start, args.end, args.force, args.workers)
    if results["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
