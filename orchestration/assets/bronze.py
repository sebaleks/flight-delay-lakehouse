"""Bronze assets: thin wrappers over the EXISTING ingestion entry points
(ingestion.bts.run_ingestion, ingestion.bts_external_table.ensure_external_table).
No pipeline logic lives here — idempotency, retries, validation, and the
payload identity checks all remain in ingestion/.

Heavy imports happen inside asset bodies so Definitions load stays
credential-free and lightweight (CI runs `dagster definitions validate`
with placeholder env values and no GCP access).
"""

from __future__ import annotations

from dagster import AssetCheckResult, MaterializeResult, asset, asset_check

# The project's fixed BTS window (CLAUDE.md §8). The monthly schedule re-runs
# idempotently: already-landed months skip in seconds.
BTS_WINDOW_START = (2022, 1)
BTS_WINDOW_END = (2024, 12)


@asset(group_name="bronze", description="BTS monthly CSVs in GCS bronze (idempotent, resumable)")
def bts_bronze() -> MaterializeResult:
    from ingestion.bts import run_ingestion
    from ingestion.util import setup_logging

    setup_logging()  # Dagster steps have no root logging config; INFO would be dropped

    results = run_ingestion(start=BTS_WINDOW_START, end=BTS_WINDOW_END)
    if results["failed"]:
        raise RuntimeError(f"months failed to land: {sorted(results['failed'])}")
    return MaterializeResult(
        metadata={
            "landed": len(results["landed"]),
            "skipped": len(results["skipped"]),
            "failed": len(results["failed"]),
        }
    )


@asset_check(asset=bts_bronze, blocking=True, description="Every window month has CSV+manifest")
def bts_bronze_partitions_complete() -> AssetCheckResult:
    import re

    from google.cloud import storage

    from ingestion.bts import SOURCE_PREFIX, iter_months
    from ingestion.config import require_env

    expected = {f"{y}-{m:02d}" for y, m in iter_months(BTS_WINDOW_START, BTS_WINDOW_END)}
    part = re.compile(rf"{SOURCE_PREFIX}/year=(\d{{4}})/month=(\d{{2}})/(.+)$")
    with_csv: set[str] = set()
    with_manifest: set[str] = set()
    client = storage.Client(project=require_env("GCP_PROJECT_ID"))
    for b in client.list_blobs(require_env("GCS_BUCKET"), prefix=f"{SOURCE_PREFIX}/"):
        m = part.match(b.name)
        if not m:
            continue
        month = f"{m.group(1)}-{m.group(2)}"
        if m.group(3).endswith(".csv"):
            with_csv.add(month)
        elif m.group(3) == "_ingest_manifest.json":
            with_manifest.add(month)
    complete = with_csv & with_manifest
    missing = sorted(expected - complete)
    return AssetCheckResult(
        passed=not missing,
        metadata={
            "months_expected": len(expected),
            "months_complete": len(complete & expected),
            "missing": str(missing),
        },
    )


@asset(
    group_name="bronze",
    deps=[bts_bronze],
    description="Hive-partitioned external table over bronze (no-op when present)",
)
def bts_external_table() -> None:
    from ingestion.bts_external_table import ensure_external_table
    from ingestion.util import setup_logging

    setup_logging()

    ensure_external_table()
