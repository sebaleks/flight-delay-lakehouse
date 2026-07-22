"""Dagster code location "flight_delays" — the deliberately-LAST piece
(CLAUDE.md §6), wrapping the standalone entry points that already run:

    bts_bronze ──> bts_external_table ──> [dbt: seeds -> silver -> gold] ──> ml_training
                                          (wired via source meta in
                                           _bronze__sources.yml)

Nothing is reimplemented: bronze calls ingestion.bts.run_ingestion, the dbt
assets run `dbt build` over the existing project via dagster-dbt, and
ml_training calls ml.train.run_training — the real path, so the leakage
audit, the is_training_row split, the canonical determinism sort, and the
self-contained artifact contract all apply unchanged.

Local development:
    uv run --all-extras dagster dev          # UI at localhost:3000
    uv run --all-extras dagster job execute -j monthly_refresh -m orchestration.definitions
"""

from __future__ import annotations

from pathlib import Path

from dagster import (
    AssetSelection,
    DefaultScheduleStatus,
    Definitions,
    ScheduleDefinition,
    define_asset_job,
)
from dagster_dbt import DbtCliResource
from dotenv import load_dotenv

# GCP identifiers flow through env vars (CLAUDE.md §2); harmless no-op if
# .env is absent (CI uses placeholder env values).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from orchestration.assets.bronze import (  # noqa: E402
    bts_bronze,
    bts_bronze_partitions_complete,
    bts_external_table,
)
from orchestration.assets.dbt_layer import dbt_project, flight_delays_dbt_assets  # noqa: E402
from orchestration.assets.ml import ml_artifact_contract, ml_training  # noqa: E402
from orchestration.assets.reference import airports_seed_csv, holidays_seed_csv  # noqa: E402

# The monthly graph: everything except the manual reference-seed refreshes.
monthly_refresh = define_asset_job(
    name="monthly_refresh",
    selection=AssetSelection.all() - AssetSelection.groups("reference"),
    description="bronze -> dbt (silver/gold + tests) -> ML training",
)

# BTS publishes On-Time Performance ~2-3 months in arrears; a monthly run on
# the 10th picks up anything newly published. With the project's fixed
# 2022-2024 window the bronze step is an idempotent skip, and the run
# refreshes dbt + retrains on the (unchanged) mart — the cadence is the
# production shape, kept honest for the course scope. Enable from the UI.
bts_monthly_schedule = ScheduleDefinition(
    name="monthly_refresh_schedule",
    job=monthly_refresh,
    cron_schedule="0 6 10 * *",
    execution_timezone="America/Los_Angeles",
    default_status=DefaultScheduleStatus.STOPPED,
)

defs = Definitions(
    assets=[
        flight_delays_dbt_assets,
        bts_bronze,
        bts_external_table,
        ml_training,
        airports_seed_csv,
        holidays_seed_csv,
    ],
    asset_checks=[bts_bronze_partitions_complete, ml_artifact_contract],
    jobs=[monthly_refresh],
    schedules=[bts_monthly_schedule],
    resources={"dbt": DbtCliResource(project_dir=dbt_project)},
)
