# orchestration/ (Dagster) — added last

Dagster code location for the end-to-end DAG: **ingest → dbt → ML**.

Per CLAUDE.md §6, this is a deliberate **placeholder** until the things it
orchestrates exist and run standalone. `definitions.py` currently loads an empty
`Definitions`.

- Entry point: `orchestration.definitions:defs` (see `[tool.dagster]` in
  `pyproject.toml`).
- Run locally (once populated): `uv run dagster dev`.
- Layout: `assets/` (ingestion, dbt, ml), `resources/` (GCS/BigQuery/dbt CLI),
  `jobs/` (jobs, schedules, sensors).

Install the stack with the `orchestration` extra: `uv sync --extra orchestration`.
