"""Dagster entry point (``[tool.dagster] module_name = "orchestration.definitions"``).

Orchestration is intentionally added LAST (see CLAUDE.md §6). Until ingestion,
the dbt models, and the ML pipeline exist and run standalone, this is an empty
placeholder that simply loads. Once they do, wire here:

  - ingestion assets (extract sources -> bronze CSV in GCS)
  - the dbt project as assets via dagster-dbt (``@dbt_assets`` over dbt/)
  - ML assets (feature build, time-split, train/eval both models)
  - resources: GCS / BigQuery clients (ADC), a DbtCliResource pointing at ./dbt
  - schedules/sensors for the end-to-end DAG

Run locally once populated:  ``uv run dagster dev``
"""

from dagster import Definitions

# Empty on purpose — assets/resources/jobs/schedules are added as the pipelines
# they orchestrate come online.
defs = Definitions(
    assets=[],
    resources={},
    jobs=[],
    schedules=[],
)
