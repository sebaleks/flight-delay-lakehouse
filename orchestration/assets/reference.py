"""Reference-data refresh assets (group "reference"): regenerate the two dbt
seed CSVs from their upstream sources. Deliberately OUTSIDE the monthly job —
airports/holidays are static reference data (dbt/seeds/README.md); refresh
manually from the Dagster UI when the upstream changes, then commit the
deterministic CSV diff.
"""

from __future__ import annotations

from dagster import asset


@asset(group_name="reference", description="Regenerate dbt/seeds/airports.csv from OpenFlights")
def airports_seed_csv() -> None:
    from ingestion.airports import generate_seed
    from ingestion.util import setup_logging

    setup_logging()

    generate_seed()


@asset(group_name="reference", description="Regenerate dbt/seeds/holidays.csv (holidays library)")
def holidays_seed_csv() -> None:
    from ingestion.holidays_cal import generate_seed
    from ingestion.util import setup_logging

    setup_logging()

    generate_seed()
