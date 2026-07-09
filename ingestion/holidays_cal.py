"""Generate the US federal holiday calendar seed with adjacency flags.

Emits one row per calendar date from Jan 1 of --start-year through Dec 31 of
--end-year: is_holiday, holiday_name, is_day_before_holiday and
is_day_after_holiday. The holiday set is padded one year on each side so the
adjacency flags are correct at the range edges (e.g. Dec 31 2024 is flagged as
day-before New Year's Day 2025). Deterministic output → stable git diffs.
Written to ``dbt/seeds/holidays.csv``; loaded by ``dbt seed`` and referenced
via ``{{ ref('holidays') }}`` (decision in dbt/seeds/README.md).

Run:
    uv run --extra ingestion python -m ingestion.holidays_cal
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import date, timedelta

import holidays

from ingestion.config import REPO_ROOT
from ingestion.util import setup_logging

log = logging.getLogger("ingestion.holidays_cal")

SEED_PATH = REPO_ROOT / "dbt" / "seeds" / "holidays.csv"


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2024)
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("--start-year is after --end-year")

    us = holidays.country_holidays("US", years=range(args.start_year - 1, args.end_year + 2))

    rows = []
    day = date(args.start_year, 1, 1)
    end = date(args.end_year, 12, 31)
    while day <= end:
        rows.append(
            {
                "date_day": day.isoformat(),
                "is_holiday": int(day in us),
                "holiday_name": us.get(day, ""),
                "is_day_before_holiday": int(day + timedelta(days=1) in us),
                "is_day_after_holiday": int(day - timedelta(days=1) in us),
            }
        )
        day += timedelta(days=1)

    SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SEED_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    n_holidays = sum(r["is_holiday"] for r in rows)
    log.info("wrote %d dates (%d holidays) to %s", len(rows), n_holidays, SEED_PATH)


if __name__ == "__main__":
    main()
