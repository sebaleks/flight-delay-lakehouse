"""Correctness harness for the dashboard's rate math.

Independently proves that what the app computes in Python (metrics.py over the
loaded view frames) equals ground truth aggregated directly in BigQuery. This is
the guard for the one rule that keeps the dashboard honest: every rolled-up rate
is SUM(numerator)/SUM(denominator), never the mean of a per-row rate.

For each grain it compares, side by side:
  * APP side  — data.load_view(...) -> metrics.aggregate / metrics.totals
  * TRUTH side — a GROUP BY query computing the same SUM/SUM in BigQuery

Run:  uv run --extra dashboard python -m dashboard.verify
Exits non-zero if any rate disagrees beyond a tiny float tolerance.
"""

from __future__ import annotations

import sys

import pandas as pd
from google.cloud import bigquery

from dashboard import data, metrics
from dashboard.config import fq_view, gcp_project

TOL = 1e-9  # rates are pure ratios of exact integer/float sums


def _bq() -> bigquery.Client:
    return bigquery.Client(project=gcp_project())


def _check(name: str, app: float, truth: float, tol: float = TOL) -> dict:
    ok = pd.notna(app) and pd.notna(truth) and abs(app - truth) <= tol
    return {"check": name, "app": app, "bigquery": truth, "abs_diff": abs(app - truth), "ok": ok}


def verify() -> list[dict]:
    bq = _bq()
    rows: list[dict] = []

    # ---- 1. overall totals (monthly_trend) vs direct SQL ----
    overall = metrics.totals(data.monthly_trend())
    t = list(
        bq.query(
            f"""SELECT SUM(n_arr_del15)/SUM(n_with_arr_outcome) AS delay_rate,
                       SUM(n_cancelled)/SUM(n_flights)          AS cancellation_rate,
                       SUM(sum_arr_delay_minutes)/SUM(n_with_arr_outcome) AS avg_arr,
                       SUM(sum_dep_delay_minutes)/SUM(n_with_dep_outcome) AS avg_dep
                FROM {fq_view("dash_monthly_trend")}"""
        )
    )[0]
    rows.append(_check("overall delay_rate", overall["delay_rate"], t.delay_rate))
    rows.append(
        _check("overall cancellation_rate", overall["cancellation_rate"], t.cancellation_rate)
    )
    rows.append(_check("overall avg_arr_delay", overall["avg_arr_delay_minutes"], t.avg_arr, 1e-6))
    rows.append(_check("overall avg_dep_delay", overall["avg_dep_delay_minutes"], t.avg_dep, 1e-6))

    # ---- 2. grouped delay_rate: app metrics.aggregate vs SQL GROUP BY ----
    for grain, cols in {
        "dep_hour": ["dep_hour"],
        "day_of_week": ["day_of_week"],
        "season": ["season"],
        "month": ["year", "month"],
    }.items():
        app = metrics.aggregate(data.delays_by_time(), cols)[cols + ["delay_rate"]]
        sql = bq.query(
            f"""SELECT {", ".join(cols)},
                       SUM(n_arr_del15)/SUM(n_with_arr_outcome) AS delay_rate
                FROM {fq_view("dash_delays_by_time")}
                GROUP BY {", ".join(cols)}"""
        ).to_dataframe()
        merged = app.merge(sql, on=cols, suffixes=("_app", "_bq"))
        max_diff = float((merged["delay_rate_app"] - merged["delay_rate_bq"]).abs().max())
        rows.append(
            {
                "check": f"delay_rate by {grain} ({len(merged)} groups)",
                "app": "—",
                "bigquery": "—",
                "abs_diff": max_diff,
                "ok": max_diff <= TOL and len(merged) == len(app) == len(sql),
            }
        )

    # ---- 3. cross-view consistency: additive time view vs monthly view totals ----
    time_flights = float(data.delays_by_time()["n_flights"].sum())
    month_flights = float(data.monthly_trend()["n_flights"].sum())
    rows.append(_check("total n_flights (time view == monthly view)", time_flights, month_flights))

    return rows


def main() -> None:
    rows = verify()
    report = pd.DataFrame(rows)
    with pd.option_context("display.max_colwidth", 44, "display.width", 120):
        print(report.to_string(index=False))
    n_fail = int((~report["ok"]).sum())
    print(
        f"\n{'✅ ALL CHECKS PASSED' if n_fail == 0 else f'❌ {n_fail} CHECK(S) FAILED'} "
        f"({len(report)} total)"
    )
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
