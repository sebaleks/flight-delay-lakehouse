"""Column contracts for the gold ``dash_*`` views, and synthetic frames built
from them.

Why this exists: ``data.load_view`` does ``SELECT *``, so a page's dataframe has
exactly the view's columns and nothing else — and there are two naming
conventions in play.

  ENTITY-GRAIN views (``dash_airport_reliability``, ``dash_carrier_reliability``,
  ``dash_route_drilldown``) carry ``n_flight_legs`` and PRE-DIVIDED rates
  (``arr_del15_rate`` ...).

  ADDITIVE views (``dash_delays_by_time``, ``dash_monthly_trend``,
  ``dash_route_carrier``, ``dash_time_slice``, ``dash_airport_hour_baseline``)
  carry ``n_flights`` plus raw numerators, and NO rate columns — rates are
  computed SUM/SUM by ``dashboard.metrics``.

Reaching for the wrong one raises KeyError at render time, and
``dashboard/app.py`` has no per-page error boundary, so the whole app shows a
traceback. That happened on the route drill-down (``sort_values("n_flights")``
on the route-grain frame) and reached production, because every dashboard test
was a pure-function test and nothing rendered a page.

The contract is enforced from both sides:

  * ``test_pages_render.py`` builds frames from ``SCHEMAS`` and renders every
    page — no credentials, so it runs in CI on every push.
  * ``test_verify.py`` checks ``SCHEMAS`` still matches BigQuery, so the
    fixtures cannot quietly drift away from the warehouse and keep passing.

Regenerate after a dbt schema change with the INFORMATION_SCHEMA query in
``test_verify.test_schemas_match_bigquery``'s docstring.
"""

from __future__ import annotations

import pandas as pd

# view name -> {column: BigQuery type}, in ordinal position.
SCHEMAS: dict[str, dict[str, str]] = {
    "dash_airport_reliability": {
        "airport_key": "STRING",
        "airport_name": "STRING",
        "city": "STRING",
        "tz": "STRING",
        "n_flight_legs": "INT64",
        "n_arr_del15": "INT64",
        "arr_del15_rate": "FLOAT64",
        "on_time_rate": "FLOAT64",
        "avg_arr_delay_minutes": "FLOAT64",
        "p90_arr_delay_minutes": "FLOAT64",
        "n_cancelled": "INT64",
        "cancellation_rate": "FLOAT64",
        "n_diverted": "INT64",
        "diversion_rate": "FLOAT64",
        "hist_arr_del15_rate": "FLOAT64",
        "hist_n_flights": "INT64",
    },
    "dash_carrier_reliability": {
        "carrier_key": "STRING",
        "carrier_name": "STRING",
        "is_regional": "BOOL",
        "dot_id": "INT64",
        "n_flight_legs": "INT64",
        "n_arr_del15": "INT64",
        "arr_del15_rate": "FLOAT64",
        "on_time_rate": "FLOAT64",
        "avg_arr_delay_minutes": "FLOAT64",
        "p90_arr_delay_minutes": "FLOAT64",
        "n_cancelled": "INT64",
        "cancellation_rate": "FLOAT64",
        "n_diverted": "INT64",
        "diversion_rate": "FLOAT64",
        "hist_arr_del15_rate": "FLOAT64",
        "hist_n_flights": "INT64",
    },
    "dash_route_drilldown": {
        "route": "STRING",
        "origin_airport_key": "STRING",
        "origin_airport_name": "STRING",
        "origin_city": "STRING",
        "dest_airport_key": "STRING",
        "dest_airport_name": "STRING",
        "dest_city": "STRING",
        "n_flight_legs": "INT64",
        "n_arr_del15": "INT64",
        "arr_del15_rate": "FLOAT64",
        "on_time_rate": "FLOAT64",
        "avg_arr_delay_minutes": "FLOAT64",
        "p90_arr_delay_minutes": "FLOAT64",
        "n_cancelled": "INT64",
        "cancellation_rate": "FLOAT64",
        "n_diverted": "INT64",
        "diversion_rate": "FLOAT64",
        "hist_arr_del15_rate": "FLOAT64",
    },
    "dash_route_carrier": {
        "route": "STRING",
        "origin_airport_key": "STRING",
        "dest_airport_key": "STRING",
        "carrier_key": "STRING",
        "carrier_name": "STRING",
        "is_regional": "BOOL",
        "dot_id": "INT64",
        "n_flights": "INT64",
        "n_with_arr_outcome": "INT64",
        "n_with_dep_outcome": "INT64",
        "n_arr_del15": "INT64",
        "n_cancelled": "INT64",
        "n_diverted": "INT64",
        "sum_arr_delay_minutes": "FLOAT64",
        "sum_dep_delay_minutes": "FLOAT64",
    },
    "dash_delays_by_time": {
        "year": "INT64",
        "month": "INT64",
        "month_name": "STRING",
        "season": "STRING",
        "season_order": "INT64",
        "day_of_week": "INT64",
        "day_name": "STRING",
        "dep_hour": "INT64",
        "n_flights": "INT64",
        "n_with_arr_outcome": "INT64",
        "n_with_dep_outcome": "INT64",
        "n_arr_del15": "INT64",
        "n_cancelled": "INT64",
        "n_diverted": "INT64",
        "sum_arr_delay_minutes": "FLOAT64",
        "sum_dep_delay_minutes": "FLOAT64",
    },
    "dash_time_slice": {
        "origin_airport_key": "STRING",
        "origin_airport_name": "STRING",
        "carrier_key": "STRING",
        "year": "INT64",
        "month": "INT64",
        "month_name": "STRING",
        "day_of_week": "INT64",
        "day_name": "STRING",
        "dep_hour": "INT64",
        "n_flights": "INT64",
        "n_with_arr_outcome": "INT64",
        "n_with_dep_outcome": "INT64",
        "n_arr_del15": "INT64",
        "n_cancelled": "INT64",
        "n_diverted": "INT64",
        "sum_arr_delay_minutes": "FLOAT64",
        "sum_dep_delay_minutes": "FLOAT64",
    },
    "dash_monthly_trend": {
        "month_start": "DATE",
        "year": "INT64",
        "month": "INT64",
        "month_name": "STRING",
        "n_flights": "INT64",
        "n_with_arr_outcome": "INT64",
        "n_with_dep_outcome": "INT64",
        "n_arr_del15": "INT64",
        "n_cancelled": "INT64",
        "n_diverted": "INT64",
        "sum_arr_delay_minutes": "FLOAT64",
        "sum_dep_delay_minutes": "FLOAT64",
    },
    "dash_airport_hour_baseline": {
        "airport_key": "STRING",
        "airport_name": "STRING",
        "city": "STRING",
        "tz": "STRING",
        "day_of_week": "INT64",
        "day_name": "STRING",
        "dep_hour": "INT64",
        "n_flights": "INT64",
        "n_with_arr_outcome": "INT64",
        "n_with_dep_outcome": "INT64",
        "n_arr_del15": "INT64",
        "n_dep_del15": "INT64",
        "n_cancelled": "INT64",
        "n_diverted": "INT64",
        "sum_arr_delay_minutes": "FLOAT64",
        "sum_dep_delay_minutes": "FLOAT64",
    },
}

_AIRPORTS = [
    ("ORD", "Chicago O'Hare International Airport", "Chicago, IL", "America/Chicago"),
    ("LAX", "Los Angeles International Airport", "Los Angeles, CA", "America/Los_Angeles"),
    ("ATL", "Hartsfield-Jackson Atlanta International Airport", "Atlanta, GA", "America/New_York"),
    ("DFW", "Dallas/Fort Worth International Airport", "Dallas/Fort Worth, TX", "America/Chicago"),
]
_CARRIERS = [("AA", "American Airlines"), ("UA", "United Airlines"), ("DL", "Delta Air Lines")]
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]  # fmt: skip
_SEASONS = [("Winter", 1), ("Spring", 2), ("Summer", 3), ("Autumn", 4)]


def _value(col: str, bq_type: str, i: int):
    """One plausible cell.

    Values must survive each page's DEFAULT filters, which is the whole point:
    the route page defaults to a 100-leg minimum, so a fixture with small counts
    would take the ``df.empty`` early return and the render test would sail past
    the very line that crashed in production. Counts are therefore large and
    every rate sits in a believable band.
    """
    # Counts are sized to clear EVERY page's default minimum-traffic filter.
    # The binding one is the delay map's 10,000-leg slider default: a smaller
    # fixture filters to nothing, the page renders its empty state, and the test
    # sails past the real code below it having proved nothing.
    legs = 40_000 + i * 9_137
    origin, o_name, o_city, o_tz = _AIRPORTS[i % len(_AIRPORTS)]
    dest, d_name, d_city, _ = _AIRPORTS[(i + 1) % len(_AIRPORTS)]
    carrier, c_name = _CARRIERS[i % len(_CARRIERS)]
    season, season_order = _SEASONS[i % len(_SEASONS)]

    exact = {
        "airport_key": origin,
        "airport_name": o_name,
        "city": o_city,
        "tz": o_tz,
        "origin_airport_key": origin,
        "origin_airport_name": o_name,
        "origin_city": o_city,
        "dest_airport_key": dest,
        "dest_airport_name": d_name,
        "dest_city": d_city,
        "route": f"{origin}-{dest}",
        "carrier_key": carrier,
        "carrier_name": c_name,
        "is_regional": False,
        "dot_id": 19000 + i,
        "season": season,
        "season_order": season_order,
        "day_of_week": (i % 7) + 1,
        "day_name": _DAYS[i % 7],
        "dep_hour": i % 24,
        "crs_dep_hour": i % 24,
        "year": 2022 + (i % 3),
        "month": (i % 12) + 1,
        "month_name": _MONTHS[i % 12],
        "month_start": pd.Timestamp(2022 + (i % 3), (i % 12) + 1, 1).date(),
        # Derived from `legs` so the SUM/SUM rates metrics.py computes land in
        # believable bands (~19% delayed, ~1.5% cancelled) rather than nonsense
        # that a page might reasonably refuse to plot.
        "n_flight_legs": legs,
        "n_flights": legs,
        "n_arr_del15": int(legs * 0.19),
        "n_dep_del15": int(legs * 0.17),
        "n_with_arr_outcome": int(legs * 0.98),
        "n_with_dep_outcome": int(legs * 0.99),
        "n_cancelled": int(legs * 0.015),
        "n_diverted": int(legs * 0.002),
        "hist_n_flights": int(legs * 0.8),
        "arr_del15_rate": 0.15 + (i % 10) * 0.01,
        "hist_arr_del15_rate": 0.15 + (i % 10) * 0.01,
        "on_time_rate": 0.85 - (i % 10) * 0.01,
        "cancellation_rate": 0.01 + (i % 5) * 0.001,
        "diversion_rate": 0.002,
        "avg_arr_delay_minutes": 10.0 + i,
        "p90_arr_delay_minutes": 45.0 + i,
        "sum_arr_delay_minutes": legs * 12.0,
        "sum_dep_delay_minutes": legs * 11.0,
    }
    if col in exact:
        return exact[col]
    # Unknown column: fall back on the declared type so a newly added column
    # still produces a renderable frame rather than a fixture KeyError.
    if bq_type == "STRING":
        return f"{col}_{i}"
    if bq_type == "BOOL":
        return False
    if bq_type == "DATE":
        return pd.Timestamp(2024, 1, 1).date()
    return float(i) if bq_type == "FLOAT64" else i


def make_frame(view: str, rows: int = 24) -> pd.DataFrame:
    """A synthetic frame with EXACTLY the columns the live view has."""
    cols = SCHEMAS[view]
    return pd.DataFrame(
        {c: [_value(c, t, i) for i in range(rows)] for c, t in cols.items()},
    )


def make_airport_coords() -> pd.DataFrame:
    """``data.airport_coords()`` — dim_airport, not a dash_ view."""
    lat = {"ORD": 41.9742, "LAX": 33.9416, "ATL": 33.6407, "DFW": 32.8998}
    lon = {"ORD": -87.9073, "LAX": -118.4085, "ATL": -84.4277, "DFW": -97.0403}
    return pd.DataFrame(
        {
            "airport_key": list(lat),
            "latitude": [lat[k] for k in lat],
            "longitude": [lon[k] for k in lat],
        }
    )
