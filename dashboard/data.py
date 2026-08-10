"""Live BigQuery data access for the dashboard.

One loader per gold ``dash_*`` view. Every view is a thin, pre-aggregated skin
over a materialized mart (≤7.6k rows, <1 MB full scan), so we simply ``SELECT *``
and cache the frame — a full page load, even with re-querying, scans well under
1 MB and never touches the 20.6M-row ``fact_flights`` (see dashboard_spec.md).

Caching: the BigQuery client is a cached resource; each view frame is cached for
``CACHE_TTL`` seconds so repeated interactions don't re-bill the same query.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from google.cloud import bigquery

from dashboard.config import fq_view, gcp_project, gold_dataset

CACHE_TTL = 3600  # views change only on a dbt rebuild; 1h is ample

# View name -> short description (drives the Overview health panel).
# The views that are safe to read WHOLE. dash_time_slice is deliberately
# absent: it is ~2.3M rows and is queried with a predicate by time_slice().
# Anything listed here gets SELECT *-ed by the Overview health panel.
VIEWS: dict[str, str] = {
    "dash_route_carrier": "one row per route x airline",
    "dash_airport_reliability": "1 row / origin airport",
    "dash_carrier_reliability": "1 row / carrier",
    "dash_delays_by_time": "year × month × day-of-week × dep-hour",
    "dash_monthly_trend": "1 row / calendar month",
    "dash_route_drilldown": "1 row / directed route",
    "dash_airport_hour_baseline": "airport × day-of-week × dep-hour",
}


@st.cache_resource(show_spinner=False)
def _client() -> bigquery.Client:
    return bigquery.Client(project=gcp_project())


@st.cache_data(ttl=CACHE_TTL, show_spinner="Querying BigQuery…")
def load_view(view: str) -> pd.DataFrame:
    """Load an entire gold dashboard view as a DataFrame (cached)."""
    if view not in VIEWS:
        raise KeyError(f"unknown dashboard view: {view!r}")
    return (
        _client().query(f"SELECT * FROM {fq_view(view)}").to_dataframe(create_bqstorage_client=True)
    )


# Convenience accessors — named so pages read clearly.
def airport_reliability() -> pd.DataFrame:
    return load_view("dash_airport_reliability")


def carrier_reliability() -> pd.DataFrame:
    return load_view("dash_carrier_reliability")


def delays_by_time() -> pd.DataFrame:
    return load_view("dash_delays_by_time")


def monthly_trend() -> pd.DataFrame:
    return load_view("dash_monthly_trend")


def route_drilldown() -> pd.DataFrame:
    return load_view("dash_route_drilldown")


def airport_hour_baseline() -> pd.DataFrame:
    return load_view("dash_airport_hour_baseline")


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def gold_freshness() -> pd.Timestamp | None:
    """Most recent build time across the gold *tables* (the marts the dashboard
    reads through views) — i.e. when the last dbt run refreshed the data.
    Returns a UTC Timestamp, or None if it can't be read."""
    sql = (
        "SELECT TIMESTAMP_MILLIS(MAX(last_modified_time)) AS t "
        f"FROM `{gcp_project()}.{gold_dataset()}.__TABLES__` "
        "WHERE type = 1"  # 1 = base table, 2 = view
    )
    try:
        rows = list(_client().query(sql))
        return pd.Timestamp(rows[0].t) if rows and rows[0].t else None
    except Exception:  # never let a metadata hiccup break the page
        return None


@st.cache_data(ttl=CACHE_TTL, show_spinner="Querying BigQuery…")
def airport_coords() -> pd.DataFrame:
    """Airport lat/long from the gold ``dim_airport`` (374 rows) — used to place
    airports on the map. Kept separate from the ``dash_*`` views since it is a
    dimension, not a pre-aggregated metric view."""
    sql = (
        "SELECT airport_key, latitude, longitude "
        f"FROM `{gcp_project()}.{gold_dataset()}.dim_airport` "
        "WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
    )
    return _client().query(sql).to_dataframe(create_bqstorage_client=True)


def route_carrier() -> pd.DataFrame:
    """Route x airline — who flies a route and how reliably. 16,463 rows."""
    return load_view("dash_route_carrier")


@st.cache_data(ttl=CACHE_TTL, show_spinner="Querying BigQuery…")
def time_slice(origin: str | None = None, carrier: str | None = None) -> pd.DataFrame:
    """Time-of-travel counts for ONE airport and/or airline.

    The only loader that does not read its view whole. dash_time_slice is
    ~2.3M rows, so it is queried WITH a predicate; the base mart is clustered
    on (origin_airport_key, carrier_key) and the filtered read is 39 MB against
    246 MB for the table — measured on executed jobs, because a dry run on a
    clustered table reports an upper bound and shows no pruning at all
    (docs/benchmarks/README.md makes the same point).

    Unfiltered callers should use delays_by_time() instead: the small
    pre-aggregated view answers the same questions for the whole network and
    costs nothing.
    """
    where, params = [], []
    if origin:
        where.append("origin_airport_key = @origin")
        params.append(bigquery.ScalarQueryParameter("origin", "STRING", origin))
    if carrier:
        where.append("carrier_key = @carrier")
        params.append(bigquery.ScalarQueryParameter("carrier", "STRING", carrier))
    sql = f"SELECT * FROM {fq_view('dash_time_slice')}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    return (
        _client()
        .query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
        .to_dataframe(create_bqstorage_client=True)
    )
