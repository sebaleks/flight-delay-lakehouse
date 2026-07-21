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

from dashboard.config import fq_view, gcp_project

CACHE_TTL = 3600  # views change only on a dbt rebuild; 1h is ample

# View name -> short description (drives the Overview health panel).
VIEWS: dict[str, str] = {
    "dash_airport_reliability": "1 row / origin airport",
    "dash_carrier_reliability": "1 row / carrier",
    "dash_delays_by_time": "year × month × day-of-week × dep-hour",
    "dash_monthly_trend": "1 row / calendar month",
    "dash_route_drilldown": "1 row / directed route",
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
