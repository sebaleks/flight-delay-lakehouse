"""Render every dashboard page and assert it does not raise.

The gap this closes: every other dashboard test exercises a PURE FUNCTION
(metrics, uncertainty, flights, capacity). None of them ever called a
``views/*.render()``, so a page could reference a column its dataframe does not
have and nothing failed until a visitor opened the tab — at which point
``dashboard/app.py``, which has no per-page error boundary, showed a traceback
instead of the app. That is exactly how ``sort_values("n_flights")`` on the
route-grain frame reached production.

No credentials and no network: ``dashboard.data``'s loaders are replaced with
frames built from ``dashboard.schemas.SCHEMAS``, which carries the real
BigQuery column lists. So this runs in CI on every push, and
``test_verify.test_schemas_match_bigquery`` stops the fixtures drifting away
from the warehouse behind its back.

What it does and does not prove: it proves each page renders on the DEFAULT
filter state with plausible data. It does not prove every interactive branch
renders — an unclicked tab body still executes, but a code path behind a button
press does not.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dashboard import schemas

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = [
    "overview",
    "reliability",
    "timing",
    "routes",
    "map_view",
    "predict_flight",
    "ops_capacity",
]


@pytest.fixture
def stub_data(monkeypatch):
    """Point every loader at fixture frames instead of BigQuery."""
    from dashboard import data

    # Patching load_view covers airport_reliability / carrier_reliability /
    # delays_by_time / monthly_trend / route_drilldown / route_carrier /
    # airport_hour_baseline in one move: they are thin wrappers over it.
    monkeypatch.setattr(data, "load_view", lambda view: schemas.make_frame(view))
    monkeypatch.setattr(data, "airport_coords", schemas.make_airport_coords)
    monkeypatch.setattr(data, "gold_freshness", lambda: pd.Timestamp("2026-08-01T00:00:00Z"))

    def _time_slice(origin: str | None = None, carrier: str | None = None) -> pd.DataFrame:
        df = schemas.make_frame("dash_time_slice")
        if origin:
            df = df[df["origin_airport_key"] == origin]
        if carrier:
            df = df[df["carrier_key"] == carrier]
        return df.reset_index(drop=True)

    monkeypatch.setattr(data, "time_slice", _time_slice)

    # config reads these through require_env, which raises SystemExit when unset.
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    monkeypatch.setenv("BQ_GOLD_DATASET", "test_gold")


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_raising(page: str, stub_data) -> None:
    # from_string, not from_function: AppTest.from_function re-executes the
    # function's SOURCE in a bare namespace, so the view module's own imports
    # (st, pandas, dashboard.*) are undefined and every page fails with a
    # NameError that has nothing to do with the page. Importing inside the
    # script keeps the real module, and the monkeypatched dashboard.data is the
    # same object because the script runs in this process.
    at = AppTest.from_string(
        f"from dashboard.views import {page}\n{page}.render()\n",
        default_timeout=60,
    )
    at.run()

    assert not at.exception, (
        f"{page}.render() raised: {[e.value for e in at.exception]}\n"
        "A page that raises takes the WHOLE app down — app.py has no error boundary."
    )


def test_fixture_columns_are_exactly_the_view_columns() -> None:
    """The fixtures must not be a superset — a frame carrying a column the real
    view lacks would let a broken reference pass here and fail in production,
    which is the precise failure mode this file exists to prevent."""
    for view, cols in schemas.SCHEMAS.items():
        assert list(schemas.make_frame(view).columns) == list(cols), view


def test_the_two_grains_stay_distinct() -> None:
    """Pins the naming trap itself.

    Entity-grain views count with ``n_flight_legs`` and ship pre-divided rates;
    additive views count with ``n_flights`` and ship none. If a dbt change ever
    makes both names valid on one view, the confusion that caused the route-page
    crash becomes silent, so fail loudly here instead.
    """
    entity = ["dash_airport_reliability", "dash_carrier_reliability", "dash_route_drilldown"]
    additive = [
        "dash_delays_by_time",
        "dash_monthly_trend",
        "dash_route_carrier",
        "dash_time_slice",
        "dash_airport_hour_baseline",
    ]
    for v in entity:
        cols = schemas.SCHEMAS[v]
        assert "n_flight_legs" in cols and "n_flights" not in cols, v
        assert "arr_del15_rate" in cols, v
    for v in additive:
        cols = schemas.SCHEMAS[v]
        assert "n_flights" in cols and "n_flight_legs" not in cols, v
        assert "arr_del15_rate" not in cols, f"{v} ships a pre-divided rate; rates must be SUM/SUM"
