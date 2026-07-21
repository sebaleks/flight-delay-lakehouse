"""Streamlit entry point for the flight-delay dashboard.

Run:  uv run --extra dashboard streamlit run dashboard/app.py

Auth is ADC (`gcloud auth application-default login`); project/dataset come from
`.env` (GCP_PROJECT_ID / BQ_GOLD_DATASET). Pages are registered below; each new
page is one module in ``dashboard/views/`` exposing ``render()``.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Flight-Delay Lakehouse",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _overview() -> None:
    from dashboard.views import overview

    overview.render()


def _reliability() -> None:
    from dashboard.views import reliability

    reliability.render()


PAGES = [
    st.Page(_overview, title="Overview", icon="🏠", default=True),
    st.Page(_reliability, title="Who is reliable?", icon="🏆"),
    # Phase 3: When do delays happen?  ·  Phase 4: Route drill-down
]


def main() -> None:
    with st.sidebar:
        st.markdown("### ✈️ Flight-Delay Lakehouse")
        st.caption("Gold layer · BigQuery · BTS 2022–2024")
    st.navigation(PAGES).run()


main()
