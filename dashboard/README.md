# Dashboard — the lakehouse end product

A Streamlit app that serves the curated **gold** layer to a non-technical
consumer (CLAUDE.md §4, assignment requirement #3). It reads the five gold
`dash_*` views **live from BigQuery via ADC** — the same gold layer the ML
feature mart is built from, so nothing is duplicated or recomputed here.

Why Streamlit (vs. the Looker Studio spec in `docs/dashboard/`): it lives in the
repo as reviewable, runnable code, fits the project's `uv`/Python stack, and can
be committed and version-controlled. The Looker Studio spec remains a valid
click-to-build alternative for a fully-managed BI surface.

## Run it

```bash
# 1. Auth (once) + config
gcloud auth application-default login
cp .env.example .env          # set GCP_PROJECT_ID and BQ_GOLD_DATASET

# 2. Install the dashboard extra and launch
uv sync --extra dashboard
uv run --extra dashboard streamlit run dashboard/app.py
```

Opens at http://localhost:8501. Requires BigQuery read access to the gold
dataset (`roles/bigquery.dataViewer` on `<project>.flight_delays_gold`).

## Layout

```
dashboard/
├── app.py         # entry point; registers pages via st.navigation
├── config.py      # project/dataset from env (ADC; no hardcoding)
├── data.py        # cached BigQuery loaders, one per dash_* view
├── metrics.py     # SUM/SUM rate math for the additive views
├── ui.py          # shared formatting + severity color palette
└── views/         # one module per page, each exposing render()
    └── overview.py
```

## The one correctness rule

The additive views (`dash_delays_by_time`, `dash_monthly_trend`) carry counts,
not rates. Every rate at a rolled-up grain is `SUM(numerator) / SUM(denominator)`
via `dashboard/metrics.py` — never the average of a per-row rate. Phase 5 adds a
harness that checks the app's numbers against direct BigQuery aggregates.
