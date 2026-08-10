"""Overview — the landing page.

Proves the lakehouse serves end-to-end: it queries all five gold views live and
surfaces headline reliability metrics computed the correct way (SUM/SUM). The
per-view health table doubles as a smoke test that every view is reachable.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import data, metrics, ui


def render() -> None:
    st.title("✈️ US Flight-Delay Lakehouse")
    st.caption(
        "Curated gold layer, served live from BigQuery. US domestic on-time "
        "performance (BTS), 2022–2024 — one row per flight leg upstream, "
        "pre-aggregated here so every interaction scans well under 1 MB."
    )

    trend = data.monthly_trend()
    overall = metrics.totals(trend)

    span = (
        f"{pd.to_datetime(trend['month_start'].min()):%b %Y}"
        f" – {pd.to_datetime(trend['month_start'].max()):%b %Y}"
    )
    st.markdown(f"**Coverage:** {span}  ·  {len(trend)} months")

    # year-over-year deltas: latest full year vs the one before it
    by_year = metrics.aggregate(trend, "year").sort_values("year")
    latest, prev = by_year.iloc[-1], by_year.iloc[-2]

    def pp(cur, before):  # percentage-point delta as a signed string
        return f"{(cur - before) * 100:+.1f} pp"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Total flights",
        ui.count(overall["n_flights"]),
        f"{latest['n_flights'] - prev['n_flights']:+,.0f} vs {int(prev['year'])}",
        delta_color="off",
    )
    c2.metric(
        "Arrival delay rate (≥15 min)",
        ui.pct(overall["delay_rate"]),
        pp(latest["delay_rate"], prev["delay_rate"]),
        delta_color="inverse",  # a higher delay rate is worse
    )
    c3.metric(
        "Cancellation rate",
        ui.pct(overall["cancellation_rate"], digits=2),
        pp(latest["cancellation_rate"], prev["cancellation_rate"]),
        delta_color="inverse",
    )
    c4.metric(
        "Avg arrival delay",
        ui.minutes(overall["avg_arr_delay_minutes"]),
        f"{latest['avg_arr_delay_minutes'] - prev['avg_arr_delay_minutes']:+.1f} min",
        delta_color="inverse",
    )
    st.caption(f"Δ = {int(latest['year'])} vs {int(prev['year'])} (full-year).")

    st.divider()
    st.subheader("What this dashboard answers")
    st.markdown(
        "- **Who is reliable?** — airport & carrier delay/cancellation rankings\n"
        "- **When do delays happen?** — hour of day, day of week, season, trend over time\n"
        "- **Route drill-down** — delay profile for any origin → destination\n\n"
        "Use the sidebar to navigate. All rates are computed as "
        "`SUM(numerator) / SUM(denominator)` over additive counts, never an "
        "average of per-row rates."
    )

    with st.expander("Data sources — live view health", expanded=False):
        rows = []
        for view, grain in data.VIEWS.items():
            df = data.load_view(view)
            rows.append({"view": view, "grain": grain, "rows": len(df)})
        health = pd.DataFrame(rows)
        st.dataframe(
            health,
            hide_index=True,
            use_container_width=True,
            height=ui.table_height(len(health)),
            column_config={"rows": st.column_config.NumberColumn(format="%d")},
        )
        st.caption(
            "Thin views over materialized gold marts — the same layer the "
            "ML feature mart is built from (no duplicated data or logic)."
        )
