"""Page 2 — "When do delays happen?"

Temporal structure of delays: day of week, season, month-over-month with a
year-over-year overlay, and a day×hour heatmap. Everything is aggregated from
the additive time view through metrics.aggregate (SUM/SUM), and the year/season
filter controls apply to every chart on the page.
"""

from __future__ import annotations

import streamlit as st

from dashboard import charts, data, metrics, ui


def _ordered(df, order_col, label_col):
    """Distinct label values sorted by their companion order column."""
    return df.sort_values(order_col)[label_col].drop_duplicates().tolist()


def render() -> None:
    st.title("When do delays happen?")
    st.caption(
        "The temporal shape of delays — by day of week, season, and over time. "
        "Rates are SUM/SUM over additive counts; filters below apply to every chart."
    )

    dbt = data.delays_by_time()

    # ---- page-level filters ----
    years = sorted(dbt["year"].unique())
    seasons = _ordered(dbt, "season_order", "season")
    f1, f2 = st.columns(2)
    sel_years = f1.multiselect("Year", years, default=years)
    sel_seasons = f2.multiselect("Season", seasons, default=seasons)
    df = dbt[dbt["year"].isin(sel_years or years) & dbt["season"].isin(sel_seasons or seasons)]
    if df.empty:
        st.warning("No data for the selected filters.")
        return

    st.divider()

    left, right = st.columns(2)

    # ---- day of week ----
    with left:
        dow = metrics.aggregate(df, ["day_of_week", "day_name"])
        fig = charts.rate_col_v(
            dow.sort_values("day_of_week"),
            category="day_name",
            rate_col="delay_rate",
            title="Delay rate by day of week",
            category_order=_ordered(dow, "day_of_week", "day_name"),
        )
        st.plotly_chart(fig, use_container_width=True)
        worst = dow.loc[dow["delay_rate"].idxmax()]
        st.caption(f"Worst day: **{worst['day_name']}** ({ui.pct(worst['delay_rate'])}).")

    # ---- season ----
    with right:
        seas = metrics.aggregate(df, ["season_order", "season"]).sort_values("season_order")
        fig = charts.grouped_rate_col_v(
            seas,
            category="season",
            value_vars={
                "delay_rate": "Delay rate",
                "cancellation_rate": "Cancellation rate",
            },
            title="Delay vs cancellation rate by season",
            category_order=_ordered(seas, "season_order", "season"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Summer drives **delays**, winter drives **cancellations** — two "
            "different operational failure modes."
        )

    st.divider()

    # ---- monthly trend, year over year ----
    monthly = metrics.aggregate(df, ["year", "month", "month_name"])
    fig = charts.rate_line_by_year(
        monthly,
        x="month_name",
        rate_col="delay_rate",
        year_col="year",
        title="Delay rate by month, year over year",
        x_order=_ordered(dbt, "month", "month_name"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---- day × hour heatmap ----
    grid = metrics.aggregate(df, ["day_of_week", "day_name", "dep_hour"])
    pivot = grid.pivot(index="day_name", columns="dep_hour", values="delay_rate").reindex(
        _ordered(dbt, "day_of_week", "day_name")
    )
    fig = charts.rate_heatmap(
        pivot,
        title="Delay rate by day of week × scheduled departure hour",
        x_title="Departure hour",
        y_title="Day",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The bright block — weekday evenings — is where the system is most "
        "congested; early mornings any day are the safest window."
    )
