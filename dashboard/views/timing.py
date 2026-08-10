"""Page 2 — "When do delays happen?"

Temporal structure of delays: day of week, month, month-over-month with a
year-over-year overlay, and a day×hour heatmap. Everything is aggregated from
the additive time view through metrics.aggregate (SUM/SUM), and the year/month
filter controls apply to every chart on the page.
"""

from __future__ import annotations

import streamlit as st

from dashboard import charts, data, metrics, ui
from dashboard import flights as fl

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def _month_options(df) -> list[int]:
    return sorted(int(m) for m in df["month"].dropna().unique())


def _ordered(df, order_col, label_col):
    """Distinct label values sorted by their companion order column."""
    return df.sort_values(order_col)[label_col].drop_duplicates().tolist()


def render() -> None:
    st.title("When do delays happen?")
    st.caption(
        "The temporal shape of delays — by day of week, month, and over time. "
        "Rates are SUM/SUM over additive counts; filters below apply to every chart."
    )

    # ---- who / where ----
    # Unfiltered uses the small pre-aggregated view (whole network, cached, no
    # scan). Choosing an airport or an airline switches to the sliceable mart,
    # which is queried with a predicate rather than read whole — see
    # data.time_slice for why the two paths exist.
    airports = data.airport_reliability()[["airport_key", "airport_name"]]
    airport_labels = {
        r["airport_key"]: fl.airport_label(r["airport_key"], r["airport_name"])
        for _, r in airports.iterrows()
    }
    cdf = data.carrier_reliability()
    carrier_labels = {
        r["carrier_key"]: fl.carrier_label(
            r["carrier_key"], r.get("carrier_name"), bool(r.get("is_regional"))
        )
        for _, r in cdf.iterrows()
    }
    carriers = sorted(carrier_labels)

    w1, w2 = st.columns(2)
    origin = w1.selectbox(
        "Departing airport",
        ["All airports", *sorted(airport_labels)],
        format_func=lambda c: "All airports" if c == "All airports" else airport_labels.get(c, c),
    )
    carrier = w2.selectbox(
        "Airline",
        ["All airlines", *carriers],
        format_func=lambda c: "All airlines" if c == "All airlines" else carrier_labels.get(c, c),
    )
    origin = None if origin == "All airports" else origin
    carrier = None if carrier == "All airlines" else carrier

    dbt = data.time_slice(origin, carrier) if (origin or carrier) else data.delays_by_time()
    if dbt.empty:
        st.warning("No flights match that airport/airline combination.")
        return

    # ---- when ----
    years = sorted(dbt["year"].unique())
    months = _month_options(dbt)
    f1, f2 = st.columns(2)
    sel_years = f1.multiselect("Year", years, default=years)
    # MONTH, not season: a season hides the thing people actually plan around
    # (a specific month), and four buckets cannot show the December peak next
    # to a quiet November.
    sel_months = f2.multiselect(
        "Month", months, default=months, format_func=lambda m: MONTH_NAMES[m - 1]
    )
    df = dbt[dbt["year"].isin(sel_years or years) & dbt["month"].isin(sel_months or months)]
    if df.empty:
        st.warning("No data for the selected filters.")
        return
    scope = " · ".join(
        [
            airport_labels.get(origin, origin) if origin else "All airports",
            carrier_labels.get(carrier, carrier) if carrier else "All airlines",
        ]
    )
    st.caption(f"Showing: **{scope}** — {int(df['n_flights'].sum()):,} flights")

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

    # ---- month ----
    # By MONTH, not season. Season buckets three months together and hides the
    # shape people plan around: December's cancellations do not look like
    # January's, and a summer bucket flattens the July peak into June and
    # August. Twelve columns still read fine and the filter above matches it.
    with right:
        mon = metrics.aggregate(df, ["month"]).sort_values("month")
        mon["month_label"] = mon["month"].map(lambda m: MONTH_NAMES[int(m) - 1][:3])
        fig = charts.grouped_rate_col_v(
            mon,
            category="month_label",
            value_vars={
                "delay_rate": "Delay rate",
                "cancellation_rate": "Cancellation rate",
            },
            title="Delay vs cancellation rate by month",
            category_order=[MONTH_NAMES[int(m) - 1][:3] for m in mon["month"]],
        )
        st.plotly_chart(fig, use_container_width=True)
        peak = mon.loc[mon["delay_rate"].idxmax()]
        cx = mon.loc[mon["cancellation_rate"].idxmax()]
        st.caption(
            f"Worst month for **delays**: {MONTH_NAMES[int(peak['month']) - 1]} "
            f"({ui.pct(peak['delay_rate'])}). Worst for **cancellations**: "
            f"{MONTH_NAMES[int(cx['month']) - 1]} ({ui.pct(cx['cancellation_rate'])}) — "
            "two different operational failure modes."
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
