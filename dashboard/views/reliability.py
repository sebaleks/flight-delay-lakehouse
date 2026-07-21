"""Page 1 — "Who is reliable?"

Airport and carrier reliability rankings plus the hour-of-day delay curve. The
airport/carrier views carry rates at their native (per-airport / per-carrier)
grain, so those rates are display-safe as-is; the hour curve is aggregated from
the additive time view through metrics.aggregate (SUM/SUM).
"""

from __future__ import annotations

import streamlit as st

from dashboard import charts, data, metrics, ui


def render() -> None:
    st.title("Who is reliable?")
    st.caption(
        "Delay and cancellation rankings across airports and carriers, and how "
        "the delay rate builds through the day. Full period, 2022–2024."
    )

    # ---- scorecard row (SUM/SUM over the monthly view) ----
    overall = metrics.totals(data.monthly_trend())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total flights", ui.count(overall["n_flights"]))
    c2.metric("Arrival delay rate (≥15 min)", ui.pct(overall["delay_rate"]))
    c3.metric("Cancellation rate", ui.pct(overall["cancellation_rate"], digits=2))
    c4.metric("Avg arrival delay", ui.minutes(overall["avg_arr_delay_minutes"]))
    st.divider()

    airports = data.airport_reliability()
    carriers = data.carrier_reliability()

    # ---- controls for the airport ranking ----
    ctrl1, ctrl2 = st.columns([2, 1])
    min_legs = ctrl1.slider(
        "Minimum flight legs (airports below this are excluded from the ranking)",
        min_value=1_000,
        max_value=100_000,
        value=10_000,
        step=1_000,
        help="Keeps tiny airfields with a handful of flights from topping the "
        "worst-reliability chart on noise.",
    )
    top_n = ctrl2.number_input("Show top N", min_value=5, max_value=50, value=15, step=5)

    left, right = st.columns(2)

    with left:
        elig = airports[airports["n_flight_legs"] >= min_legs]
        worst = elig.nlargest(int(top_n), "arr_del15_rate")
        fig = charts.rate_bar_h(
            worst,
            category="airport_name",
            rate_col="arr_del15_rate",
            title=f"Least reliable airports (≥{min_legs:,} legs)",
            hover_cols=["city", "n_flight_legs", "cancellation_rate"],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"{len(elig)} of {len(airports)} airports clear the {min_legs:,}-leg bar. "
            "Rate = delayed arrivals ÷ arrivals with an outcome, at airport grain."
        )

    with right:
        fig = charts.grouped_rate_bar_h(
            carriers,
            category="carrier_key",
            value_vars={
                "arr_del15_rate": "Delay rate",
                "cancellation_rate": "Cancellation rate",
            },
            title="Least reliable carriers (all 17)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Two-letter BTS carrier codes (the source ships no display names).")

    st.divider()

    # ---- hour-of-day delay curve (aggregated SUM/SUM) ----
    by_hour = metrics.aggregate(data.delays_by_time(), "dep_hour").sort_values("dep_hour")
    fig = charts.rate_col_v(
        by_hour,
        category="dep_hour",
        rate_col="delay_rate",
        title="Arrival delay rate by scheduled departure hour",
        color=ui.COLOR_DELAY,
    )
    fig.update_xaxes(title="Scheduled departure hour (0–23)", dtick=1)
    st.plotly_chart(fig, use_container_width=True)
    peak = by_hour.loc[by_hour["delay_rate"].idxmax()]
    st.caption(
        f"Delays compound through the day — the peak is hour {int(peak['dep_hour'])}:00 "
        f"at {ui.pct(peak['delay_rate'])}, roughly "
        f"{peak['delay_rate'] / by_hour['delay_rate'].min():.1f}× the early-morning low."
    )
