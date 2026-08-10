"""Delay map — airport reliability across the US at a glance.

Joins the airport reliability view to `dim_airport` coordinates and plots one
bubble per airport: position = location, size = traffic, color = delay rate. The
most immediately readable view for a non-technical audience — hubs light up hot
or cool without reading a table.
"""

from __future__ import annotations

import streamlit as st

from dashboard import charts, data, ui


def render() -> None:
    st.title("Delay map")
    st.caption(
        "Every US origin airport, placed by location, sized by traffic, colored "
        "by arrival-delay rate. Full period, 2022–2024."
    )

    airports = data.airport_reliability()
    coords = data.airport_coords()
    df = airports.merge(coords, on="airport_key", how="inner")

    min_legs = st.slider(
        "Minimum flight legs",
        1_000,
        100_000,
        10_000,
        step=1_000,
        help="Hide small airfields so the map reflects meaningful traffic.",
    )
    shown = df[df["n_flight_legs"] >= min_legs]

    if shown.empty:
        # idxmax/idxmin below raise "attempt to get argmax of an empty sequence"
        # on an empty frame, which takes the whole app down (app.py has no error
        # boundary). Today no slider position can empty this — ATL alone clears
        # the 100,000 ceiling — so the guard is protection against the data
        # changing, not a bug a user can currently reach. The route page already
        # guards the same way.
        st.warning("No airports carry that much traffic. Lower the minimum flight legs.")
        return

    fig = charts.airport_map(
        shown,
        lat="latitude",
        lon="longitude",
        size="n_flight_legs",
        color="arr_del15_rate",
        hover_name="airport_name",
        hover_cols=["city", "n_flight_legs", "arr_del15_rate", "cancellation_rate"],
        title="Arrival delay rate by airport",
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Airports shown", ui.count(len(shown)))
    worst = shown.loc[shown["arr_del15_rate"].idxmax()]
    best = shown.loc[shown["arr_del15_rate"].idxmin()]
    c2.metric(
        "Hottest",
        worst["airport_name"].split("/")[0][:22],
        ui.pct(worst["arr_del15_rate"]),
        delta_color="off",
    )
    c3.metric(
        "Coolest",
        best["airport_name"].split("/")[0][:22],
        ui.pct(best["arr_del15_rate"]),
        delta_color="off",
    )
    st.caption(
        "Bubble size scales with flight volume; color runs cool→hot with delay "
        "rate. Hover any airport for its exact numbers."
    )
