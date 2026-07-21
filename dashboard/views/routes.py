"""Page 3 — "Route drill-down".

Delay profile for any origin → destination. The route view carries rates at its
native directed-route grain, so rates display as-is; filters narrow the set and
the table + scatter update together. Busy-and-late routes surface in the
top-right of the scatter.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts, data, ui


def _airport_options(df: pd.DataFrame, key_col: str, name_col: str) -> dict[str, str]:
    """Map 'KEY — Name' label -> key, sorted, for a searchable picker."""
    pairs = df[[key_col, name_col]].drop_duplicates().sort_values(key_col)
    return {f"{k} — {n}": k for k, n in zip(pairs[key_col], pairs[name_col], strict=False)}


def render() -> None:
    st.title("Route drill-down")
    st.caption(
        "Delay profile for any origin → destination pair. Rates are at the "
        "directed-route grain (one row per origin→dest)."
    )

    routes = data.route_drilldown()

    origins = _airport_options(routes, "origin_airport_key", "origin_airport_name")
    dests = _airport_options(routes, "dest_airport_key", "dest_airport_name")

    c1, c2, c3 = st.columns([2, 2, 2])
    sel_o = c1.multiselect("Origin airport(s)", list(origins), placeholder="All origins")
    sel_d = c2.multiselect("Destination airport(s)", list(dests), placeholder="All destinations")
    min_legs = c3.slider("Minimum flight legs", 1, 5_000, 100, step=50)

    df = routes
    if sel_o:
        df = df[df["origin_airport_key"].isin(origins[o] for o in sel_o)]
    if sel_d:
        df = df[df["dest_airport_key"].isin(dests[d] for d in sel_d)]
    df = df[df["n_flight_legs"] >= min_legs]

    if df.empty:
        st.warning("No routes match these filters. Try lowering the minimum flight legs.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Routes shown", ui.count(len(df)))
    m2.metric("Flight legs covered", ui.count(df["n_flight_legs"].sum()))
    weighted = (df["arr_del15_rate"] * df["n_flight_legs"]).sum() / df["n_flight_legs"].sum()
    m3.metric("Volume-weighted delay rate", ui.pct(weighted))

    st.divider()

    # ---- route table (sorted by volume, conditional formatting on delay rate) ----
    st.subheader("Routes")
    table = (
        df[
            [
                "route",
                "origin_city",
                "dest_city",
                "n_flight_legs",
                "arr_del15_rate",
                "avg_arr_delay_minutes",
                "p90_arr_delay_minutes",
                "cancellation_rate",
            ]
        ]
        .sort_values("n_flight_legs", ascending=False)
        .reset_index(drop=True)
    )
    styled = table.style.background_gradient(
        subset=["arr_del15_rate"], cmap="OrRd", vmin=0, vmax=0.5
    ).background_gradient(subset=["cancellation_rate"], cmap="Purples", vmin=0, vmax=0.1)
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=430,
        column_config={
            "route": "Route",
            "origin_city": "Origin city",
            "dest_city": "Dest city",
            "n_flight_legs": st.column_config.NumberColumn("Legs", format="%d"),
            "arr_del15_rate": st.column_config.NumberColumn("Delay rate", format="percent"),
            "avg_arr_delay_minutes": st.column_config.NumberColumn("Avg delay", format="%.1f min"),
            "p90_arr_delay_minutes": st.column_config.NumberColumn("P90 delay", format="%.0f min"),
            "cancellation_rate": st.column_config.NumberColumn("Cancel rate", format="percent"),
        },
    )
    st.caption(
        f"{len(table):,} routes · sorted by traffic. Delay-rate cells shaded red, "
        "cancellation-rate cells shaded purple, by intensity."
    )

    st.divider()

    # ---- traffic vs delay scatter ----
    fig = charts.scatter_volume_vs_rate(
        df,
        x="n_flight_legs",
        y="arr_del15_rate",
        hover_name="route",
        hover_cols=["origin_city", "dest_city", "avg_arr_delay_minutes"],
        title="Traffic vs delay rate — busy-and-late routes sit top-right",
    )
    st.plotly_chart(fig, use_container_width=True)
