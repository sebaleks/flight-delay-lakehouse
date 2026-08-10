"""Page 3 — "Route drill-down".

Delay profile for any origin → destination. The route view carries rates at its
native directed-route grain, so rates display as-is; filters narrow the set and
the table + scatter update together. Busy-and-late routes surface in the
top-right of the scatter.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts, data, metrics, ui
from dashboard import flights as fl


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

    # ---- route table + per-airline breakdown ----
    tab_routes, tab_airlines = st.tabs(["Routes", "Airline breakdown"])

    with tab_routes:
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
            # n_flight_legs, NOT n_flights: this frame is dash_route_drilldown
            # (route grain). n_flights is the route x CARRIER grain's name and
            # belongs to the breakdown tab below — mixing them raised a KeyError
            # that took the whole page down.
            .sort_values("n_flight_legs", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            # exact height: the default viewport pads a short result out with
            # blank rows, which reads as missing data rather than "3 matches"
            height=ui.table_height(len(table)),
            column_config={
                "route": "Route",
                "origin_city": "Origin city",
                "dest_city": "Dest city",
                "n_flight_legs": st.column_config.NumberColumn("Legs", format="%d"),
                "arr_del15_rate": st.column_config.ProgressColumn(
                    "Delay rate", format="%.1f%%", min_value=0, max_value=0.5
                ),
                "avg_arr_delay_minutes": st.column_config.NumberColumn(
                    "Avg delay", format="%.1f min"
                ),
                "p90_arr_delay_minutes": st.column_config.NumberColumn(
                    "P90 delay", format="%.0f min"
                ),
                "cancellation_rate": st.column_config.NumberColumn("Cancel rate", format="percent"),
            },
        )

        # inside the tab, not after it: at function-body indent these render
        # below the whole tab container, so the Airline breakdown tab showed a
        # "routes_filtered.csv" button and a route count belonging to the
        # other tab.
        ui.download_button(table, "routes_filtered.csv")
        st.caption(f"{len(table):,} routes · sorted by traffic.")

    with tab_airlines:
        _render_airline_breakdown(df)

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


def _render_airline_breakdown(df) -> None:
    """Who flies these routes, how much of them, and how reliably.

    Three columns, because a bare delay rate is not a fair comparison:

      SHARE   how much of the selected traffic is theirs — a 30% rate on 4% of
              the flights is a footnote; on 45% of them it IS the route.
      RATE    the raw SUM/SUM delay rate.
      VS MIX  actual delays against what the ROUTES THEY FLY would predict
              (indirect standardisation, dashboard/metrics.mix_adjusted).
              Comparing raw rates across many routes compares route mixes too:
              an airline concentrated on congested hubs looks worse than one
              flying Hawaii whatever its operations are like. 1.00x means
              "exactly as its own routes predict".
    """
    routes = set(df["route"])
    rc = data.route_carrier()
    rc = rc[rc["route"].isin(routes)]
    if rc.empty:
        st.info("No airline breakdown for the current filters.")
        return

    names = {
        r["carrier_key"]: fl.carrier_label(
            r["carrier_key"], r.get("carrier_name"), bool(r.get("is_regional"))
        )
        for _, r in data.carrier_reliability().iterrows()
    }
    adj = metrics.mix_adjusted(rc, "carrier_key", "route").sort_values("n_flights", ascending=False)
    adj["share"] = metrics.share_of(adj)

    single = len(routes) == 1
    st.caption(
        f"{len(adj)} airlines across the {len(routes):,} route(s) selected · "
        f"{int(adj['n_flights'].sum()):,} flights. Rates are SUM/SUM, never an "
        "average of per-route rates."
    )
    ui.floating_rows(
        [
            {
                "carrier": names.get(r["carrier_key"], r["carrier_key"]),
                "share": ui.pct(r["share"], 0),
                "legs": ui.count(r["n_flights"]),
                "rate": ui.pct(r["rate"]),
                "vs": "—" if single or pd.isna(r["index"]) else f"{r['index']:.2f}x",
            }
            for _, r in adj.iterrows()
        ],
        [
            ("carrier", "Airline", "key"),
            ("share", "Share of flights", "num"),
            ("legs", "Legs", "num"),
            ("rate", "Delay rate", "num"),
            ("vs", "vs its route mix", "num"),
        ],
    )
    if single:
        st.caption(
            "One route selected, so every airline flies the same route and the "
            "mix adjustment has nothing to correct for — the delay rates are "
            "already like-for-like."
        )
    else:
        worst = adj.loc[adj["index"].idxmax()] if adj["index"].notna().any() else None
        if worst is not None:
            st.caption(
                f"**vs its route mix**: 1.00x = exactly as the routes that airline flies "
                f"would predict; above 1 is worse than its routes explain. "
                f"Highest here is **{names.get(worst['carrier_key'], worst['carrier_key'])}** "
                f"at {worst['index']:.2f}x. "
                "Read it alongside share and legs — a small carrier's number moves easily."
            )
