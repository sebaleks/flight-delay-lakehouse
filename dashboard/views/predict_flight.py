"""Consumer page: "Will my flight be late?"

PICK, don't type. Choose a departure airport and date, and the page loads that
airport's departure board; optional filters (airline, destination, flight
number, departure window) narrow ~900 flights down to yours. Typing every field
by hand was cumbersome and, worse, produced a WEAKER estimate: a picked flight
carries its aircraft-rotation context, so it scores with
rotation_context="provided" instead of the typical-profile fallback.

CONNECTIONS. Tick "I have a connecting flight" and the same picker opens for
the second leg. The page then answers the question a traveller actually has —
not "will both be late", but "am I going to miss it". See
dashboard/flights.connection_risk for why those are different and why the two
probabilities are never multiplied.

The page's job is to give a calibrated probability WITHOUT it reading as a
verdict. Deliberately absent: a gauge (a needle in a red zone is a decision
wearing a probability costume), a DELAYED/ON-TIME badge, an "expected delay
+23 min" point estimate (held-out MAE 18.99 / RMSE 49.26 — the error bar dwarfs
the number), and the uncalibrated logreg baseline. Wording rules live in
dashboard/uncertainty.py; the picker and connection math in dashboard/flights.py.

Mode: LIVE forecast, PROXY schedule. Real NDFD weather for the chosen date; the
departure board is a historical same-weekday board standing in for it, because
no future airline schedule feed exists here. The page says so.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from dashboard import charts, data, ui, uncertainty
from dashboard import flights as fl
from dashboard.config import predictor_url
from dashboard.predict_client import (
    PredictorUnavailable,
    calibration,
    outcome_mix,
    predict_selected,
    schedule_airport_day,
)

FORECAST_DAYS = 7
MAX_DAYS_AHEAD = 330
# How many flights the list will render at once. Each row is five Streamlit
# widgets, so this is the render-cost ceiling rather than a readability one —
# 100 keeps a typical filtered board (one airline out of a hub) complete, which
# is the case that matters: being shown 40 of 87 matches means scrolling to a
# flight that is not there.
MAX_LISTED = 100


@st.cache_data(ttl=3600, show_spinner=False)
def _evidence(url: str) -> tuple[dict, dict]:
    return calibration(url), outcome_mix(url)


@st.cache_data(ttl=3600, show_spinner=False)
def _board(url: str, origin: str, day: date) -> dict:
    """One airport-day departure board. Cached — the proxy board is stable."""
    return schedule_airport_day(url, origin=origin, flight_date=day)


@st.cache_data(ttl=3600, show_spinner=False)
def _carrier_names() -> dict[str, str]:
    """code -> 'United Airlines (UA)', from the already-cached gold view."""
    df = data.carrier_reliability()
    return {
        r["carrier_key"]: fl.carrier_label(
            r["carrier_key"], r.get("carrier_name"), bool(r.get("is_regional"))
        )
        for _, r in df.iterrows()
    }


@st.cache_data(ttl=3600, show_spinner=False)
def _airport_names() -> dict[str, str]:
    """code -> 'Full Name (CODE)', from the already-cached gold view."""
    df = data.airport_reliability()
    return {
        r["airport_key"]: fl.airport_label(r["airport_key"], r.get("airport_name"), r.get("city"))
        for _, r in df.iterrows()
    }


def _band_of(p: float, bins: list[dict]) -> dict | None:
    for b in bins:
        if b["lo"] <= p < b["hi"] or (b["hi"] == 1.0 and p == 1.0):
            return b
    return None


def _pick_flight(url: str, key: str, title: str, origin_default: str, day: date) -> dict | None:
    """Airport -> board -> filters -> one selected flight. None until chosen."""
    names = _airport_names()
    codes = sorted(names)
    st.markdown(f"##### {title}")
    origin = st.selectbox(
        "Departing from",
        codes,
        index=codes.index(origin_default) if origin_default in codes else 0,
        format_func=lambda c: names.get(c, c),
        key=f"{key}_origin",
    )
    try:
        board = _board(url, origin, day)
    except PredictorUnavailable as exc:
        st.error(f"Could not load the departure board. ({exc})")
        return None

    rows = board["flights"]
    carriers = sorted({f["carrier"] for f in rows})
    dests = sorted({f["dest"] for f in rows})

    c1, c2, c3 = st.columns([1, 2, 1])
    cnames = _carrier_names()
    carrier = c1.selectbox(
        "Airline",
        ["Any", *carriers],
        format_func=lambda c: "Any" if c == "Any" else cnames.get(c, c),
        key=f"{key}_carrier",
    )
    dest = c2.selectbox(
        "Going to",
        ["Any", *dests],
        format_func=lambda c: "Any" if c == "Any" else names.get(c, c),
        key=f"{key}_dest",
    )
    number = c3.text_input("Flight no.", key=f"{key}_no", placeholder="e.g. 2842")
    early, late = st.select_slider(
        "Departure window",
        options=[f"{h:02d}:00" for h in range(25)],
        value=("00:00", "24:00"),
        key=f"{key}_win",
    )

    shown = fl.filter_flights(
        rows,
        carrier=None if carrier == "Any" else carrier,
        dest=None if dest == "Any" else dest,
        flight_number=number or None,
        dep_from=early,
        dep_to=late,
    )
    if not shown:
        st.warning("No flights match those filters. Try clearing one.")
        return None

    picked_key = f"{key}_picked"
    chosen = st.session_state.get(picked_key)
    chosen_visible = chosen is not None and any(
        c["carrier"] == chosen["carrier"] and c["flight_number"] == chosen["flight_number"]
        for c in shown  # the full filtered set, not just the listed page
    )

    # COLLAPSED: once a flight is chosen the list has done its job, so it folds
    # down to the one selection. Leaving 100 rows on screen buries the thing the
    # page is now about, and on a connecting itinerary it buried leg 2's picker
    # under leg 1's board.
    if chosen_visible:
        ui.inject_row_css()
        st.markdown(
            f'<div class="fdl-row" style="background:rgba(46,134,171,.10);">'
            f'<div class="fdl-cell fdl-key"><strong>{chosen["carrier"]} '
            f"{chosen['flight_number']}</strong></div>"
            f'<div class="fdl-cell">{names.get(chosen["dest"], chosen["dest"])}</div>'
            f'<div class="fdl-cell">departs {chosen["dep_time"]}</div>'
            f'<div class="fdl-cell">arrives {chosen["arr_time"]}</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Change flight", key=f"{key}_change", type="secondary"):
            del st.session_state[picked_key]
            st.rerun()
        return chosen

    # One honest caption: say whether the list is complete, and only ask for a
    # narrower filter when it actually is truncated.
    if len(shown) > MAX_LISTED:
        st.caption(
            f"**{len(shown)}** of {board['n_flights']} departures match — showing the "
            f"first {MAX_LISTED}. Add an airline, a destination or a departure window "
            "to see the rest."
        )
    else:
        st.caption(f"**{len(shown)}** of {board['n_flights']} departures match.")
    # ALWAYS show a list. Hiding it until the filters are narrow enough left
    # the page with nothing to select from on arrival, which is the opposite of
    # the point — the list IS the interface, the filters just shorten it.
    listed = shown[:MAX_LISTED]

    # A LIST, not a spreadsheet: no grid, no empty filler. Each row carries its
    # own select button, so picking is one click rather than hunting a value in
    # a dropdown.
    ui.inject_row_css()
    st.markdown(
        '<div class="fdl-head"><div class="fdl-cell fdl-key">Flight</div>'
        '<div class="fdl-cell">To</div><div class="fdl-cell">Departs</div>'
        '<div class="fdl-cell">Arrives</div><div class="fdl-cell fdl-num"></div></div>',
        unsafe_allow_html=True,
    )
    for i, f in enumerate(listed):
        c1, c2, c3, c4, c5 = st.columns([1.1, 2.4, 1, 1, 0.9], vertical_alignment="center")
        c1.markdown(f"**{f['carrier']} {f['flight_number']}**")
        c2.markdown(names.get(f["dest"], f["dest"]))
        c3.markdown(f["dep_time"])
        c4.markdown(f["arr_time"])
        if c5.button("Select", key=f"{key}_pick_{i}", use_container_width=True):
            st.session_state[picked_key] = f
            # rerun immediately so the list collapses in the same interaction
            st.rerun()
    return None


def _render_notes(basis: dict) -> None:
    for level, text in uncertainty.basis_notes(basis):
        {"error": st.error, "warning": st.warning, "info": st.info}[level](text)


def _render_one(pred, base: float, mix: dict, cal: dict, heading: str | None = None) -> None:
    ph = uncertainty.phrase(pred.delay_probability, base)
    if heading:
        st.markdown(f"#### {heading}")
    st.markdown(f"### {ph.headline}")
    st.markdown(f"**{ph.complement}**  \n{ph.precise} {ph.comparison}")
    st.plotly_chart(
        charts.probability_vs_base_rate(
            pred.delay_probability, base_rate=base, weather_known=pred.has_origin_weather
        ),
        use_container_width=True,
    )
    st.caption(ph.caveat)
    _render_notes(pred.basis)
    _render_outcome_mix(pred.delay_probability, mix)


def _render_outcome_mix(p: float, mix: dict) -> None:
    band = _band_of(p, mix.get("bins", []))
    if not band or not band.get("n"):
        return
    e = band["exceedance"]
    st.markdown("##### If it *is* late, how late?")
    st.caption(
        f"Among **{band['n']:,}** held-out flights scored between "
        f"{band['lo']:.0%} and {band['hi']:.0%} — flights the model never trained on."
    )
    rows = [
        ("Within 15 minutes", 1 - e["15"]),
        ("15 min – 1 hour", e["15"] - e["60"]),
        ("1 – 2 hours", e["60"] - e["120"]),
        ("Over 2 hours", e["120"]),
    ]
    cols = st.columns(len(rows))
    for col, (label, frac) in zip(cols, rows, strict=True):
        col.metric(label, ui.pct(frac))


def _render_connection(leg1, leg2, f1: dict, f2: dict, mix: dict) -> None:
    risk = fl.connection_risk(
        f1["arr_time"], f2["dep_time"], mix.get("bins", []), leg1.delay_probability
    )
    st.markdown("### Will you make the connection?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Scheduled layover", f"{risk.layover_min} min")
    c2.metric(
        "Slack after changing planes",
        f"{risk.slack_min} min",
        help=f"Assumes {fl.DEFAULT_MCT_MIN} minutes to get between gates.",
    )
    if risk.probability is None:
        c3.metric("Risk of misconnecting", "—")
        st.error(risk.note)
        return
    if risk.upper_bound:
        # long layover, past what we measured: a bound, not an estimate
        c3.metric("Risk of misconnecting", f"under {ui.pct(risk.probability)}")
        st.markdown(
            f"**You have {risk.slack_min} minutes of slack** — more than the "
            f"{risk.threshold_min} minutes we can measure. The risk is **below "
            f"{ui.pct(risk.probability)}**, and in practice well below: this is the "
            "chance of a delay big enough to eat the longest gap we have data for, "
            "not the chance of missing *this* connection."
        )
    else:
        k, d = uncertainty.natural_frequency(risk.probability)
        c3.metric("Risk of misconnecting", ui.pct(risk.probability))
        st.markdown(
            f"**About {k} in {d}** itineraries like this one miss the connection — that "
            f"is how often the first leg arrived late enough to eat the slack, measured "
            f"on held-out flights."
        )
    st.caption(
        risk.note + ". This is **not** the two probabilities multiplied: the question is whether "
        "leg 1's arrival delay exceeds your slack, and multiplying would also assume the "
        "legs are independent — they share weather and the same airline's operations, so "
        "they are positively correlated."
    )


def _render_calibration(p: float, cal: dict) -> None:
    bands = cal.get("reliability")
    if not bands:
        return
    with st.expander("Should you believe this number? — the held-out evidence"):
        st.markdown(
            f"On **{cal['n_test']:,}** flights the model had never seen "
            f"(from {cal['test_start']}), we compared what it said against what happened. "
            "Bars on the dotted line mean the probabilities are honest."
        )
        band = _band_of(p, bands)
        st.plotly_chart(
            charts.reliability(pd.DataFrame(bands), highlight_lo=band["lo"] if band else None),
            use_container_width=True,
        )
        st.caption(
            f"Your flight's band is highlighted. Expected calibration error {cal['ece']:.3f}."
        )


def _render_model_card(cal: dict) -> None:
    with st.expander("About this model"):
        st.markdown(
            f"""
- **Held-out performance** ({cal.get("n_test", 0):,} flights the model never saw,
  Jul–Dec 2024): ROC-AUC **{cal.get("roc_auc", float("nan")):.4f}**, PR-AUC
  **{cal.get("pr_auc", float("nan")):.4f}** against a base rate of
  **{cal.get("base_rate", float("nan")):.4f}**.
- **Probabilities are calibrated** (`{cal.get("calibration_method")}`), so "30%"
  means about 30 in 100.
- **Only pre-departure information is used** — no realised departure delay, no
  actual times, nothing about how the aircraft's earlier legs actually went.
- **The schedule is a proxy.** No future airline schedule feed exists here, so
  the departure board is a historical same-weekday board for the airport. The
  weather forecast is real and live.
- **One honest gap:** the numbers above were measured against *observed*
  weather; live scoring substitutes a *forecast*, so real performance is
  somewhat worse.
- **We don't store your search.**
"""
        )


def render() -> None:
    st.title("Will my flight be late?")
    url = predictor_url()
    if not url:
        st.info(
            "The prediction service is not configured for this deployment "
            "(`PREDICTOR_URL` is unset), so this page is disabled. The analytics "
            "pages are unaffected."
        )
        return

    st.caption(
        "Pick your flight from the departure board and get a calibrated "
        "probability, from a model that uses only what is knowable before departure."
    )

    today = date.today()
    day = st.date_input(
        "Travel date",
        value=today + timedelta(days=2),
        min_value=today,
        max_value=today + timedelta(days=MAX_DAYS_AHEAD),
    )

    leg1 = _pick_flight(url, "l1", "Your flight", "ORD", day)
    connecting = st.checkbox("I have a connecting flight")
    leg2 = None
    if connecting and leg1:
        st.divider()
        leg2 = _pick_flight(url, "l2", "Connecting flight", leg1["dest"], day)
        if leg2 and leg2["origin"] != leg1["dest"]:
            st.warning(
                f"Your first flight lands at {leg1['dest']} but the connection departs "
                f"{leg2['origin']}. Change the connecting airport to {leg1['dest']}."
            )
            leg2 = None

    if not leg1 or (connecting and not leg2):
        st.caption("Choose a flight above to see its estimate.")
        return
    if not st.button("Estimate", type="primary"):
        return

    with st.spinner(
        "Waking the model and fetching the forecast — the first request after a quiet "
        "period takes about 30 seconds, because nothing is left running when nobody is "
        "using it."
    ):
        try:
            p1 = predict_selected(url, leg1, day)
            p2 = predict_selected(url, leg2, day) if leg2 else None
            cal, mix = _evidence(url)
        except PredictorUnavailable as exc:
            st.error(f"The prediction service is unavailable right now. ({exc})")
            return

    if p1.flight_in_past or (p2 and p2.flight_in_past):
        _render_notes(p1.basis)
        return

    base = cal.get("base_rate") or uncertainty.BASE_RATE
    st.divider()
    if p2:
        _render_connection(p1, p2, leg1, leg2, mix)
        st.divider()
        _render_one(p1, base, mix, cal, heading=f"Leg 1 — {leg1['origin']} → {leg1['dest']}")
        st.divider()
        _render_one(p2, base, mix, cal, heading=f"Leg 2 — {leg2['origin']} → {leg2['dest']}")
    else:
        _render_one(p1, base, mix, cal)

    _render_calibration(p1.delay_probability, cal)
    _render_model_card(cal)
