"""Consumer page: "Will my flight be late?"

The page's job is to give a person a calibrated probability WITHOUT it reading
as a verdict, and to show the evidence rather than ask for trust. The wording
rules live in dashboard/uncertainty.py (unit-tested); this module is layout,
state and error handling.

What is deliberately absent, and why:
  * no gauge / speedometer — a needle in a red zone is a decision, not a number
  * no DELAYED / ON TIME badge — the model does not know that
  * no "expected delay: +23 min" — held-out MAE 18.99 / RMSE 49.26, so the
    error bar dwarfs the number. The outcome-mix table answers "how late?"
  * no logreg_baseline_probability — uncalibrated; predict_client drops it

Mode: LIVE. Real NDFD forecast, no ground truth. Past dates are blocked in the
picker AND refused on the server's own say-so (prediction_basis.flight_in_past),
because a past-date score looks exactly like a forecast.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from dashboard import charts, data, ui, uncertainty
from dashboard.config import predictor_url
from dashboard.predict_client import (
    PredictorUnavailable,
    calibration,
    outcome_mix,
    predict_one,
)

# NDFD publishes ~7 days out. Beyond that a request still works but every
# weather feature is missing, so the picker allows it and the page says so.
FORECAST_DAYS = 7
MAX_DAYS_AHEAD = 330


@st.cache_data(ttl=3600, show_spinner=False)
def _evidence(url: str) -> tuple[dict, dict]:
    """The two held-out evidence tables. Cached — they change only on redeploy."""
    return calibration(url), outcome_mix(url)


def _airport_options() -> list[str]:
    """Reuse the already-cached gold view — no new query for the pickers."""
    df = data.airport_reliability()
    return sorted(df["airport_key"].dropna().unique().tolist())


def _carrier_options() -> list[str]:
    df = data.carrier_reliability()
    return sorted(df["carrier_key"].dropna().unique().tolist())


def _band_of(p: float, bins: list[dict]) -> dict | None:
    for b in bins:
        if b["lo"] <= p < b["hi"] or (b["hi"] == 1.0 and p == 1.0):
            return b
    return None


def _render_notes(basis: dict) -> None:
    for level, text in uncertainty.basis_notes(basis):
        {"error": st.error, "warning": st.warning, "info": st.info}[level](text)


def _render_outcome_mix(p: float, mix: dict) -> None:
    band = _band_of(p, mix.get("bins", []))
    if not band or not band.get("n"):
        return
    exc = band["exceedance"]
    st.markdown("##### If it *is* late, how late?")
    st.caption(
        f"Among **{band['n']:,}** held-out flights we scored between "
        f"{band['lo']:.0%} and {band['hi']:.0%} — flights the model never trained on."
    )
    rows = [
        ("Arrived within 15 minutes", 1 - exc["15"]),
        ("15 minutes to an hour late", exc["15"] - exc["60"]),
        ("1 to 2 hours late", exc["60"] - exc["120"]),
        ("More than 2 hours late", exc["120"]),
    ]
    cols = st.columns(len(rows))
    for col, (label, frac) in zip(cols, rows, strict=True):
        col.metric(label, ui.pct(frac))
    st.caption(
        "These are what actually happened to similar flights — not a prediction "
        "of how late *your* flight will be."
    )


def _render_calibration(p: float, cal: dict) -> None:
    bands = cal.get("reliability")
    if not bands:
        return
    with st.expander("Should you believe this number? — the held-out evidence", expanded=False):
        st.markdown(
            f"On **{cal['n_test']:,}** flights the model had never seen "
            f"(from {cal['test_start']}), we compared what it said against what "
            "happened. Bars on the dotted line mean the probabilities are honest."
        )
        df = pd.DataFrame(bands)
        band = _band_of(p, [{**b, "n": b["n"]} for b in bands])
        fig = charts.reliability(df, highlight_lo=band["lo"] if band else None)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Your flight's band is highlighted. Expected calibration error "
            f"{cal['ece']:.3f} — on average the stated probability is within "
            f"{cal['ece']:.1%} of the real frequency."
        )


def _render_model_card(cal: dict) -> None:
    with st.expander("About this model"):
        st.markdown(
            f"""
- **Held-out performance** (Jul–Dec 2024, {cal.get("n_test", 0):,} flights the
  model never saw): ROC-AUC **{cal.get("roc_auc", float("nan")):.4f}**, PR-AUC
  **{cal.get("pr_auc", float("nan")):.4f}** against a base rate of
  **{cal.get("base_rate", float("nan")):.4f}** — about **2.4x** better than
  guessing at the population rate.
- **Probabilities are calibrated** (`{cal.get("calibration_method")}`), so "30%"
  means about 30 in 100, not merely "riskier than average".
- **Only pre-departure information is used.** No realised departure delay, no
  actual times, nothing about how the aircraft's earlier legs actually went —
  only what is knowable before your flight leaves.
- **Trained through June 2024**, evaluated on July–December 2024.
- **One honest gap:** those numbers were measured against *observed* weather.
  Live scoring substitutes a *forecast* for the same hour, so real-world
  performance is somewhat worse than the figures above.
- **We don't store your search.** The app has no write path of any kind.
"""
        )


def render() -> None:
    st.title("Will my flight be late?")
    url = predictor_url()
    if not url:
        st.info(
            "The prediction service is not configured for this deployment "
            "(`PREDICTOR_URL` is unset), so this page is disabled. The "
            "analytics pages are unaffected."
        )
        return

    st.caption(
        "A calibrated probability for one flight, from a model that uses only "
        "information knowable before departure."
    )

    with st.form("flight"):
        c1, c2, c3 = st.columns(3)
        airports = _airport_options()
        origin = c1.selectbox(
            "From", airports, index=airports.index("ORD") if "ORD" in airports else 0
        )
        dest = c2.selectbox("To", airports, index=airports.index("LAX") if "LAX" in airports else 1)
        carriers = _carrier_options()
        carrier = c3.selectbox(
            "Airline", carriers, index=carriers.index("UA") if "UA" in carriers else 0
        )
        c4, c5, c6 = st.columns(3)
        today = date.today()
        flight_date = c4.date_input(
            "Date",
            value=today + timedelta(days=2),
            # past dates are not selectable: a past-date score looks exactly
            # like a forecast, and the server flags but still returns one
            min_value=today,
            max_value=today + timedelta(days=MAX_DAYS_AHEAD),
        )
        dep_time = c5.time_input("Scheduled departure", value=None, step=300)
        arr_time = c6.time_input("Scheduled arrival", value=None, step=300)
        submitted = st.form_submit_button("Estimate", type="primary")

    if not submitted:
        st.caption(
            "Enter a flight above. Everything the estimate uses — schedule, route "
            "history, the weather forecast for your departure hour — is public "
            "information available before departure."
        )
        return
    if origin == dest:
        st.error("Origin and destination must differ.")
        return
    if dep_time is None or arr_time is None:
        st.error("Please give both scheduled times, as printed on your ticket.")
        return

    days_out = (flight_date - date.today()).days
    with st.spinner(
        "Waking the model and fetching the forecast — the first request after a "
        "quiet period takes about 30 seconds, because nothing is left running "
        "when nobody is using it."
    ):
        try:
            pred = predict_one(
                url,
                origin=origin,
                dest=dest,
                carrier=carrier,
                flight_date=flight_date,
                dep_time=dep_time.strftime("%H:%M"),
                arr_time=arr_time.strftime("%H:%M"),
            )
            cal, mix = _evidence(url)
        except PredictorUnavailable as exc:
            st.error(
                f"The prediction service is unavailable right now, so there is no "
                f"estimate to show. ({exc})"
            )
            return

    # the server's own verdict wins over the picker: never render a past-date
    # score as if it were a forecast
    if pred.flight_in_past:
        _render_notes(pred.basis)
        return

    base = cal.get("base_rate") or uncertainty.BASE_RATE
    ph = uncertainty.phrase(pred.delay_probability, base)

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
    if days_out > FORECAST_DAYS and pred.has_origin_weather:
        st.caption(f"Forecast weather was available even {days_out} days out.")

    _render_outcome_mix(pred.delay_probability, mix)
    _render_calibration(pred.delay_probability, cal)
    _render_model_card(cal)
