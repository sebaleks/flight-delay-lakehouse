"""Ops page: the capacity picture for one airport-day, replayed honestly.

The demo beat this page exists for:
    "On 2024-09-13 the model expected 41 ± 9 delayed departures in ORD's
    18:00 bank. There were 44."

Mode: REPLAY, not live. The flights are HELD-OUT test rows — the model never
trained on them, and the outcome is known — so every expectation renders next
to what actually happened. That is the page's honesty device and its limit:
these rows carry observed weather (the test-set regime); live serving
substitutes forecasts and is somewhat worse. The math (Poisson-binomial
bands, schedule-linkage downstream exposure, fragile-bank screen) lives
unit-tested in dashboard/capacity.py; this module is layout, state and error
handling only.

One-day caveat, stated on the page: a single day is a single draw. Before a
specific date is used as a *demo*, the day-typicality harness must have shown
it is representative — not the first day that made the model look good.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from dashboard import capacity, charts, data, ui
from dashboard.config import predictor_url
from dashboard.predict_client import PredictorUnavailable, replay_airport_day

# The shipped run's held-out window (Jul-Dec 2024). The SERVER is the
# authority — it 404s anything outside its own split boundary — these bounds
# just keep the picker from offering dates that can only fail.
HOLDOUT_MIN = date(2024, 7, 1)
HOLDOUT_MAX = date(2024, 12, 31)
DEFAULT_DAY = date(2024, 9, 13)


@st.cache_data(ttl=3600, show_spinner=False)
def _replay(url: str, origin: str, day: date) -> dict:
    """Cached: a held-out day never changes for a given artifacts run."""
    return replay_airport_day(url, origin=origin, flight_date=day)


def _airport_options() -> list[str]:
    df = data.airport_reliability()
    return sorted(df["airport_key"].dropna().unique().tolist())


def _history_implied(origin: str, day: date, banks: list[dict]) -> pd.DataFrame:
    """What history ALONE would predict for this schedule: the bank's
    TRAINING-WINDOW same-weekday delay rate (SUM/SUM, never averaged rates)
    times the number of departures actually scheduled this day. The mart
    behind dash_airport_hour_baseline ends at the train/test cutoff, so this
    comparator never contains the held-out outcomes it is judged against —
    the comparison that shows whether the model adds anything beyond
    climatology, kept fair."""
    frame = pd.DataFrame(banks)
    try:
        base = data.airport_hour_baseline()
    except Exception:  # baseline missing (e.g. mart not built yet) — chart still renders
        return frame
    base = base[(base["airport_key"] == origin) & (base["day_of_week"] == day.isoweekday())]
    if base.empty:
        return frame
    rate = base.assign(hist_rate=base["n_arr_del15"] / base["n_with_arr_outcome"]).rename(
        columns={"dep_hour": "hour"}
    )[["hour", "hist_rate"]]
    frame = frame.merge(rate, on="hour", how="left")
    frame["baseline"] = frame["hist_rate"] * frame["n_flights"]
    return frame.drop(columns=["hist_rate"])


def render() -> None:
    st.title("Ops capacity replay")
    url = predictor_url()
    if not url:
        st.info(
            "The prediction service is not configured for this deployment "
            "(`PREDICTOR_URL` is unset), so this page is disabled. The "
            "analytics pages are unaffected."
        )
        return

    st.caption(
        "One airport, one held-out day: what the model expected bank by bank, "
        "next to what actually happened. These flights were never part of "
        "training — the outcomes are real, not fitted."
    )

    c1, c2 = st.columns([1, 1])
    airports = _airport_options()
    origin = c1.selectbox(
        "Airport", airports, index=airports.index("ORD") if "ORD" in airports else 0
    )
    day = c2.date_input(
        "Held-out date (Jul–Dec 2024)",
        value=DEFAULT_DAY,
        min_value=HOLDOUT_MIN,
        max_value=HOLDOUT_MAX,
    )

    with st.spinner(
        "Waking the model and replaying the day — the first request after a "
        "quiet period takes about 30 seconds."
    ):
        try:
            payload = _replay(url, origin, day)
        except PredictorUnavailable as exc:
            st.error(f"No replay to show. ({exc})")
            return

    flights = payload["flights"]
    summary = capacity.day_summary(flights)
    banks = capacity.hourly_banks(flights)

    lo = max(0, summary["expected"] - 2 * summary["sd"])
    hi = summary["expected"] + 2 * summary["sd"]
    within = lo <= summary["actual"] <= hi
    st.markdown(
        f"### On {day:%A %b %d, %Y} the model expected "
        f"**{summary['expected']:.0f} ± {2 * summary['sd']:.0f}** delayed departures "
        f"at {origin}. There were **{summary['actual']}**."
    )
    st.caption(
        ("Inside the model's own ±2σ band." if within else "OUTSIDE the model's ±2σ band — ")
        + (
            ""
            if within
            else "a day where something the pre-departure features don't carry took over."
        )
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scheduled departures", ui.count(summary["n_flights"]))
    m2.metric("Model expected delayed", f"{summary['expected']:.0f} ± {2 * summary['sd']:.0f}")
    m3.metric("Actually delayed", ui.count(summary["actual"]))
    m4.metric(
        "Downstream legs at stake",
        f"{summary['expected_downstream']:.0f}",
        help=(
            "Σ p × remaining scheduled legs for each aircraft — schedule "
            "linkage only. Swap-disrupted linkages count zero, so this is an "
            "honest undercount."
        ),
    )

    st.plotly_chart(
        charts.ops_banks(
            _history_implied(origin, day, banks),
            title=f"{origin} {day} — delayed departures by bank: model vs reality",
        ),
        use_container_width=True,
    )
    st.caption(
        "Band: ±2σ from the calibrated probabilities themselves (√Σp(1−p)), "
        "assuming independence within the hour — shared shocks like a storm "
        "make the real spread wider. Dotted line: what the pre-cutoff "
        "(2022 – Jun 2024) same-weekday delay rate alone would have predicted "
        "for this schedule — history that ends where the holdout begins."
    )

    st.markdown("##### Fragile banks — where a slip cascades")
    st.caption(
        "Ranked by p-weighted downstream legs (schedule linkage only). "
        f"Flagged fragile when ≥{capacity.TIGHT_SHARE_FLAG:.0%} of the bank "
        "sits on tight scheduled turnarounds."
    )
    fragile = pd.DataFrame(capacity.fragile_banks(banks))
    if not fragile.empty:
        show = fragile.assign(
            bank=fragile["hour"].map(lambda h: f"{h:02d}:00"),
            expected=fragile["expected"].round(1),
            tight_share=fragile["tight_share"].map(ui.pct),
            expected_downstream=fragile["expected_downstream"].round(1),
        )[
            [
                "bank",
                "n_flights",
                "expected",
                "actual",
                "tight_share",
                "expected_downstream",
                "fragile",
            ]
        ].rename(
            columns={
                "n_flights": "departures",
                "expected": "expected delayed",
                "actual": "actually delayed",
                "tight_share": "tight turnarounds",
                "expected_downstream": "downstream legs at stake",
            }
        )
        st.dataframe(show, hide_index=True, use_container_width=True)

    st.markdown("##### Proactive comms — who to call first")
    st.caption(
        "Highest delay probability first. The 'actually' column is the "
        "replay's answer sheet — on a live day it would not exist."
    )
    comms = pd.DataFrame(capacity.comms_ranking(flights, top_n=10))
    if not comms.empty:
        show = comms.assign(
            delay_probability=comms["delay_probability"].map(ui.pct),
            actual_delayed=comms["actual_delayed"].map(
                lambda v: "was delayed" if v else "was on time"
            ),
        ).rename(
            columns={
                "dep_time": "departs",
                "delay_probability": "P(delay)",
                "actual_delayed": "actually",
                "remaining_legs": "legs after this one",
            }
        )
        st.dataframe(show, hide_index=True, use_container_width=True)

    with st.expander("What this replay does and does not claim"):
        st.markdown(
            f"""
- **Held-out, provably.** The server refuses any training-window date
  (artifacts run `{payload.get("artifacts", "?")}`); every row here is from the
  test window the headline metrics were measured on.
- **Observed weather.** Replay rows carry the weather that was *observed* at
  scheduled departure — the test-set regime. Live serving substitutes
  forecasts for the same hour and is somewhat worse.
- **One day is one draw.** A single day inside (or outside) the band proves
  little by itself; the day-typicality check across the whole held-out window
  is what says whether a chosen demo date is representative.
- **Downstream exposure is schedule linkage only.** Rotation restructured by
  day-of tail swaps is a day-of outcome, so those aircraft count **zero**
  downstream legs here — an honest undercount, never backfilled.
"""
        )
