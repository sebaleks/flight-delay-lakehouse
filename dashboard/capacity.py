"""Pure aggregation math for the ops-capacity page — unit-tested, no
Streamlit, no HTTP. Input everywhere is the predictor's /replay/airport-day
per-flight rows: delay_probability, dep_hour, actual_delayed,
rotation_position, legs_today, is_tight_turnaround.

THE BAND. With calibrated per-flight probabilities p_i, the count of delayed
departures in a bank is a sum of independent-ish Bernoullis: expectation Σp_i,
variance Σp_i(1−p_i) — the Poisson-binomial moments, exactly what the model's
own probabilities imply. Independence is the assumption doing the work: a
storm delays flights TOGETHER, so the true variance is larger under shared
shocks. The page says so rather than pretending the band is a guarantee.

DOWNSTREAM EXPOSURE — schedule linkage ONLY. remaining_legs counts the legs
this aircraft is *scheduled* to fly after this one (legs_today −
rotation_position), which is knowable at planning time. The leakage rule
extends to linkage (CLAUDE.md §9): swap-shaped rows carry NULL rotation
columns and contribute ZERO here — an honest undercount, stated on the page,
never backfilled from day-of operational data.
"""

from __future__ import annotations

from math import sqrt

TIGHT_SHARE_FLAG = 0.25  # a quarter of the bank on tight turnarounds reads as fragile


def remaining_legs(flight: dict) -> int:
    """Scheduled legs this aircraft still has to fly today AFTER this one.
    Zero when the linkage is absent (swap-shaped or unknown) — never guessed."""
    legs, pos = flight.get("legs_today"), flight.get("rotation_position")
    if legs is None or pos is None:
        return 0
    return max(0, int(legs) - int(pos))


def day_summary(flights: list[dict]) -> dict:
    """The headline: expected delayed departures Σp with the Poisson-binomial
    sd, next to what actually happened, plus the p-weighted downstream stake."""
    p = [float(fl["delay_probability"]) for fl in flights]
    return {
        "n_flights": len(flights),
        "expected": sum(p),
        "sd": sqrt(sum(x * (1 - x) for x in p)),
        "actual": sum(1 for fl in flights if fl["actual_delayed"]),
        "expected_downstream": sum(
            float(fl["delay_probability"]) * remaining_legs(fl) for fl in flights
        ),
    }


def hourly_banks(flights: list[dict]) -> list[dict]:
    """Per departure-hour bank stats, sorted by hour.

    tight_share treats an absent flag (swap-shaped linkage) as not-tight — the
    same honest-undercount convention as remaining_legs. expected_downstream
    is Σ p·remaining_legs: the p-weighted count of scheduled follow-on legs
    riding on this bank, the number the fragile-bank screen ranks by.
    """
    banks: dict[int, list[dict]] = {}
    for fl in flights:
        banks.setdefault(int(fl["dep_hour"]), []).append(fl)
    out = []
    for hour in sorted(banks):
        group = banks[hour]
        p = [float(fl["delay_probability"]) for fl in group]
        n_tight = sum(1 for fl in group if fl.get("is_tight_turnaround"))
        out.append(
            {
                "hour": hour,
                "n_flights": len(group),
                "expected": sum(p),
                "sd": sqrt(sum(x * (1 - x) for x in p)),
                "actual": sum(1 for fl in group if fl["actual_delayed"]),
                "n_tight": n_tight,
                "tight_share": n_tight / len(group),
                "downstream_legs": sum(remaining_legs(fl) for fl in group),
                "expected_downstream": sum(
                    float(fl["delay_probability"]) * remaining_legs(fl) for fl in group
                ),
            }
        )
    return out


def fragile_banks(banks: list[dict]) -> list[dict]:
    """The screening order: banks ranked by what is AT STAKE downstream
    (expected_downstream, then tight_share as the tiebreak). A bank is flagged
    `fragile` when at least TIGHT_SHARE_FLAG of its departures sit on tight
    scheduled turnarounds — the shape where one late arrival cascades."""
    ranked = sorted(banks, key=lambda b: (-b["expected_downstream"], -b["tight_share"]))
    return [{**b, "fragile": b["tight_share"] >= TIGHT_SHARE_FLAG} for b in ranked]


def comms_ranking(flights: list[dict], top_n: int = 10) -> list[dict]:
    """Who to proactively contact, highest delay probability first (dep_time
    breaks ties deterministically). Carries the flight's own downstream count
    so the desk can also see the network stake behind each call."""
    ranked = sorted(
        flights,
        key=lambda fl: (-float(fl["delay_probability"]), str(fl.get("dep_time", ""))),
    )
    return [
        {
            "dep_time": fl.get("dep_time"),
            "flight": f"{fl.get('carrier', '')} {fl.get('flight_number', '')}".strip(),
            "dest": fl.get("dest"),
            "delay_probability": float(fl["delay_probability"]),
            "actual_delayed": bool(fl["actual_delayed"]),
            "remaining_legs": remaining_legs(fl),
        }
        for fl in ranked[:top_n]
    ]
