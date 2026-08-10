"""How a calibrated probability is SPOKEN to a person.

Pure functions — no Streamlit, no BigQuery, no network — so the wording rules
are unit-testable on their own (dashboard/test_uncertainty.py).

THE PROBLEM THIS SOLVES. The model outputs 0.3273. Rendering that as "32.73%"
implies a resolution it does not have, and rendering it alone reads as a verdict
about YOUR flight. It is neither. It is a rate for flights that look like yours,
carrying roughly +/-2 points of slack, and the honest presentation says all of
that without burying the number.

The rules, each of which exists because the obvious alternative misleads:

  * NATURAL FREQUENCY FIRST. "About 3 in 10" is understood correctly by far more
    people than "31%", and it carries its own coarseness — nobody reads "3 in
    10" as precise to the percentage point.
  * SHOW THE COMPLEMENT. A lone "31% delayed" reads as a prediction of lateness.
    "About 3 in 10 late, 7 in 10 on time" reads as a distribution, which is what
    it is.
  * ANCHOR TO THE BASE RATE. 31% means nothing without knowing that ~20% of all
    US domestic flights arrive late. The comparison is the information.
  * NEVER A VERDICT. No "DELAYED"/"ON TIME" badge, no gauge with a needle in a
    red zone. Those are decisions wearing a probability costume.
"""

from __future__ import annotations

from dataclasses import dataclass

# Held-out base rate: the fraction of all US domestic flights 2022-2024 that
# arrived 15+ minutes late. The API reports it on /calibration; this is the
# fallback for when that endpoint is unavailable, and the number every
# comparison is made against.
BASE_RATE = 0.1969


@dataclass(frozen=True)
class Phrasing:
    """Everything a page needs to render one probability honestly."""

    headline: str  # "About 3 in 10 flights like this one arrive 15+ minutes late."
    complement: str  # "About 7 in 10 arrive on time."
    precise: str  # "Modelled probability 31%."
    comparison: str  # how it sits against the base rate
    caveat: str  # the standing "this is not about your flight" line


def natural_frequency(p: float, denominator: int = 10) -> tuple[int, int]:
    """p -> (k, denominator) such that k/denominator is the nearest simple ratio.

    Deliberately coarse. The model's held-out ECE is 0.017, i.e. its
    probabilities are trustworthy to roughly +/-2 points; a "37 in 100" would
    imply a precision the calibration does not support. Rounding to tenths keeps
    the claim inside what the evidence backs.

    Never rounds a non-zero risk down to 0 in 10, or a non-certainty up to
    10 in 10 — "0 in 10" reads as "cannot happen", which is a promise no model
    should make. p == 0.0 and p == 1.0 exactly are passed through, since those
    are the caller asserting certainty rather than the model estimating it.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability out of range: {p}")
    k = round(p * denominator)
    if p > 0.0:
        k = max(k, 1)
    if p < 1.0:
        k = min(k, denominator - 1)
    return int(k), denominator


def vs_base_rate(p: float, base: float = BASE_RATE) -> str:
    """One clause placing p against the population rate.

    Banded, not a ratio: "1.6x more likely than average" invites multiplying it
    by something. The bands are deliberately wide because the difference between
    0.24 and 0.26 is not something a traveller should act on differently.
    """
    if base <= 0:
        raise ValueError("base rate must be positive")
    ratio = p / base
    if ratio < 0.6:
        return "well below the typical US domestic flight"
    if ratio < 0.85:
        return "somewhat below the typical US domestic flight"
    if ratio <= 1.15:
        return "about the same as the typical US domestic flight"
    if ratio <= 1.6:
        return "somewhat above the typical US domestic flight"
    if ratio <= 2.5:
        return "well above the typical US domestic flight"
    return "far above the typical US domestic flight"


def phrase(p: float, base: float = BASE_RATE) -> Phrasing:
    """The full spoken form of one probability."""
    k, d = natural_frequency(p)
    return Phrasing(
        headline=f"About **{k} in {d}** flights like this one arrive 15+ minutes late.",
        complement=f"About **{d - k} in {d}** arrive on time.",
        precise=f"Modelled probability {p:.0%}.",
        comparison=f"That is {vs_base_rate(p, base)}, where about "
        f"{natural_frequency(base)[0]} in 10 arrive late.",
        caveat=(
            "This is a rate for flights **like** yours — same route, carrier, "
            "time of day, forecast — not a forecast about your specific flight. "
            "We cannot tell you whether *your* flight will be late."
        ),
    )


def basis_notes(basis: dict) -> list[tuple[str, str]]:
    """Plain-English notes for the prediction_basis block, as (level, text).

    level is "error" | "warning" | "info", so the page can style without
    re-deriving meaning. Ordered most to least severe.

    Every one of these exists because the number ALONE would mislead: a
    weatherless estimate looks identical to a weather-informed one, and a
    past-date score looks identical to a forecast.
    """
    notes: list[tuple[str, str]] = []
    horizon = basis.get("weather_horizon")

    if basis.get("flight_in_past"):
        notes.append(
            (
                "error",
                "This flight has already departed. The model scores it for "
                "debugging, but it is **not** a pre-departure prediction and "
                "must not be read as one.",
            )
        )
    elif horizon == "beyond_horizon":
        notes.append(
            (
                "warning",
                "Your flight is more than about a week out, so no weather "
                "forecast exists for it yet. All 12 weather inputs are missing "
                "and this estimate uses schedule, route history and a typical "
                "aircraft rotation only — **check back within 7 days**, when "
                "the forecast lands and the estimate sharpens.",
            )
        )
    elif horizon == "unavailable":
        notes.append(
            (
                "warning",
                "We could not place this airport's local time, so no weather "
                "was used. The estimate rests on schedule and history alone.",
            )
        )

    if basis.get("rotation_context") == "typical_estimate":
        notes.append(
            (
                "info",
                "We do not know which physical aircraft flies this route, or "
                "what it did earlier that day, so we assumed a **typical** "
                "rotation. What the aircraft did earlier is one of the model's "
                "strongest signals, so an airline's own estimate — which knows "
                "the real rotation — would be sharper than this one.",
            )
        )
    if basis.get("origin_density_source") == "estimated":
        notes.append(
            (
                "info",
                "Departure traffic at that hour is estimated from history "
                "rather than a published schedule for the day.",
            )
        )
    return notes
