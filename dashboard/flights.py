"""Flight-picker helpers: labels, progressive filtering, connection risk.

Pure functions — no Streamlit, no network — so the selection logic and
especially the misconnect arithmetic are unit-testable
(dashboard/test_flights.py).
"""

from __future__ import annotations

from dataclasses import dataclass

# Minutes a traveller needs on the ground to actually change planes. US
# domestic minimum connect times run ~30-45 min at large hubs; 30 is the
# optimistic end, which makes the risk estimate CONSERVATIVE in the direction
# that matters (it does not tell you a tight connection is safe).
DEFAULT_MCT_MIN = 30


def airport_label(code: str, name: str | None = None, city: str | None = None) -> str:
    """'San Francisco International Airport (SFO)'.

    Falls back to the bare code, then to 'City (CODE)', so a missing name never
    renders an empty option. The CODE always appears — it is what a traveller
    reads off a boarding pass.
    """
    if name and str(name).strip():
        return f"{str(name).strip()} ({code})"
    if city and str(city).strip():
        return f"{str(city).strip()} ({code})"
    return code


def code_from_label(label: str) -> str:
    """'San Francisco International Airport (SFO)' -> 'SFO'."""
    if label.endswith(")") and "(" in label:
        return label.rsplit("(", 1)[1].rstrip(")").strip()
    return label.strip()


def filter_flights(
    flights: list[dict],
    *,
    carrier: str | None = None,
    dest: str | None = None,
    flight_number: str | None = None,
    dep_from: str | None = None,
    dep_to: str | None = None,
) -> list[dict]:
    """Narrow a departure board. Every filter is optional and they compose.

    The point of the picker is that you should never need to fill all of them:
    an airline plus a rough departure window is usually enough to find your
    flight in a list of ~900. Unset filters (None or empty) are ignored rather
    than matching nothing, so a half-filled form still returns useful rows.

    flight_number matches as a prefix (typing '19' finds 19 and 1900) because
    a traveller reading a booking often types the leading digits first.
    """
    out = flights
    if carrier:
        out = [f for f in out if f["carrier"] == carrier]
    if dest:
        out = [f for f in out if f["dest"] == dest]
    if flight_number:
        want = str(flight_number).strip().lstrip("0")
        if want:
            out = [f for f in out if str(f["flight_number"]).lstrip("0").startswith(want)]
    if dep_from:
        out = [f for f in out if f["dep_time"] >= dep_from]
    if dep_to:
        out = [f for f in out if f["dep_time"] <= dep_to]
    return out


def flight_label(f: dict, dest_name: str | None = None) -> str:
    """'AA 2842  05:00 → CLT  (Charlotte Douglas International Airport)'."""
    base = f"{f['carrier']} {f['flight_number']}   {f['dep_time']} → {f['dest']}"
    return f"{base}   ({dest_name})" if dest_name else base


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def layover_minutes(arr_time: str, dep_time: str) -> int:
    """Scheduled ground time between an arrival and the next departure.

    Wraps past midnight: a 23:40 arrival and a 00:25 departure is 45 minutes,
    not -1,395. Both times are local clock at the connecting airport, which is
    the same airport for both legs, so no timezone conversion is needed.
    """
    gap = _minutes(dep_time) - _minutes(arr_time)
    return gap + 24 * 60 if gap < 0 else gap


@dataclass(frozen=True)
class ConnectionRisk:
    layover_min: int
    slack_min: int  # minutes leg 1 can be late before the connection breaks
    threshold_min: int | None  # the measured threshold actually used
    probability: float | None  # P(leg 1 arrives >= threshold late)
    conservative: bool  # True when threshold < slack, so the risk is overstated
    note: str
    # True when the slack runs past the largest MEASURED threshold, so the
    # number is an upper bound and nothing more. A UI must render it as
    # "below X%", never as the estimate: with 209 minutes of slack, reporting
    # P(>=120) = 25% as if it were the answer told a traveller a comfortable
    # 3.5-hour layover was a one-in-four risk.
    upper_bound: bool = False


def connection_risk(
    arr_time: str,
    dep_time: str,
    exceedance_bins: list[dict],
    leg1_probability: float,
    mct_min: int = DEFAULT_MCT_MIN,
) -> ConnectionRisk:
    """How likely leg 1 is late enough to break the connection.

    NOT a product of the two legs' delay probabilities. Multiplying them would
    answer "will both be late", which is not the question and would be wrong
    anyway — two legs on the same day share weather and airline operations, so
    they are positively correlated and the product understates joint risk.

    The question is whether LEG 1's arrival delay eats the slack:

        slack = scheduled layover - minimum connect time
        risk  = P(leg 1 arrives >= slack minutes late)

    That probability is read from the HELD-OUT outcome mix for leg 1's own
    probability band — measured frequencies from flights the model never
    trained on, not a modelled distribution.

    The table carries thresholds at 15/30/60/90/120 minutes, so exact slack
    values are answered at the largest threshold AT OR BELOW the slack. Since
    P(delay >= t) is decreasing in t, that OVERSTATES the risk slightly — the
    safe direction for someone deciding whether a connection is tight. The
    flag and note say so rather than implying false precision.
    """
    layover = layover_minutes(arr_time, dep_time)
    slack = layover - mct_min
    band = next(
        (
            b
            for b in exceedance_bins
            if b.get("n")
            and (
                b["lo"] <= leg1_probability < b["hi"]
                or (b["hi"] == 1.0 and leg1_probability == 1.0)
            )
        ),
        None,
    )
    if band is None:
        return ConnectionRisk(layover, slack, None, None, False, "no held-out data for this band")

    thresholds = sorted(int(t) for t in band["exceedance"])
    if slack <= 0:
        return ConnectionRisk(
            layover,
            slack,
            None,
            None,
            False,
            f"the {layover}-minute layover is already inside the {mct_min}-minute "
            "connection time — this is not a connection you can rely on even if "
            "leg 1 is perfectly on time",
        )
    usable = [t for t in thresholds if t <= slack]
    if not usable:
        # slack smaller than the smallest measured threshold: even the mildest
        # measured delay breaks it, so P(>= smallest) is a LOWER bound
        t = thresholds[0]
        return ConnectionRisk(
            layover,
            slack,
            t,
            float(band["exceedance"][str(t)]),
            False,
            f"leg 1 has only {slack} minutes of slack — less than the smallest "
            f"measured threshold ({t} min), so the real risk is HIGHER than shown",
        )
    t = usable[-1]
    p = float(band["exceedance"][str(t)])
    if t == thresholds[-1] and slack > t:
        # Slack runs past everything we measured. P(>= slack) is strictly less
        # than P(>= t), and possibly MUCH less, so this is only an upper bound.
        # Reporting it as the estimate is actively misleading: a 209-minute
        # slack scored against the 120-minute threshold reads as a 25% risk on
        # a layover that is in practice comfortable.
        return ConnectionRisk(
            layover,
            slack,
            t,
            p,
            True,
            f"leg 1 would need to arrive {slack}+ minutes late to break this "
            f"connection — beyond the {t}-minute limit of what we measured. The "
            f"real risk is BELOW the figure shown, likely well below",
            upper_bound=True,
        )
    return ConnectionRisk(
        layover,
        slack,
        t,
        p,
        t < slack,
        f"leg 1 would have to arrive {slack}+ minutes late to break this "
        f"connection; the nearest measured threshold is {t} minutes"
        + (", so this slightly overstates the risk" if t < slack else ""),
    )
