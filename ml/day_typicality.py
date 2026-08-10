"""Is a demo date TYPICAL, or the day the model happened to look good?

The calibration-implies-counts check. If the calibrated per-flight
probabilities are right, an airport-day's delayed-departure count is a
Poisson-binomial draw: mean Σp, sd √Σp(1−p). Standardising every held-out day
at an origin, z = (actual − Σp) / sd, turns the whole test window into one
distribution — and a candidate demo day into one point inside (or outside) it.

Under perfect calibration AND independence, z is ~standard normal. In reality
delays share shocks (a storm hits the whole bank), so the observed z spread is
WIDER than 1 — the report prints the actual spread rather than pretending.
What makes a day fair to demo is not |z| ≈ 0 but being unexceptional: inside
the central 10th–90th percentile band of the origin's own z distribution.
Both tails are cherry-picks — a day the model nailed suspiciously well is as
selected as a day it missed. This is the same trap as the replay's top-8
examples (see ml/replay.py's report), applied to days instead of flights.

NOT A SELECTION. Scores the one shipped model and reports — the
diagnostic-report case rule 7 of docs/leakage_discipline.md permits. Nothing
here feeds back into training or model choice.

Run (needs ADC + the mart; scores one origin's whole held-out window):
    uv run --extra ml --extra ingestion python -m ml.day_typicality --origin ORD
    uv run --extra ml --extra ingestion python -m ml.day_typicality \\
        --origin ORD --date 2024-09-13
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging

import pandas as pd

from ml.replay import load_holdout, score
from ml.serving import build_context

log = logging.getLogger("ml.day_typicality")

# the unexceptional band: a demo day must sit inside the central 80% of the
# origin's own z distribution — both tails are selection
QUANTILE_LO = 0.10
QUANTILE_HI = 0.90


def daily_moments(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-day Poisson-binomial moments and the standardised miss z.

    One row per flight_date: n, expected (Σp), sd (√Σp(1−p)), actual, z.
    Pure — takes ml/replay.py's scored frame, touches nothing else.
    """
    g = scored.groupby(scored["flight_date"].dt.date)
    daily = pd.DataFrame(
        {
            "n": g.size(),
            "expected": g["delay_probability"].sum(),
            "variance": g["delay_probability"].apply(lambda s: float((s * (1 - s)).sum())),
            "actual": g["label_arr_del15"].sum().astype(int),
        }
    )
    daily["sd"] = daily["variance"] ** 0.5
    daily["z"] = (daily["actual"] - daily["expected"]) / daily["sd"]
    return daily.drop(columns=["variance"])


def verdict(daily: pd.DataFrame, day: dt.date) -> dict:
    """Where one day sits in the origin's z distribution, and whether that is
    unexceptional enough to demo. Raises KeyError if the day is not present."""
    z = daily["z"]
    day_z = float(daily.loc[day, "z"])
    lo, hi = float(z.quantile(QUANTILE_LO)), float(z.quantile(QUANTILE_HI))
    return {
        "date": day,
        "z": day_z,
        "percentile": float((z < day_z).mean()),
        "band": (lo, hi),
        "typical": lo <= day_z <= hi,
    }


def report(origin: str, daily: pd.DataFrame, chosen: dict | None) -> None:
    z = daily["z"]
    print(f"\nDAY TYPICALITY — {origin}, {len(daily)} held-out days")
    print(f"  window            {daily.index.min()} .. {daily.index.max()}")
    print(f"  z mean            {z.mean():+.3f}   (0 = counts match calibration on average)")
    print(
        f"  z sd              {z.std():.3f}   (1 = independence would hold; wider = shared "
        "shocks, which is expected)"
    )
    for k in (1, 2):
        print(f"  days within ±{k}sd  {(z.abs() <= k).mean():.1%}")
    print("\n  MOST EXTREME DAYS (either tail — both are cherry-picks)")
    print("    date          n   expected     actual      z")
    extremes = daily.reindex(z.abs().sort_values(ascending=False).index).head(5)
    for day, r in extremes.iterrows():
        print(
            f"    {day}  {int(r['n']):>4}   {r['expected']:>7.1f}    "
            f"{int(r['actual']):>5}   {r['z']:>+6.2f}"
        )
    if chosen is not None:
        r = daily.loc[chosen["date"]]
        lo, hi = chosen["band"]
        print(f"\n  CHOSEN DAY {chosen['date']}")
        print(
            f"    expected {r['expected']:.1f} ± {2 * r['sd']:.1f} (2sd), actual {int(r['actual'])}"
            f", z = {chosen['z']:+.2f} — {chosen['percentile']:.0%}th percentile"
        )
        print(f"    typicality band   z in [{lo:+.2f}, {hi:+.2f}] (central 80% of this origin)")
        state = "TYPICAL — fair to demo" if chosen["typical"] else "NOT typical — pick another day"
        print(f"    verdict           {state}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origin", required=True)
    ap.add_argument("--date", default=None, help="candidate demo date, YYYY-MM-DD")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ctx = build_context()
    # the WHOLE held-out window for one origin, no sampling: a day's z from a
    # partial day would mix sampling noise into a calibration statement
    scored = score(ctx, load_holdout(ctx, sample=None, origin=args.origin))
    daily = daily_moments(scored)
    chosen = None
    if args.date:
        day = dt.date.fromisoformat(args.date)
        try:
            chosen = verdict(daily, day)
        except KeyError:
            raise SystemExit(f"{day} has no held-out flights at {args.origin}") from None
    report(args.origin.upper(), daily, chosen)


if __name__ == "__main__":
    main()
