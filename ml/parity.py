"""Golden-vector parity harness for the serving path.

Run this before and after ANY change to the serving lookup layer — the
`serving_*` dbt models or the load/lookup code in ml/serving.py. Those models
are the SAME queries the request path used to run, materialized, so a pure
refactor must return bit-identical predictions.

    uv run --extra ml --extra serve --extra ingestion python -m ml.parity \\
        capture before.json          # on the old code
    # ... make the change, rebuild the lookups ...
    uv run --extra ml --extra serve --extra ingestion python -m ml.parity \\
        capture after.json
    uv run --extra ml python -m ml.parity compare before.json after.json

compare() exits NON-ZERO on any unexpected difference. By default every request
must be bit-identical — the contract for a pure refactor. Add
`--expect-medians-change` only when the change deliberately alters the typical
rotation profile or the density medians; requests that supplied both their
rotation context and their density must STILL be bit-identical, and the rest
may move only in the features those medians can reach.

The 184 requests deliberately span both rotation paths (context provided vs the
typical-profile estimate), both density paths (provided vs estimated), and four
entities absent from the training vocabulary, so the unseen-category and NaN
paths are exercised rather than assumed.

DETERMINISM NOTE: the flight date is deliberately ~60 days out, beyond the
~7-day NDFD horizon, so every row takes the weather NULL path. A near-date
would fetch a live forecast that changes between the two runs, which would make
parity untestable — the differences would be weather, not the code under test.
Weather-present behaviour is covered by a live API smoke test instead.

INTERPRETING A DIFF: a change that deliberately alters the typical rotation
profile or the density medians will move requests that USE them, and only in
the features those values feed. Split the comparison by
`rotation_context`/`origin_density_source` before concluding anything — the
requests that supplied both must still be bit-identical. See
docs/benchmarks/serving_preload_benchmark.md for a worked example.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date, timedelta

TARGET_DATE = date.today() + timedelta(days=60)


def build_flights():
    from google.cloud import bigquery

    from ingestion.config import require_env
    from ml.serving import FlightRequest

    project = require_env("GCP_PROJECT_ID")
    gold = require_env("BQ_GOLD_DATASET")
    bq = bigquery.Client(project=project)

    # ONE REAL FLIGHT PER GROUP, deterministically chosen.
    #
    # The obvious version of this query — group by (carrier, origin, dest,
    # dep_time) with any_value() per column — is wrong twice over, in exactly
    # the way this harness exists to detect. A group spans many dates, so each
    # any_value() picks INDEPENDENTLY: the assembled rotation context could
    # take rotation_position from one flight and sched_turnaround_min from
    # another, a shape no real flight has. And any_value() is arbitrary, so the
    # picks can differ between two captures of the SAME code, producing false
    # parity regressions. row_number() over a total order takes every field
    # from a single row and returns the same row every time.
    rows = list(
        bq.query(
            f"""
            select * except (rn) from (
                select carrier, origin, dest,
                       format_time('%H:%M', crs_dep_time) as dep_time,
                       distance, rotation_position, legs_today,
                       sched_turnaround_min, inbound_distance,
                       inbound_crs_elapsed_min, origin_dep_density_hour,
                       row_number() over (
                           partition by carrier, origin, dest, crs_dep_time
                           order by flight_date, flight_number, dest
                       ) as rn
                from `{project}.{gold}.ml_flight_features`
            )
            where rn = 1
            order by farm_fingerprint(concat(carrier, origin, dest, dep_time))
            limit 180
            """
        ).result()
    )

    flights = []
    for i, r in enumerate(rows):
        # alternate: half carry the full rotation context, half omit it entirely
        # so both the "provided" and typical-profile paths are covered
        with_rot = i % 2 == 0 and r["rotation_position"] is not None
        flights.append(
            FlightRequest(
                origin=r["origin"],
                dest=r["dest"],
                carrier=r["carrier"],
                flight_date=TARGET_DATE,
                dep_time=r["dep_time"],
                arr_time="23:15",
                distance=float(r["distance"]) if r["distance"] is not None else None,
                rotation_position=int(r["rotation_position"]) if with_rot else None,
                legs_today=int(r["legs_today"]) if with_rot else None,
                sched_turnaround_min=(
                    float(r["sched_turnaround_min"])
                    if with_rot and r["sched_turnaround_min"] is not None
                    else None
                ),
                inbound_distance=(
                    float(r["inbound_distance"])
                    if with_rot and r["inbound_distance"] is not None
                    else None
                ),
                inbound_crs_elapsed_min=(
                    float(r["inbound_crs_elapsed_min"])
                    if with_rot and r["inbound_crs_elapsed_min"] is not None
                    else None
                ),
                # exercise both the provided and the estimated density path
                origin_dep_density_hour=(
                    float(r["origin_dep_density_hour"])
                    if i % 3 == 0 and r["origin_dep_density_hour"] is not None
                    else None
                ),
            )
        )

    # Edge cases: entities absent from the training vocabulary must still take
    # the NaN/unseen-category path rather than crash or silently change.
    edge = [
        ("ZZ", "ORD", "LAX"),  # unknown carrier
        ("AA", "XXX", "LAX"),  # unknown origin
        ("AA", "ORD", "YYY"),  # unknown dest
        ("ZZ", "XXX", "YYY"),  # everything unknown
    ]
    for carrier, origin, dest in edge:
        flights.append(
            FlightRequest(
                origin=origin,
                dest=dest,
                carrier=carrier,
                flight_date=TARGET_DATE,
                dep_time="17:30",
                arr_time="20:05",
                distance=1200.0,
            )
        )
    return flights


def capture(path: str) -> None:
    from ml.serving import build_context, predict

    ctx = build_context()
    flights = build_flights()
    out = predict(ctx, flights)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, allow_nan=True)
    print(f"captured {len(out)} predictions -> {path}")


def _diff(a, b, trail):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            yield from _diff(a.get(k), b.get(k), f"{trail}.{k}")
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield f"{trail}: length {len(a)} != {len(b)}"
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):  # lengths checked above
            yield from _diff(x, y, f"{trail}[{i}]")
        return
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return  # NaN == NaN for our purposes: both took the missing path
    if a != b:
        yield f"{trail}: {a!r} != {b!r}"


# Exactly the features the typical rotation profile and the density estimate
# can reach. A deliberate change to those medians may move these and NOTHING
# else; anything outside this set moving is a regression, not an expectation.
MEDIAN_REACHABLE = {
    "rotation_position",
    "legs_today",
    "origin_dep_density_hour",
    "has_inbound_leg",
    "sched_turnaround_min",
    "sched_turnaround_slack_min",
    "is_tight_turnaround",
    "inbound_distance",
    "inbound_crs_elapsed_min",
} | {
    f"hist_{grain}_{stat}"
    # a changed typical profile can move the derived band / position KEY, which
    # then selects a different hist triple
    for grain in ("turnaround_band", "rotation_position")
    for stat in ("arr_del15_rate", "avg_arr_delay_minutes", "n_flights")
}

# response fields that are model OUTPUTS — they are expected to move whenever
# an input feature legitimately moves, so they are not themselves evidence
_OUTPUTS = {"delay_probability", "expected_delay_minutes", "logreg_baseline_probability"}


def compare(before: str, after: str, expect_medians_change: bool = False) -> None:
    """Compare two captures. Exits non-zero on any unexpected difference.

    DEFAULT MODE (pure refactor): every request must be bit-identical. This is
    the contract for a change that only moves where values are read from.

    --expect-medians-change: for a change that deliberately alters the typical
    rotation profile or the density medians. Requests that supplied BOTH their
    rotation context and their density are independent of those values and must
    still be bit-identical; the rest may move, but ONLY in the features those
    medians can reach (MEDIAN_REACHABLE). A moved feature outside that set —
    say a hist_route_* value, or has_origin_weather — means the lookup layer
    broke, and fails even in this mode.

    The point of the second mode is that "some things changed, here are the
    numbers" must never be an exit code of 0. A broken density table would land
    entirely in the non-independent subset.
    """
    a = json.load(open(before))
    b = json.load(open(after))
    if len(a) != len(b):
        raise SystemExit(f"capture sizes differ: {len(a)} vs {len(b)}")

    def independent(r):
        return r["rotation_context"] == "provided" and r["origin_density_source"] == "provided"

    pairs = list(zip(a, b, strict=True))
    clean = [(x, y) for x, y in pairs if independent(x)]
    rest = [(x, y) for x, y in pairs if not independent(x)]

    failures = []
    for x, y in clean:
        failures += list(_diff(x, y, x["flight"]))
    print(f"INDEPENDENT of the typical profile / density medians: {len(clean)} requests")
    if failures:
        print(f"  FAILED — {len(failures)} difference(s):")
        for d in failures[:40]:
            print("    " + d)
    else:
        n_feat = len(clean[0][0].get("features", {})) if clean else 0
        print(f"  bit-identical ({n_feat} features each) — PASS")

    if rest:
        print(f"\nDEPENDENT on those values: {len(rest)} requests")
        if not expect_medians_change:
            # pure-refactor contract: these must be identical too
            rest_diffs = []
            for x, y in rest:
                rest_diffs += list(_diff(x, y, x["flight"]))
            if rest_diffs:
                failures += rest_diffs
                print(f"  FAILED — {len(rest_diffs)} difference(s); pass")
                print("  --expect-medians-change only if this change is SUPPOSED to move them")
                for d in rest_diffs[:20]:
                    print("    " + d)
            else:
                print("  bit-identical — PASS")
        else:
            dp = [abs(x["delay_probability"] - y["delay_probability"]) for x, y in rest]
            moved: set[str] = set()
            other: list[str] = []
            for x, y in rest:
                for k, v in x.get("features", {}).items():
                    if v != y.get("features", {}).get(k):
                        moved.add(k)
                for k, v in x.items():
                    if k in ("features", "flight") or k in _OUTPUTS:
                        continue
                    if v != y.get(k):
                        other.append(f"{x['flight']}: {k} {v!r} -> {y.get(k)!r}")
            print(f"  unchanged anyway: {sum(1 for d in dp if d == 0)}")
            print(f"  |delta p|: max {max(dp):.4f}, mean {sum(dp) / len(dp):.5f}")
            print(f"  features that moved: {sorted(moved) or 'none'}")
            unexpected = sorted(moved - MEDIAN_REACHABLE)
            if unexpected:
                failures.append("unexpected")
                print(f"  FAILED — moved OUTSIDE the median-reachable set: {unexpected}")
                print("  the medians cannot reach those; the lookup layer changed behaviour")
            if other:
                failures.append("unexpected")
                print(f"  FAILED — non-output response fields moved: {other[:10]}")
            if not unexpected and not other:
                print("  all moves are median-reachable — PASS")

    if failures:
        sys.exit(1)
    print("\nPARITY OK")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    unknown = flags - {"--expect-medians-change"}
    if unknown:
        raise SystemExit(f"unknown flag(s) {sorted(unknown)}")
    cmd = args[0] if args else ""
    if cmd == "capture":
        capture(args[1])
    elif cmd == "compare":
        compare(args[1], args[2], expect_medians_change="--expect-medians-change" in flags)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
