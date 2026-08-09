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

    # Deterministic, spread across the mart: real entities so the hist lookups
    # actually hit, ordered by fingerprint so the sample is stable run to run.
    rows = list(
        bq.query(
            f"""
            select carrier, origin, dest,
                   format_time('%H:%M', crs_dep_time) as dep_time,
                   any_value(distance) as distance,
                   any_value(rotation_position) as rotation_position,
                   any_value(legs_today) as legs_today,
                   any_value(sched_turnaround_min) as sched_turnaround_min,
                   any_value(inbound_distance) as inbound_distance,
                   any_value(inbound_crs_elapsed_min) as inbound_crs_elapsed_min,
                   any_value(origin_dep_density_hour) as origin_dep_density_hour
            from `{project}.{gold}.ml_flight_features`
            group by carrier, origin, dest, dep_time
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


def compare(before: str, after: str) -> None:
    """Compare two captures, SPLIT by whether a request depends on the values a
    deliberate change would move.

    A request that supplied both its rotation context and its density touches
    neither the typical profile nor the density medians, so it must be
    bit-identical under any pure refactor AND under a medians-only change. That
    subset is the hard assertion; the rest is reported quantitatively so a real
    regression is not hidden behind an expected shift.
    """
    a = json.load(open(before))
    b = json.load(open(after))
    if len(a) != len(b):
        raise SystemExit(f"capture sizes differ: {len(a)} vs {len(b)}")

    def independent(r):
        return r["rotation_context"] == "provided" and r["origin_density_source"] == "provided"

    clean = [(x, y) for x, y in zip(a, b, strict=True) if independent(x)]
    rest = [(x, y) for x, y in zip(a, b, strict=True) if not independent(x)]

    failures = []
    for x, y in clean:
        failures += list(_diff(x, y, x["flight"]))
    print(f"INDEPENDENT of the typical profile / density medians: {len(clean)} requests")
    if failures:
        print(f"  PARITY FAILED — {len(failures)} difference(s):")
        for d in failures[:40]:
            print("    " + d)
    else:
        n_feat = len(clean[0][0].get("features", {})) if clean else 0
        print(f"  bit-identical ({n_feat} features each) — PASS")

    if rest:
        dp = [abs(x["delay_probability"] - y["delay_probability"]) for x, y in rest]
        moved = set()
        for x, y in rest:
            for k, v in x.get("features", {}).items():
                if v != y.get("features", {}).get(k):
                    moved.add(k)
        print(f"\nDEPENDENT on those values: {len(rest)} requests")
        print(f"  unchanged anyway: {sum(1 for d in dp if d == 0)}")
        print(f"  |delta p|: max {max(dp):.4f}, mean {sum(dp) / len(dp):.5f}")
        print(f"  features that moved: {sorted(moved) or 'none'}")
        print("  (expected to be EMPTY for a pure refactor; for a medians change,")
        print("   only the features those medians feed may appear here)")

    if failures:
        sys.exit(1)


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "capture":
        capture(sys.argv[2])
    elif cmd == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
