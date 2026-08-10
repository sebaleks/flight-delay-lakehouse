"""Replay the shipped models over HELD-OUT flights: prediction vs what happened.

The demo counterpart to ml/api.py. The API scores FUTURE flights (real NDFD
forecast, outcome unknowable yet); this scores flights from the held-out
window, where the outcome IS known — so a prediction can be shown next to the
truth.

WEATHER, STATED HONESTLY. These rows carry the OBSERVED weather at scheduled
departure (the mart's training-time definition), not the forecast a live
caller would get. api.weather.gov serves only the current forecast, so the
forecast issued before a 2024 flight is not retrievable — routing a past date
through /predict yields has_origin_weather=false and silently drops all twelve
weather features. Consequence to state whenever these numbers are shown:
**this replay is the test-set regime; live serving substitutes forecasts for
observations (train/serve gap #1 in ml/README.md) and will be somewhat
worse.** Quantifying that gap needs archived NDFD grids for the test window.

NOT A SELECTION. Nothing here fits, tunes, or chooses anything — it scores the
ONE shipped model and reports. That is the diagnostic-report case rule 7 of
docs/leakage_discipline.md explicitly permits; no candidate is compared, so
the held-out set is not being re-used as a selection surface.

Run:
    uv run --extra ml python -m ml.replay                      # aggregate + examples
    uv run --extra ml python -m ml.replay --origin ORD --limit 15
    uv run --extra ml python -m ml.replay --date 2024-09-13 --sample 200000
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ml import features as f
from ml.serving import ServingContext, build_context, coerce_feature_frame

log = logging.getLogger("ml.replay")

MART_TABLE = "ml_flight_features"
DEFAULT_SAMPLE = 100_000
IDENTITY = ["flight_date", "carrier", "flight_number", "origin", "dest"]
# the earliest date the held-out window can contain — the belt behind the
# `not is_training_row` filter. Used by main()'s assert and by the API's
# airport-day guard when the artifacts run predates metrics.json.
HOLDOUT_FLOOR = pd.Timestamp("2024-07-01")
# schedule-linkage columns the ops page needs ALONGSIDE the score: hour for
# bank aggregation, rotation position/legs for downstream exposure (all
# schedule-derived; NULL on swap-shaped linkages per the tail-swap
# restriction), the tight-turnaround flag for fragile-bank screening.
AIRPORT_DAY_EXTRA_COLS = ("crs_dep_hour", "rotation_position", "legs_today", "is_tight_turnaround")


class NoHeldOutRows(LookupError):
    """No held-out mart rows matched the requested filters."""


def load_holdout(
    ctx: ServingContext,
    sample: int | None = DEFAULT_SAMPLE,
    origin: str | None = None,
    flight_date: str | None = None,
) -> pd.DataFrame:
    """Held-out mart rows, features + labels + identity.

    Sampling is DETERMINISTIC (a fingerprint ordering, not RAND()) so a demo
    shows the same flights every run, and spreads across the whole test window
    instead of taking the first N calendar days.

    sample=None loads the WHOLE held-out window (3,561,782 rows) with no LIMIT —
    used by ml/exceedance.py, which needs every row for stable tail estimates
    and is not a demo. Downcast to float32/category on arrival, as ml/data.py
    does, or the frame is several GB of float64 and object strings.
    """
    # carrier / origin / dest / route are already FEATURES — re-selecting them
    # here would emit duplicate column names and make ORDER BY ambiguous
    cols = ", ".join(
        [
            *f.FEATURES,
            *f.LABELS,
            "flight_date",
            "cast(flight_number as string) as flight_number",
            "format_time('%H:%M', crs_dep_time) as dep_time",
        ]
    )
    where = [f"not {f.SPLIT_COL}"]
    params = []
    if origin:
        where.append("origin = @origin")
        params.append(("origin", "STRING", origin.upper()))
    if flight_date:
        where.append("flight_date = @d")
        params.append(("d", "DATE", flight_date))

    from google.cloud import bigquery

    sql = (
        f"select {cols} from `{ctx.bq.project}.{ctx.gold}.{MART_TABLE}` "
        f"where {' and '.join(where)} "
        # the fingerprint ordering is what makes a SAMPLE deterministic; with
        # no limit it only costs a sort, so skip it when taking everything
        + (
            "order by farm_fingerprint(concat(cast(flight_date as string), carrier, "
            "cast(flight_number as string), origin, dest, cast(crs_dep_time as string))) "
            f"limit {int(sample)}"
            if sample is not None
            else ""
        )
    )
    df = (
        ctx.bq.query(
            sql,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter(*p) for p in params]
            ),
        )
        .result()
        .to_dataframe(create_bqstorage_client=True)
    )
    if df.empty:
        raise SystemExit("no held-out rows matched those filters")
    # downcast on arrival, as ml/data.py does — at the full 3.56M rows the
    # float64 + object-string frame is several GB
    for c in f.CATEGORICAL_FEATURES:
        df[c] = df[c].astype("category")
    for c in f.NUMERIC_FEATURES:
        df[c] = (
            df[c].astype("Float32").astype("float32")
            if str(df[c].dtype) == "boolean"
            else pd.to_numeric(df[c]).astype("float32")
        )
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    return df


def score(ctx: ServingContext, df: pd.DataFrame) -> pd.DataFrame:
    """Score held-out rows through the shipped artifacts.

    Same coercion and same schema gates as request serving (shared
    coerce_feature_frame), same Platt map — so a replayed probability is the
    number the deployed endpoint would produce for an identical feature vector.
    """
    x = coerce_feature_frame(ctx, df[list(f.FEATURES)].copy())
    p_raw = ctx.models.clf.predict_proba(x)[:, 1]
    out = df[[*IDENTITY, "dep_time", *f.LABELS]].copy()
    out["delay_probability"] = ctx.models.calibrator.transform(p_raw)
    out["expected_delay_minutes"] = ctx.models.reg.predict(x)
    out["has_origin_weather"] = x["has_origin_weather"].to_numpy() == 1.0
    return out


def score_airport_day(ctx: ServingContext, origin: str, day: str) -> pd.DataFrame:
    """Score ONE airport's held-out day, keeping the ops page's schedule columns.

    The whole day, no sampling — an airport-day is a few hundred to ~1,000
    rows, and a capacity view of a bank with flights missing would be wrong,
    not just imprecise. Features come straight from the mart (no forecast
    call, no feature assembly); labels ride along so prediction can be shown
    next to what happened.
    """
    try:
        df = load_holdout(ctx, sample=None, origin=origin, flight_date=day)
    except SystemExit as exc:  # load_holdout's CLI-shaped empty signal
        raise NoHeldOutRows(str(exc)) from None
    scored = score(ctx, df)
    for c in AIRPORT_DAY_EXTRA_COLS:
        # score() copies out of df with the index preserved, so this aligns
        scored[c] = df[c]
    return scored


def _opt(v, cast=float):
    """NaN-safe JSON value: the mart's NULLs (e.g. swap-shaped rotation
    linkage) must serialize as null, never as the string 'nan'."""
    return None if pd.isna(v) else cast(v)


def airport_day_payload(origin: str, day: str, scored: pd.DataFrame, artifacts: str) -> dict:
    """The /replay/airport-day response body. Pure — a frame in, JSON out.

    summary carries the headline the ops page opens with: expected delayed
    departures Σp with the Poisson-binomial standard deviation √Σp(1−p)
    (exact for a sum of independent Bernoullis — the model's own claim about
    the day), next to the actual count. Per-flight rows keep labels and the
    schedule-linkage columns so the page can aggregate by bank without a
    second query.
    """
    p = scored["delay_probability"].to_numpy(dtype=float)
    ordered = scored.sort_values(["dep_time", "carrier", "flight_number", "dest"])
    flights = [
        {
            "carrier": str(r["carrier"]),
            "flight_number": str(r["flight_number"]),
            "dest": str(r["dest"]),
            "dep_time": str(r["dep_time"]),
            "dep_hour": int(r["crs_dep_hour"]),
            "delay_probability": round(float(r["delay_probability"]), 4),
            "expected_delay_minutes": round(float(r["expected_delay_minutes"]), 1),
            "actual_delayed": bool(r["label_arr_del15"]),
            "actual_delay_minutes": _opt(r["label_arr_delay_minutes"]),
            "has_origin_weather": bool(r["has_origin_weather"]),
            "rotation_position": _opt(r["rotation_position"], int),
            "legs_today": _opt(r["legs_today"], int),
            "is_tight_turnaround": _opt(r["is_tight_turnaround"], lambda v: bool(int(v))),
        }
        for _, r in ordered.iterrows()
    ]
    return {
        "origin": origin,
        "date": day,
        "artifacts": artifacts,
        "n_flights": len(scored),
        # the honesty block, same spirit as /predict's prediction_basis: these
        # rows carry OBSERVED weather at scheduled departure (the test-set
        # regime); live serving substitutes forecasts and will be somewhat
        # worse — see the module docstring
        "prediction_basis": {
            "mode": "held_out_replay",
            "weather": "observed_at_scheduled_departure",
            "note": (
                "test-set regime: replay uses observed weather; "
                "live serving substitutes forecasts (ml/replay.py)"
            ),
        },
        "summary": {
            "expected_delayed": round(float(p.sum()), 2),
            "expected_delayed_sd": round(float((p * (1 - p)).sum() ** 0.5), 2),
            "actual_delayed": int(scored["label_arr_del15"].astype(bool).sum()),
        },
        "flights": flights,
    }


def report(scored: pd.DataFrame, limit: int) -> None:
    y = scored["label_arr_del15"].astype(int).to_numpy()
    p = scored["delay_probability"].to_numpy()
    base = y.mean()

    print(f"\nHELD-OUT REPLAY — {len(scored):,} flights the model has never seen")
    print(
        f"  window        {scored['flight_date'].min():%Y-%m-%d} .. "
        f"{scored['flight_date'].max():%Y-%m-%d}"
    )
    print(f"  base rate     {base:.4f} delayed >=15 min")
    if y.min() != y.max():
        print(f"  ROC-AUC       {roc_auc_score(y, p):.4f}   (sample, not the pinned headline)")
        print(f"  PR-AUC        {average_precision_score(y, p):.4f}   (sample)")
    print("  weather       OBSERVED at scheduled departure — production serves FORECASTS")

    # Does the calibrated probability mean what it says? Predicted vs actual
    # per band is the claim (held-out ECE 0.017) made legible.
    print("\n  CALIBRATION — does 'p' behave like a frequency?")
    print("    predicted band      n        mean p    actual rate")
    bands = pd.cut(scored["delay_probability"], [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    for band, g in scored.groupby(bands, observed=True):
        print(
            f"    {str(band):18s} {len(g):>8,}   {g['delay_probability'].mean():>7.3f}"
            f"        {g['label_arr_del15'].mean():>7.3f}"
        )

    top = scored.nlargest(max(1, len(scored) // 10), "delay_probability")
    print(
        f"\n  TOP DECILE    {top['label_arr_del15'].mean():.3f} actually delayed "
        f"vs {base:.3f} base — {top['label_arr_del15'].mean() / base:.2f}x lift"
    )

    # Context for the example block below. Showing the top-N flights and
    # counting how many were delayed measures the EXTREME TAIL of the ranking,
    # not the model — the further up the cut, the better it looks. Print the
    # whole curve so nobody (including us) reads "7 of the top 8 were delayed"
    # as an accuracy claim.
    print("\n  HOW EXTREME IS THE TOP? actual delay rate by cut")
    print("    cut                n        actual rate")
    cuts = [c for c in (10, 100, 1_000, 10_000, len(scored) // 10) if 0 < c <= len(scored)]
    for n in sorted(set(cuts)):
        g = scored.nlargest(n, "delay_probability")
        pct = 100 * n / len(scored)
        print(f"    top {n:<8,} ({pct:>5.2f}%) {n:>8,}      {g['label_arr_del15'].mean():>6.3f}")
    print(f"    all              {len(scored):>8,}      {base:>6.3f}")

    _examples(
        scored.nlargest(limit, "delay_probability"),
        f"TOP OF THE RANKING ({limit} highest-scored of {len(scored):,} — the extreme tail,"
        " NOT a representative sample)",
    )

    # The honest counterpart: a deterministic walk across the probability
    # range, so the misses are visible alongside the hits.
    # sample, not head: the frame arrives in fingerprint order, so a fixed
    # random_state is reproducible run-to-run while spreading picks across the
    # whole window. Sorting by IDENTITY and taking head() would be equally
    # deterministic but would put every example on the earliest date.
    per_band = max(1, limit // 6)
    picks = [
        g.sample(min(per_band, len(g)), random_state=0)
        for _, g in scored.groupby(bands, observed=True)
    ]
    _examples(
        pd.concat(picks).sort_values("delay_probability", ascending=False),
        f"ACROSS THE RANGE ({per_band} per predicted band — this is the representative view)",
    )


def _examples(rows: pd.DataFrame, title: str) -> None:
    print(f"\n  {title}")
    print("    flight                                  p      E[min]   ACTUAL")
    for _, r in rows.iterrows():
        actual = "DELAYED" if r["label_arr_del15"] else "on time"
        label = (
            f"{r['carrier']} {r['flight_number']:>4s} {r['origin']}->{r['dest']} "
            f"{r['flight_date']:%Y-%m-%d} {r['dep_time']}"
        )
        print(
            f"    {label:38s} {r['delay_probability']:.3f}  {r['expected_delay_minutes']:>6.1f}"
            f"   {actual:7s} ({r['label_arr_delay_minutes']:+.0f} min)"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    ap.add_argument("--origin", default=None)
    ap.add_argument("--date", default=None, help="a single held-out flight_date, YYYY-MM-DD")
    ap.add_argument("--limit", type=int, default=10, help="example flights to print")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ctx = build_context()
    df = load_holdout(ctx, args.sample, args.origin, args.date)
    scored = score(ctx, df)
    # the filter is on the mart's own split column, but a demo that silently
    # scored a training row would be the exact failure this whole project is
    # about — so prove it, don't assume it
    assert scored["flight_date"].min() >= HOLDOUT_FLOOR, "training row in replay!"
    report(scored, args.limit)


if __name__ == "__main__":
    main()
