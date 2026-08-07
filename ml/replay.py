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


def load_holdout(
    ctx: ServingContext,
    sample: int = DEFAULT_SAMPLE,
    origin: str | None = None,
    flight_date: str | None = None,
) -> pd.DataFrame:
    """Held-out mart rows, features + labels + identity.

    Sampling is DETERMINISTIC (a fingerprint ordering, not RAND()) so a demo
    shows the same flights every run, and spreads across the whole test window
    instead of taking the first N calendar days.
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
        "order by farm_fingerprint(concat(cast(flight_date as string), carrier, "
        "cast(flight_number as string), origin, dest, cast(crs_dep_time as string))) "
        f"limit {int(sample)}"
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

    print(f"\n  EXAMPLES ({limit} flights, highest predicted risk first)")
    print("    flight                                  p      E[min]   ACTUAL")
    for _, r in scored.nlargest(limit, "delay_probability").iterrows():
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
    assert scored["flight_date"].min() >= pd.Timestamp("2024-07-01"), "training row in replay!"
    report(scored, args.limit)


if __name__ == "__main__":
    main()
