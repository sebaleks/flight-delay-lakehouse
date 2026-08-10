"""Held-out outcome mix: what ACTUALLY happened to flights we scored near p.

Writes `exceedance.json` next to `metrics.json` in an artifacts run, served by
the predictor's GET /outcome-mix.

WHY THIS EXISTS. The regressor's held-out MAE is 18.99 minutes and its RMSE is
49.26 — the error bar dwarfs the number — so showing a consumer
"expected delay: +23 min" is the most misleading thing the model can say. The
honest answer to "how late might I be?" is empirical and comes from data the
model never trained on:

    among held-out flights scored near 30%, ~70% arrived within 15 minutes,
    ~22% were 15-60 late, ~8% were an hour or more late

Same source, same discipline, no new model, and no false precision.

BINS MATCH THE RELIABILITY TABLE. Ten equal-width probability bands, exactly the
bins `ml/calibration.py` writes into metrics.json for the calibration panel. A
consumer page shows both, so they must agree about which band a flight is in;
deciles of the predicted distribution would not line up.

LEAKAGE. Held-out rows only (`is_training_row = false`, asserted), scored by the
one shipped model, reported not selected on — the diagnostic-report case rule 7
of docs/leakage_discipline.md permits. Nothing here fits or chooses anything.

Run (after training, before publishing artifacts):
    uv run --extra ml --extra ingestion python -m ml.exceedance
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ml import features as f
from ml.replay import load_holdout, score
from ml.serving import build_context

log = logging.getLogger("ml.exceedance")

OUTPUT_NAME = "exceedance.json"
# minutes late. 15 is the classifier's own threshold (so the first column must
# reproduce the calibration table's frac_pos, a free consistency check); 60 and
# 120 are the thresholds that matter commercially — missed connections and the
# territory where compensation schemes start to bite.
# 180 and 240 exist for CONNECTION risk: a 3-4 hour layover is common, and
# without them the largest measurable threshold is 120 min, so a comfortable
# 209-minute slack gets scored against P(>=120) and reads as a ~25% risk.
# Beyond the top threshold the consumer page can only report an upper bound.
THRESHOLDS_MIN = (15, 30, 60, 90, 120, 180, 240)
N_BINS = 10


def build_table(scored: pd.DataFrame) -> dict:
    """One row per probability band: how often each delay threshold was crossed."""
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    # np.digitize on the INTERIOR edges, clipped — the same binning
    # ml/calibration.py uses, so the bands line up with the reliability table
    idx = np.clip(np.digitize(scored["delay_probability"].to_numpy(), edges[1:-1]), 0, N_BINS - 1)
    minutes = scored["label_arr_delay_minutes"].to_numpy()
    delayed15 = scored["label_arr_del15"].to_numpy().astype(bool)

    bins = []
    for b in range(N_BINS):
        m = idx == b
        n = int(m.sum())
        row = {
            "lo": round(float(edges[b]), 2),
            "hi": round(float(edges[b + 1]), 2),
            "n": n,
            "mean_pred": round(float(scored["delay_probability"].to_numpy()[m].mean()), 4)
            if n
            else None,
            "frac_delayed_15": round(float(delayed15[m].mean()), 4) if n else None,
            # P(arrival delay >= t) among flights scored in this band.
            # >=, not >: the mart's label_arr_del15 is "delay >= 15", so the 15
            # column must reproduce frac_delayed_15 EXACTLY. With > it was
            # systematically low (0.0682 vs 0.0721) and the self-check could
            # only ever be approximate.
            "exceedance": {
                str(t): round(float((minutes[m] >= t).mean()), 4) if n else None
                for t in THRESHOLDS_MIN
            },
        }
        bins.append(row)
    return {
        "thresholds_min": list(THRESHOLDS_MIN),
        "n_bins": N_BINS,
        "n_rows": int(len(scored)),
        "test_window": [
            scored["flight_date"].min().strftime("%Y-%m-%d"),
            scored["flight_date"].max().strftime("%Y-%m-%d"),
        ],
        "base_rate": round(float(delayed15.mean()), 4),
        "bins": bins,
    }


def run(sample: int | None = None, out_dir: Path | None = None) -> dict:
    ctx = build_context()
    log.info("loading the held-out window (%s)", "all rows" if sample is None else f"{sample:,}")
    df = load_holdout(ctx, sample=sample)
    scored = score(ctx, df)
    # the whole point is that these rows were never trained on — prove it
    assert scored["flight_date"].min() >= pd.Timestamp("2024-07-01"), "training row in exceedance!"
    table = build_table(scored)
    table["artifacts"] = ctx.models.artifacts_dir.name
    table["features"] = len(f.FEATURES)

    target = (out_dir or ctx.models.artifacts_dir) / OUTPUT_NAME
    target.write_text(json.dumps(table, indent=2) + "\n")
    log.info("wrote %s (%d rows, %d bands)", target, table["n_rows"], N_BINS)
    return table


def _print(table: dict) -> None:
    ts = table["thresholds_min"]
    print(
        f"\nHELD-OUT OUTCOME MIX — {table['n_rows']:,} flights, {table['test_window'][0]} .. "
        f"{table['test_window'][1]}, base rate {table['base_rate']}"
    )
    print("  band          n        mean p   " + "".join(f" P(>={t}m)" for t in ts))
    for b in table["bins"]:
        if not b["n"]:
            continue
        cells = "".join(f"  {b['exceedance'][str(t)]:>7.3f}" for t in ts)
        print(f"  {b['lo']:.1f}-{b['hi']:.1f}  {b['n']:>9,}    {b['mean_pred']:>6.3f}{cells}")
    print("\n  P(>=15m) must EQUAL frac_delayed_15 (same threshold, different source):")
    for b in table["bins"][:3]:
        if b["n"]:
            print(
                f"    band {b['lo']:.1f}-{b['hi']:.1f}: "
                f"{b['exceedance']['15']:.4f} vs {b['frac_delayed_15']:.4f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sample",
        type=int,
        default=None,
        help="rows to use; default is the WHOLE held-out window (3,561,782)",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _print(run(sample=args.sample, out_dir=args.out_dir))


if __name__ == "__main__":
    main()
