"""Overnight model-family search: five learners not yet tried, tracked in MLflow.

WHAT THIS IS. ml/experiments.py compares ONE alternative (LightGBM) against the
shipped XGBoost. This widens that to five untried families with a bounded
hyperparameter search each, and reports a VALIDATION ranking. It is a search,
not an adoption: nothing here changes the shipped model.

THE TEST SET IS NEVER TOUCHED. Selection happens on the validation slice carved
from inside the training window (rule 6). The held-out test is reserved for a
one-time confirmation of whichever candidate a human decides to adopt — spending
it here, unattended, would burn it on a search rather than a decision, and
"adopting on a test comparison re-selects on test" (docs/leakage_discipline.md
rule 7). The report says what won on validation and stops.

WHY THIS RE-DERIVES hist_* — THE PART THAT MAKES THE COMPARISON VALID.
Rule 10: the mart's hist_* aggregate the WHOLE pre-cutoff window, so a
validation slice carved from inside the training window carries hist_* computed
partly from validation-period labels. That leak is measured and is NOT
common-mode — it inflated XGBoost's val PR-AUC by +0.00079 against LightGBM's
+0.00020, roughly a 4x difference. On the previous two-way comparison it did not
flip the winner, and the rule says so explicitly:

    "for a closer comparison or a WIDER SEARCH it could matter, so the rigorous
     form remains to re-derive rates as-of the validation cutoff"

A five-family search is exactly that wider search. So this recomputes all 18
hist_* columns using ONLY flights before VAL_START, with the mart's own
smoothing formula, and scores every family on the corrected features. Skipping
this would rank learners partly by how well each exploits a leak.

The recompute is verified before it is used: run with the PRODUCTION cutoff it
must reproduce the mart's own hist_* values. If it does not, the run aborts
rather than searching on features nobody has checked.

Run (this is what the overnight job executes):
    uv run --extra ml --extra ingestion python -m ml.model_search --budget-min 420
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ingestion.config import require_env
from ml import features as f
from ml import tracking
from ml.tuning import VAL_START

log = logging.getLogger("ml.model_search")

SEED = 0
SMOOTHING_M = 50  # var('hist_smoothing_prior_strength') in dbt_project.yml
PROD_CUTOFF = date(2024, 7, 1)  # var('train_test_cutoff_date')
OUT_DIR = Path(__file__).resolve().parent / "search_results"
HIST_GRAINS = ("route", "carrier", "origin", "dest", "turnaround_band", "rotation_position")


# --------------------------------------------------------------------------
# 1. hist_* re-derivation (rule 10)
# --------------------------------------------------------------------------
def _hist_sql(project: str, gold: str, cutoff: str) -> str:
    """Smoothed hist_* per (grain, key) from flights strictly before `cutoff`.

    Mirrors the mart: rate and average are shrunk toward the window's global
    with prior strength m, (n*p + m*global)/(n+m); hist_n_flights is the raw
    count. The band and position keys are derived exactly as
    int_aircraft_rotation derives them, and both are restricted to rows with a
    non-null rotation_position — swap-shaped linkages carry NULL rotation under
    the tail-swap restriction and must not define a band.
    """
    band = """case
            when not has_inbound_leg then 'no_inbound'
            when sched_turnaround_min < 35 then 'lt_35'
            when sched_turnaround_min < 60 then '35_60'
            when sched_turnaround_min < 120 then '60_120'
            else 'ge_120' end"""
    levels = [
        ("route", "route", ""),
        ("carrier", "carrier", ""),
        ("origin", "origin", ""),
        ("dest", "dest", ""),
        ("turnaround_band", band, "and rotation_position is not null"),
        (
            "rotation_position",
            "cast(least(rotation_position, 6) as string)",
            "and rotation_position is not null",
        ),
    ]
    parts = "\nunion all\n".join(
        f"""select '{name}' as grain, cast({expr} as string) as key,
                   count(*) as n,
                   avg(cast(label_arr_del15 as int64)) as p,
                   avg(label_arr_delay_minutes) as avg_min
            from base where true {extra}
            group by key"""
        for name, expr, extra in levels
    )
    return f"""
    with base as (
        select * from `{project}.{gold}.ml_flight_features`
        where flight_date < date('{cutoff}')
    ),
    glob as (
        select avg(cast(label_arr_del15 as int64)) as g_rate,
               avg(label_arr_delay_minutes) as g_avg
        from base
    ),
    raw as ({parts})
    select raw.grain, raw.key, raw.n,
           (raw.n * raw.p + {SMOOTHING_M} * glob.g_rate) / (raw.n + {SMOOTHING_M}) as rate,
           (raw.n * raw.avg_min + {SMOOTHING_M} * glob.g_avg) / (raw.n + {SMOOTHING_M}) as avg_min
    from raw cross join glob
    """


def fit_window_hist(bq, project: str, gold: str, cutoff: date) -> pd.DataFrame:
    return bq.query(_hist_sql(project, gold, cutoff.isoformat())).to_dataframe()


def verify_recompute(bq, project: str, gold: str) -> dict:
    """Run the recompute at the PRODUCTION cutoff and check it reproduces the
    mart's own hist_* values. If this fails, every number downstream is built on
    a formula nobody has validated, so the caller aborts."""
    mine = fit_window_hist(bq, project, gold, PROD_CUTOFF)
    theirs = bq.query(
        f"select entity_level as grain, entity_key as key, hist_arr_del15_rate as rate, "
        f"hist_avg_arr_delay_minutes as avg_min, hist_n_flights as n "
        f"from `{project}.{gold}.serving_entity_profile`"
    ).to_dataframe()
    m = mine.merge(theirs, on=["grain", "key"], suffixes=("_mine", "_mart"))
    d_rate = (m["rate_mine"] - m["rate_mart"]).abs()
    d_avg = (m["avg_min_mine"] - m["avg_min_mart"]).abs()
    return {
        "compared": int(len(m)),
        "max_abs_rate_diff": float(d_rate.max()),
        "max_abs_avg_diff": float(d_avg.max()),
        "n_exact_match": int((m["n_mine"] == m["n_mart"]).sum()),
    }


def apply_hist(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Overwrite the mart's 18 hist_* columns with the fit-window recompute."""
    band = np.where(
        ~df["has_inbound_leg"].fillna(0).astype(bool),
        "no_inbound",
        np.select(
            [
                df["sched_turnaround_min"] < 35,
                df["sched_turnaround_min"] < 60,
                df["sched_turnaround_min"] < 120,
            ],
            ["lt_35", "35_60", "60_120"],
            default="ge_120",
        ),
    )
    keys = {
        "route": df["route"].astype(str),
        "carrier": df["carrier"].astype(str),
        "origin": df["origin"].astype(str),
        "dest": df["dest"].astype(str),
        "turnaround_band": pd.Series(band, index=df.index),
        "rotation_position": df["rotation_position"]
        .clip(upper=6)
        .astype("Float64")
        .astype(str)
        .str.replace(r"\.0$", "", regex=True),
    }
    for grain in HIST_GRAINS:
        lk = hist[hist["grain"] == grain].set_index("key")
        for src, dst in (
            ("rate", "arr_del15_rate"),
            ("avg_min", "avg_arr_delay_minutes"),
            ("n", "n_flights"),
        ):
            col = f"hist_{grain}_{dst}"
            if col in df.columns:
                df[col] = keys[grain].map(lk[src]).astype("float32")
    return df


# --------------------------------------------------------------------------
# 2. data
# --------------------------------------------------------------------------
def load_training_window(bq, project: str, gold: str, limit: int | None = None) -> pd.DataFrame:
    cols = ", ".join([*f.FEATURES, "flight_date", *f.LABELS])
    log.info("loading the training window ...")
    df = (
        bq.query(f"select {cols} from `{project}.{gold}.ml_flight_features` where {f.SPLIT_COL}")
        .result()
        .to_dataframe(create_bqstorage_client=True)
    )
    for c in f.CATEGORICAL_FEATURES:
        df[c] = df[c].astype("category")
    for c in f.NUMERIC_FEATURES:
        df[c] = (
            df[c].astype("Float32").astype("float32")
            if str(df[c].dtype) == "boolean"
            else pd.to_numeric(df[c]).astype("float32")
        )
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    df["label_arr_del15"] = df["label_arr_del15"].astype("int8")
    log.info("loaded %s training rows", f"{len(df):,}")
    return df


# --------------------------------------------------------------------------
# 3. the five families
# --------------------------------------------------------------------------
def families() -> dict[str, list[dict]]:
    """Five learners NOT yet tried, with a small curated grid each.

    Chosen for a reason, not variety for its own sake:
      catboost      ordered target statistics are built for high-cardinality
                    categoricals — route has 7,539 levels, the single feature
                    XGBoost handles least naturally. Best prior odds.
      hist_gbdt     sklearn's histogram GBDT with native categorical support:
                    a different implementation of the same idea, which is the
                    honest way to ask "is the gain XGBoost-specific?".
      extra_trees   extremely randomised splits — a genuinely different
                    bias/variance trade-off rather than another boosted tree.
      random_forest bagged trees, the classic contrast to boosting.
      ensemble      rank-average blend of the per-family bests (built after the
                    others run, weights chosen on validation).
    """
    return {
        "catboost": [
            {"depth": d, "learning_rate": lr, "l2_leaf_reg": l2, "iterations": 600}
            for d, lr, l2 in [(6, 0.1, 3), (8, 0.1, 3), (8, 0.05, 6), (10, 0.05, 6), (6, 0.05, 1)]
        ],
        "hist_gbdt": [
            {"max_depth": d, "learning_rate": lr, "max_iter": 400, "min_samples_leaf": msl}
            for d, lr, msl in [(None, 0.1, 20), (8, 0.1, 20), (12, 0.05, 50), (None, 0.05, 100)]
        ],
        "extra_trees": [
            {"n_estimators": 300, "max_depth": d, "min_samples_leaf": msl, "max_features": mf}
            for d, msl, mf in [(20, 50, "sqrt"), (30, 20, "sqrt"), (None, 100, 0.3)]
        ],
        "random_forest": [
            {"n_estimators": 300, "max_depth": d, "min_samples_leaf": msl, "max_features": mf}
            for d, msl, mf in [(20, 50, "sqrt"), (30, 20, "sqrt"), (None, 100, 0.3)]
        ],
    }


def _prep(X: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Categoricals the way each library wants them."""
    X = X.copy()
    if kind == "catboost":
        for c in f.CATEGORICAL_FEATURES:
            X[c] = X[c].astype(str).fillna("__NA__")
    elif kind == "hist_gbdt":
        for c in f.CATEGORICAL_FEATURES:
            X[c] = X[c].cat.codes.astype("float32")  # -1 for missing
    else:  # forests cannot take NaN or categories
        for c in f.CATEGORICAL_FEATURES:
            X[c] = X[c].cat.codes.astype("float32")
        X = X.fillna(-999.0)
    return X


def build(kind: str, params: dict, spw: float):
    if kind == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            **params,
            random_seed=SEED,
            verbose=0,
            thread_count=-1,
            scale_pos_weight=spw,
            cat_features=list(f.CATEGORICAL_FEATURES),
        )
    if kind == "hist_gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            **params,
            random_state=SEED,
            class_weight="balanced",
            categorical_features=[f.FEATURES.index(c) for c in f.CATEGORICAL_FEATURES],
        )
    if kind == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            **params, random_state=SEED, n_jobs=-1, class_weight="balanced_subsample"
        )
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        **params, random_state=SEED, n_jobs=-1, class_weight="balanced_subsample"
    )


# --------------------------------------------------------------------------
# 4. the search
# --------------------------------------------------------------------------
def run(budget_min: int, fit_sample: int, val_sample: int, limit: int | None = None) -> dict:
    from google.cloud import bigquery

    OUT_DIR.mkdir(exist_ok=True)
    started = time.time()
    deadline = started + budget_min * 60
    project, gold = require_env("GCP_PROJECT_ID"), require_env("BQ_GOLD_DATASET")
    bq = bigquery.Client(project=project)

    log.info("verifying the hist_* recompute against the mart ...")
    check = verify_recompute(bq, project, gold)
    log.info("recompute check: %s", check)
    if check["max_abs_rate_diff"] > 1e-6 or check["compared"] < 8000:
        raise SystemExit(f"hist_* recompute does NOT reproduce the mart: {check} — aborting")

    df = load_training_window(bq, project, gold, limit)
    log.info("re-deriving hist_* on the fit window (< %s) ...", VAL_START)
    df = apply_hist(df, fit_window_hist(bq, project, gold, VAL_START))

    is_val = df["flight_date"].to_numpy() >= np.datetime64(VAL_START)
    fit_df, val_df = df[~is_val], df[is_val]
    assert fit_df["flight_date"].max() < val_df["flight_date"].min()
    log.info("fit %s rows | val %s rows", f"{len(fit_df):,}", f"{len(val_df):,}")

    rng = np.random.default_rng(SEED)
    fit_idx = rng.choice(len(fit_df), min(fit_sample, len(fit_df)), replace=False)
    val_idx = rng.choice(len(val_df), min(val_sample, len(val_df)), replace=False)
    Xf_all, yf = fit_df.iloc[fit_idx][list(f.FEATURES)], fit_df.iloc[fit_idx]["label_arr_del15"]
    Xv_all, yv = val_df.iloc[val_idx][list(f.FEATURES)], val_df.iloc[val_idx]["label_arr_del15"]
    spw = float((yf == 0).sum() / max((yf == 1).sum(), 1))
    log.info(
        "search on %s fit / %s val rows, scale_pos_weight %.3f",
        f"{len(Xf_all):,}",
        f"{len(Xv_all):,}",
        spw,
    )

    results: list[dict] = []
    val_scores: dict[str, np.ndarray] = {}
    meta = {
        "val_start": str(VAL_START),
        "recompute_check": check,
        "n_fit_search": int(len(Xf_all)),
        "n_val": int(len(Xv_all)),
        "scale_pos_weight": spw,
        "hist_rederived_on_fit_window": True,
        "test_set_touched": False,
    }

    def save():
        (OUT_DIR / "results.json").write_text(
            json.dumps({"meta": meta, "trials": results}, indent=2, default=str) + "\n"
        )

    for kind, grid in families().items():
        prepped = False
        for i, params in enumerate(grid):
            if time.time() > deadline:
                log.warning("budget exhausted — stopping before %s trial %d", kind, i)
                break
            if not prepped:
                Xf, Xv = _prep(Xf_all, kind), _prep(Xv_all, kind)
                prepped = True
            t0 = time.time()
            try:
                model = build(kind, params, spw)
                model.fit(Xf, yf)
                p = model.predict_proba(Xv)[:, 1]
                trial = {
                    "family": kind,
                    "params": params,
                    "val_roc_auc": float(roc_auc_score(yv, p)),
                    "val_pr_auc": float(average_precision_score(yv, p)),
                    "fit_seconds": round(time.time() - t0, 1),
                    "ok": True,
                }
                key = f"{kind}#{i}"
                if key not in val_scores or trial["val_pr_auc"] > max(
                    (t["val_pr_auc"] for t in results if t["family"] == kind), default=-1
                ):
                    val_scores[kind] = p  # keep the best-so-far scores for blending
            except Exception as exc:  # noqa: BLE001 — one family must not kill the run
                log.exception("%s trial %d failed", kind, i)
                trial = {"family": kind, "params": params, "ok": False, "error": str(exc)[:400]}
            results.append(trial)
            save()
            log.info(
                "%-14s %d/%d  PR-AUC %s  ROC %s  (%.0fs)",
                kind,
                i + 1,
                len(grid),
                f"{trial.get('val_pr_auc', float('nan')):.5f}",
                f"{trial.get('val_roc_auc', float('nan')):.5f}",
                trial.get("fit_seconds", 0),
            )
            tracking.log_run(
                run_name=f"search:{kind}:{i}",
                params={"family": kind, **{k: str(v) for k, v in params.items()}},
                metrics={k: v for k, v in trial.items() if isinstance(v, (int, float))},
                tags={
                    "stage": "model_search",
                    "selection_surface": "validation",
                    "hist_rederived_on_fit_window": "true",
                    "test_touched": "false",
                },
            )

    # ---- family 5: rank-average blend of the per-family bests ----
    if len(val_scores) >= 2:
        from scipy.stats import rankdata

        ranked = {k: rankdata(v) / len(v) for k, v in val_scores.items()}
        blend = np.mean(list(ranked.values()), axis=0)
        trial = {
            "family": "ensemble_rank_blend",
            "params": {"members": sorted(ranked)},
            "val_roc_auc": float(roc_auc_score(yv, blend)),
            "val_pr_auc": float(average_precision_score(yv, blend)),
            "ok": True,
        }
        results.append(trial)
        save()
        log.info("ensemble  PR-AUC %.5f  ROC %.5f", trial["val_pr_auc"], trial["val_roc_auc"])
        tracking.log_run(
            run_name="search:ensemble_rank_blend",
            params={"family": "ensemble", "members": ",".join(sorted(ranked))},
            metrics={"val_roc_auc": trial["val_roc_auc"], "val_pr_auc": trial["val_pr_auc"]},
            tags={
                "stage": "model_search",
                "selection_surface": "validation",
                "test_touched": "false",
            },
        )

    meta["elapsed_min"] = round((time.time() - started) / 60, 1)
    save()
    _report(meta, results)
    return {"meta": meta, "trials": results}


def _report(meta: dict, results: list[dict]) -> None:
    ok = [t for t in results if t.get("ok")]
    ok.sort(key=lambda t: -t["val_pr_auc"])
    lines = [
        "# Overnight model search — validation results",
        "",
        f"- fit rows searched on: **{meta['n_fit_search']:,}** · val rows: **{meta['n_val']:,}**",
        f"- validation window starts **{meta['val_start']}**",
        "- `hist_*` re-derived on the fit window (rule 10): **yes**",
        f"- recompute verified against the mart: max rate diff "
        f"**{meta['recompute_check']['max_abs_rate_diff']:.2e}** over "
        f"{meta['recompute_check']['compared']:,} entities",
        "- **the held-out test set was NOT touched** — this is a validation ranking, "
        "not an adoption",
        f"- elapsed: {meta.get('elapsed_min', '?')} min",
        "",
        "| rank | family | val PR-AUC | val ROC-AUC | params |",
        "|---|---|---|---|---|",
    ]
    for i, t in enumerate(ok[:20], 1):
        lines.append(
            f"| {i} | {t['family']} | {t['val_pr_auc']:.5f} | {t['val_roc_auc']:.5f} | "
            f"`{t['params']}` |"
        )
    failed = [t for t in results if not t.get("ok")]
    if failed:
        lines += ["", "## Failures", ""]
        lines += [f"- `{t['family']}` {t['params']}: {t.get('error', '')[:160]}" for t in failed]
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")
    log.info("wrote %s", OUT_DIR / "REPORT.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget-min", type=int, default=420, help="wall-clock budget")
    ap.add_argument("--fit-sample", type=int, default=2_500_000)
    ap.add_argument("--val-sample", type=int, default=800_000)
    ap.add_argument("--limit", type=int, default=None, help="cap loaded rows (smoke test)")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(args.budget_min, args.fit_sample, args.val_sample, args.limit)


if __name__ == "__main__":
    main()
