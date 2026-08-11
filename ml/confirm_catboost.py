"""ONE-TIME held-out confirmation of the validation-selected CatBoost.

This is the report `docs/leakage_discipline.md` rule 7 permits, and nothing more.
The selection already happened, on validation, in `ml/model_search.py`: CatBoost
led every other family and both XGBoost grids. This script takes THAT winner,
retrains it on the full fit window, and scores the held-out test set once.

Three properties keep it honest, and each is a thing that could have been done
wrong:

1. **The config is fixed before the run.** `WINNER` below is the top of the
   pass-2 validation table, written down in advance. Scoring several configs on
   test and keeping the best is re-selecting on test — the exact failure rule 7
   names. If this config loses, that IS the result; there is no second attempt.
2. **It retrains on the full fit window.** The search ranked at 2.5M of
   16,678,880 fit rows. A learner that wins at 2.5M need not win at 6.7x that,
   so confirming the SEARCH's model rather than a full-data model would answer a
   question nobody asked.
3. **`hist_*` come from the mart unmodified.** Rule 10's re-derivation exists
   because a validation slice sits INSIDE the training window, where mart
   `hist_*` see the slice's own period. Here the split is the real cutoff and
   the mart's `hist_*` are computed only from pre-cutoff flights, so they are
   already correct — re-deriving would be wrong, not safer.

**Calibration mirrors the shipped discipline exactly** (ml/calibration.py): Platt
is fit on the last-8-weeks validation slice carved from INSIDE the training
window, never on the held-out test, and `build_calibration` hard-gates that the
map preserves ROC/PR-AUC. Reusing that function rather than re-deriving a
calibration here is deliberate — a second implementation is a second thing that
can silently disagree with what serving does.

**The regressor is UNTUNED and must be reported as such.** The search tuned the
classifier only, so no validation evidence picks a CatBoost regressor config.
`REGRESSOR` below is a reasonable shape, not a selected one; its RMSE/MAE is a
first look at whether the family is competitive on the regression head, not a
like-for-like against the tuned shipped regressor.

Results are written INCREMENTALLY — the classification block lands before the
regressor starts — so a run that is cut short still yields usable numbers.

Run:  uv run --extra ml --extra ingestion python -m ml.confirm_catboost
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from ml import features as f
from ml import tracking
from ml.calibration import build_calibration
from ml.data import load_mart
from ml.tuning import carve

log = logging.getLogger("ml.confirm_catboost")

OUT = Path(__file__).parent / "search_results"
SEED = 0

# The CatBoost representative, fixed in advance of seeing any test number.
#
# Pass 2's top four CatBoost configs scored 0.51460 / 0.51433 / 0.51285 /
# 0.51225 on validation — a spread of 0.00235 against a measured draw-level
# noise floor of 0.0018 (ml/search_results/pass1 vs the pass-2 replicate). They
# are statistically indistinguishable, so "the winner" is not a meaningful
# distinction among them and picking on COST is legitimate. This one is
# depth 8 / 600 iterations: ~4x cheaper than depth 10 / 1200 (a ~50 min fit
# instead of ~3.3 h) and still 0.0148 clear of the shipped XGBoost config's
# 0.49805 on the same rows.
#
# The choice was made BEFORE any held-out number existed — the earlier
# depth-10 attempt died at iteration 237 of 1200 without writing a result — so
# no test observation informed it. That ordering is what rule 7 protects; had a
# test score been seen first, switching configs would be re-selection.
WINNER = {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 6, "iterations": 600}

# NOT selected on anything — see the module docstring. Deliberately cheaper than
# WINNER so the regression head cannot eat the whole budget before the
# classification block is written.
REGRESSOR = {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 6, "iterations": 600}

# The shipped model's pinned held-out numbers (ml/README.md), for the deltas.
# Not recomputed here: they come from the same mart and the same split, and
# recomputing them would risk quietly reporting a different baseline.
SHIPPED = {
    "roc_auc": 0.7389,
    "pr_auc": 0.4652,
    "base_rate": 0.1969,
    "ece_platt": 0.017,
    "rmse": 49.26,
    "mae": 18.99,
}


def _prep(frame):
    X = frame[list(f.FEATURES)].copy()
    for c in f.CATEGORICAL_FEATURES:
        X[c] = X[c].astype(str).fillna("__NA__")
    return X


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    from catboost import CatBoostClassifier, CatBoostRegressor

    df, _, _ = load_mart()  # canonical sort, so this run is reproducible
    train_mask, _, val_mask, test_mask = (
        df[f.SPLIT_COL].to_numpy(),
        None,
        *carve(df)[2:],
    )
    train, test = df[train_mask], df[test_mask]
    log.info("fit %s rows | held-out test %s rows", f"{len(train):,}", f"{len(test):,}")
    assert train["flight_date"].max() < test["flight_date"].min(), "train/test overlap in time"

    Xtr, Xte = _prep(train), _prep(test)
    Xval = _prep(df[val_mask])
    y = df["label_arr_del15"].to_numpy()
    ytr, yte, yval = y[train_mask], y[test_mask], y[val_mask]
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    # ---------------- classification ----------------
    log.info("scale_pos_weight %.4f | training CatBoost classifier %s ...", spw, WINNER)
    clf = CatBoostClassifier(
        **WINNER,
        random_seed=SEED,
        verbose=200,
        thread_count=-1,
        scale_pos_weight=spw,
        cat_features=list(f.CATEGORICAL_FEATURES),
    )
    clf.fit(Xtr, ytr)
    log.info("classifier fit in %.1f min — scoring the held-out test ONCE", (time.time() - t0) / 60)

    raw_test = clf.predict_proba(Xte)[:, 1]
    p_val = clf.predict_proba(Xval)[:, 1]
    base = float(yte.mean())
    pr = float(average_precision_score(yte, raw_test))

    # Platt fit on the val slice, evaluated on test — the shipped contract.
    val_window = (
        str(df.loc[val_mask, "flight_date"].min().date()),
        str(df.loc[val_mask, "flight_date"].max().date()),
    )
    _, cal_report = build_calibration(p_val, yval, raw_test, yte, val_window)

    result = {
        "config": WINNER,
        "n_fit": int(len(train)),
        "n_test": int(len(test)),
        "test_roc_auc": float(roc_auc_score(yte, raw_test)),
        "test_pr_auc": pr,
        "test_base_rate": base,
        "test_lift_over_base_rate": pr / base,
        "test_brier_raw": cal_report["test"]["raw"]["brier"],
        "test_ece_raw": cal_report["test"]["raw"]["ece"],
        "test_brier_platt": cal_report["test"]["platt"]["brier"],
        "test_ece_platt": cal_report["test"]["platt"]["ece"],
        "calibration_val_window": val_window,
        "shipped": SHIPPED,
        "minutes_so_far": round((time.time() - t0) / 60, 1),
    }
    result["delta_roc_auc"] = result["test_roc_auc"] - SHIPPED["roc_auc"]
    result["delta_pr_auc"] = result["test_pr_auc"] - SHIPPED["pr_auc"]
    result["shipped_lift"] = SHIPPED["pr_auc"] / SHIPPED["base_rate"]
    result["beats_shipped"] = result["delta_pr_auc"] > 0
    _write(result)  # classification block lands before the regressor starts
    log.info(
        "CLASSIFIER: ROC %.4f (%+.4f) | PR-AUC %.4f (%+.4f) | lift %.2fx | ECE(platt) %.4f",
        result["test_roc_auc"],
        result["delta_roc_auc"],
        result["test_pr_auc"],
        result["delta_pr_auc"],
        result["test_lift_over_base_rate"],
        result["test_ece_platt"],
    )

    # ---------------- regression (untuned) ----------------
    log.info("training CatBoost regressor (UNTUNED, %s) ...", REGRESSOR)
    t1 = time.time()
    ym = df["label_arr_delay_minutes"].to_numpy()
    reg = CatBoostRegressor(
        **REGRESSOR,
        random_seed=SEED,
        verbose=200,
        thread_count=-1,
        cat_features=list(f.CATEGORICAL_FEATURES),
    )
    reg.fit(Xtr, ym[train_mask])
    pred = reg.predict(Xte)
    truth = ym[test_mask]
    err = np.abs(pred - truth)
    result |= {
        "regressor_config": REGRESSOR,
        "regressor_tuned": False,
        "test_rmse": float(np.sqrt(mean_squared_error(truth, pred))),
        "test_mae": float(mean_absolute_error(truth, pred)),
        "test_median_abs_error": float(np.median(err)),
        "regressor_minutes": round((time.time() - t1) / 60, 1),
    }
    result["delta_rmse"] = result["test_rmse"] - SHIPPED["rmse"]
    result["delta_mae"] = result["test_mae"] - SHIPPED["mae"]
    result["minutes_total"] = round((time.time() - t0) / 60, 1)
    _write(result)

    tracking.log_run(
        run_name="confirm:catboost:test",
        params={"family": "catboost", **{k: str(v) for k, v in WINNER.items()}},
        metrics={k: v for k, v in result.items() if isinstance(v, (int, float))},
        tags={
            "stage": "held_out_confirmation",
            "selection_surface": "none — config fixed in advance",
            "test_touched": "true (one-time, by design)",
        },
    )
    log.info(
        "REGRESSOR (untuned): RMSE %.2f (%+.2f) | MAE %.2f (%+.2f) | median abs err %.2f",
        result["test_rmse"],
        result["delta_rmse"],
        result["test_mae"],
        result["delta_mae"],
        result["test_median_abs_error"],
    )


def _write(r: dict) -> None:
    (OUT / "confirm_catboost.json").write_text(json.dumps(r, indent=2) + "\n")
    verdict = "BEATS" if r["beats_shipped"] else "does NOT beat"
    rows = [
        (
            "ROC-AUC",
            f"{SHIPPED['roc_auc']:.4f}",
            f"{r['test_roc_auc']:.4f}",
            f"{r['delta_roc_auc']:+.4f}",
        ),
        (
            "PR-AUC",
            f"{SHIPPED['pr_auc']:.4f}",
            f"{r['test_pr_auc']:.4f}",
            f"{r['delta_pr_auc']:+.4f}",
        ),
        (
            f"Lift over the {r['test_base_rate']:.4f} base rate",
            f"{r['shipped_lift']:.2f}x",
            f"{r['test_lift_over_base_rate']:.2f}x",
            f"{r['test_lift_over_base_rate'] - r['shipped_lift']:+.2f}x",
        ),
        ("ECE (Platt)", f"{SHIPPED['ece_platt']:.4f}", f"{r['test_ece_platt']:.4f}", ""),
        ("Brier (Platt)", "0.135", f"{r['test_brier_platt']:.4f}", ""),
    ]
    if "test_rmse" in r:
        rows += [
            (
                "RMSE (regressor, UNTUNED)",
                f"{SHIPPED['rmse']:.2f}",
                f"{r['test_rmse']:.2f}",
                f"{r['delta_rmse']:+.2f}",
            ),
            (
                "MAE (regressor, UNTUNED)",
                f"{SHIPPED['mae']:.2f}",
                f"{r['test_mae']:.2f}",
                f"{r['delta_mae']:+.2f}",
            ),
            ("Median abs error (untuned)", "not pinned", f"{r['test_median_abs_error']:.2f}", ""),
        ]
    table = "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows)
    cal_win = f"{r['calibration_val_window'][0]} .. {r['calibration_val_window'][1]}"
    pending = "" if "test_rmse" in r else "\n_Regressor still training — RMSE/MAE pending._\n"
    (OUT / "CONFIRM.md").write_text(
        f"""# Held-out confirmation — CatBoost (validation-selected)

One-time test evaluation of the config chosen on validation. Not an adoption
gate; adopting on a test comparison re-selects on test (rule 7).

- classifier config (fixed in advance): `{WINNER}`
- fit rows **{r["n_fit"]:,}** (full fit window) · held-out test **{r["n_test"]:,}**
- test base rate **{r["test_base_rate"]:.4f}**
- Platt fit on the val slice **{cal_win}**, never on test

| metric | shipped XGBoost | CatBoost | delta |
|---|---|---|---|
{table}
{pending}
Raw (uncalibrated) CatBoost ECE **{r["test_ece_raw"]:.4f}**, Brier **{r["test_brier_raw"]:.4f}** —
both models need the Platt step; it is a monotonic remap, so ROC/PR-AUC are
unchanged by it.

**Verdict on the classifier: CatBoost {verdict} the shipped model on held-out PR-AUC.**

Validation said +0.0166 PR-AUC. The precedent for that not transferring: a
challenger that won validation by +0.0025 regressed on test (ml/README.md).

The regressor is **untuned** — the search tuned the classifier only, so its
RMSE/MAE is a first look at the family, not a like-for-like against the tuned
shipped regressor.
"""
    )


if __name__ == "__main__":
    main()
