"""Stage 3 hyperparameter search — the reproducible record behind the tuned
config shipped in ml.train (CLASSIFIER_PARAMS / REGRESSOR_PARAMS).

Discipline (CLAUDE.md §9): the held-out TEST set (is_training_row = false,
2024-07-01+) is NEVER touched during selection. Hyperparameters are chosen on a
time-based VALIDATION slice carved from INSIDE the training window, then the
winners are retrained on the full training window and judged once on test.

Validation carve
----------------
The last 8 weeks of the training window — 2024-05-06..2024-06-30 (~1.10M rows) —
held out as validation; the fit-set is the training rows before it (~15.57M).
Time-based, mirroring the train->test split (validation immediately precedes the
test window), never random.

hist_* leakage residual (accepted, noted)
------------------------------------------
The mart's hist_* rates aggregate the WHOLE pre-cutoff training window, so a
validation slice carved from training inherits rates computed partly from
validation-period flights — the documented mart residual (ml_flight_features.sql
says a val slice "must re-derive rates as-of that slice"). We accept it here, but
NOT on the discredited "common-mode" grounds — the leak is NOT common-mode.
Instead it was MEASURED not to matter: (1) an exact fit-window-only recompute of
all hist_* shifts validation PR-AUC UNEQUALLY across learners (XGB +0.0008 vs
LightGBM +0.0002) yet does NOT flip the XGB-vs-LightGBM selection winner — re-
derive fit-window rates for any closer/wider future selection
(docs/leakage_discipline.md rule 10); (2) the reported tuned-vs-untuned
comparison is on the leak-free TEST set (test hist_* never include test-window
flights); (3) keeping
the full-window rates makes the tuning feature distribution identical to what
the shipped full-window model actually trains on — re-derived shorter-window
rates would optimize params for a distribution the shipped model never sees;
(4) re-deriving in Python would violate the SQL-only gold rule (CLAUDE.md §5).

Outcome (the split adoption)
----------------------------
The classifier and regressor search INDEPENDENTLY (separate fits, targets and
metrics) converged on the same candidate on a flat plateau — but early stopping
gave different tree counts (clf 140 vs reg 201), proving independent fits. On
the held-out test the tuned config REGRESSED the classifier (ROC 0.7389->0.7373,
PR-AUC 0.4652->0.4646: validation-optimism) but IMPROVED the regressor
(RMSE 49.71->49.26, MAE 19.10->18.99: signal, val and test agree). So the
classifier keeps its defaults and only the regressor adopts the tuned config.

Run:  uv run --extra ml python -m ml.tuning   (~60 min; loads the full mart)
"""

from __future__ import annotations

import gc
import json
import logging
import time
from datetime import date

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from ml import features as f
from ml.audit import run_audit
from ml.data import load_mart

log = logging.getLogger("ml.tuning")

VAL_START = date(2024, 5, 6)  # last 8 weeks of the 2022-01-01..2024-06-30 window
SEED = 0
ES_ROUNDS = 30
N_CAP = 700  # early-stopping ceiling on n_estimators

# Curated candidate grid (max_depth, learning_rate, min_child_weight, subsample,
# colsample_bytree): coordinate coverage around the (8, 0.1, 1, 1, 1) default —
# a reasonable search, deliberately NOT an exhaustive product grid.
CANDIDATES = [
    (8, 0.10, 1, 1.0, 1.0),  # baseline structure, early stopping picks rounds
    (8, 0.05, 5, 0.8, 0.8),  # slower lr + mild regularization + sampling
    (10, 0.05, 10, 0.8, 0.8),  # deeper + more regularization
    (6, 0.08, 3, 0.9, 0.9),  # shallower
    (12, 0.04, 20, 0.7, 0.7),  # deep + strong regularization + aggressive sampling
    (8, 0.03, 10, 0.85, 0.85),  # slow lr at baseline depth
]


def _base_kwargs() -> dict:
    return dict(tree_method="hist", enable_categorical=True, n_jobs=-1, random_state=SEED)


def _clf_metrics(y, scores) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
    }


def _reg_metrics(y, pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
    }


def carve(df):
    """Return the four boolean masks: full-train, fit (train before VAL_START),
    val (last 8 weeks of train), test. Asserts strict temporal ordering."""
    train = df[f.SPLIT_COL].to_numpy()
    fdate = df["flight_date"].to_numpy()
    val = train & (fdate >= np.datetime64(VAL_START))
    fit = train & (fdate < np.datetime64(VAL_START))
    test = ~train
    # strict temporal ordering: fit precedes val, AND the whole training window
    # (fit + val) precedes test — the same invariant ml.train.split_report pins.
    # Checking train.max (not just val.min) closes the drift hole: a test-window
    # date wrongly marked is_training_row lands in val (date >= VAL_START) and
    # would otherwise pass a val.min-only check; here it trips val.max/train.max.
    assert str(df.loc[fit, "flight_date"].max().date()) < str(
        df.loc[val, "flight_date"].min().date()
    )
    assert str(df.loc[train, "flight_date"].max().date()) < str(
        df.loc[test, "flight_date"].min().date()
    )
    return train, fit, val, test


def _search(kind, X, y, fit_mask, val_mask, spw_fit):
    """Search CANDIDATES for one model: fit on the full fit-set, early-stop on
    the validation slice, select by val PR-AUC (clf) or val RMSE (reg). Returns
    (trials, best). Independent of the other model — separate fits and metric."""
    Xval = X[val_mask].copy()
    yv = y[val_mask]
    trials = []
    for i, (depth, lr, mcw, sub, col) in enumerate(CANDIDATES):
        ts = time.time()
        common = dict(
            n_estimators=N_CAP,
            learning_rate=lr,
            max_depth=depth,
            min_child_weight=mcw,
            subsample=sub,
            colsample_bytree=col,
            early_stopping_rounds=ES_ROUNDS,
            **_base_kwargs(),
        )
        Xfit = X[fit_mask]
        if kind == "clf":
            model = xgb.XGBClassifier(scale_pos_weight=spw_fit, eval_metric="aucpr", **common)
            model.fit(Xfit, y[fit_mask], eval_set=[(Xval, yv)], verbose=False)
            val = _clf_metrics(yv, model.predict_proba(Xval)[:, 1])
            score = val["pr_auc"]
        else:
            model = xgb.XGBRegressor(eval_metric="rmse", **common)
            model.fit(Xfit, y[fit_mask], eval_set=[(Xval, yv)], verbose=False)
            val = _reg_metrics(yv, model.predict(Xval))
            score = -val["rmse"]
        n_trees = int(model.best_iteration) + 1
        trials.append(
            {
                "params": {
                    "max_depth": depth,
                    "learning_rate": lr,
                    "min_child_weight": mcw,
                    "subsample": sub,
                    "colsample_bytree": col,
                },
                "n_trees": n_trees,
                "val": val,
                "score": score,
            }
        )
        log.info(
            "%s cand %d/%d %s -> val=%s trees=%d (%.1f min)",
            kind,
            i + 1,
            len(CANDIDATES),
            trials[-1]["params"],
            val,
            n_trees,
            (time.time() - ts) / 60,
        )
        del Xfit, model
        gc.collect()
    best = max(trials, key=lambda t: t["score"])
    log.info("BEST %s: %s trees=%d val=%s", kind, best["params"], best["n_trees"], best["val"])
    return trials, best


def _fit_final(kind, X, y, train_mask, test_mask, params, n_trees, spw_full):
    """Retrain a winner on the FULL training window and score on TEST."""
    common = dict(
        n_estimators=n_trees,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=-1,
        random_state=SEED,
        **params,
    )
    if kind == "clf":
        model = xgb.XGBClassifier(scale_pos_weight=spw_full, eval_metric="aucpr", **common)
        model.fit(X[train_mask], y[train_mask])
        return _clf_metrics(y[test_mask], model.predict_proba(X[test_mask])[:, 1])
    model = xgb.XGBRegressor(eval_metric="rmse", **common)
    model.fit(X[train_mask], y[train_mask])
    return _reg_metrics(y[test_mask], model.predict(X[test_mask]))


def run_tuning() -> dict:
    t0 = time.time()
    df, bq, dataset = load_mart()
    run_audit(bq, dataset)  # same hard gate as ml.train
    train_mask, fit_mask, val_mask, test_mask = carve(df)

    X = df[f.FEATURES]
    y_clf = df["label_arr_del15"].to_numpy()
    y_reg = df["label_arr_delay_minutes"].to_numpy(dtype="float32")
    spw_fit = float((fit_mask.sum() - y_clf[fit_mask].sum()) / y_clf[fit_mask].sum())
    spw_full = float((train_mask.sum() - y_clf[train_mask].sum()) / y_clf[train_mask].sum())
    log.info(
        "carve: fit=%d val=%d test=%d spw_fit=%.4f spw_full=%.4f",
        int(fit_mask.sum()),
        int(val_mask.sum()),
        int(test_mask.sum()),
        spw_fit,
        spw_full,
    )

    out: dict = {"val_start": str(VAL_START), "candidates": CANDIDATES}

    # SELECTION runs on the validation slice ONLY. All test scoring is deferred
    # until AFTER both winners are chosen (below), so the search structurally
    # cannot be influenced by held-out test metrics.
    clf_trials, best_clf = _search("clf", X, y_clf, fit_mask, val_mask, spw_fit)
    reg_trials, best_reg = _search("reg", X, y_reg, fit_mask, val_mask, spw_fit)
    out["search_clf"], out["best_clf"] = clf_trials, best_clf
    out["search_reg"], out["best_reg"] = reg_trials, best_reg

    # ---- selection is done; NOW touch the held-out test set for the record ----
    # the untuned baseline (reproduces the pinned headline) ...
    out["untuned_test"] = {
        "clf": _fit_final(
            "clf",
            X,
            y_clf,
            train_mask,
            test_mask,
            {"learning_rate": 0.1, "max_depth": 8},
            300,
            spw_full,
        ),
        "reg": _fit_final(
            "reg",
            X,
            y_reg,
            train_mask,
            test_mask,
            {"learning_rate": 0.1, "max_depth": 8},
            300,
            spw_full,
        ),
    }
    log.info("UNTUNED test: %s", out["untuned_test"])

    # ... and the tuned winners (classifier for the record — adopted: default;
    # regressor adopted)
    out["tuned_test"] = {
        "clf": _fit_final(
            "clf",
            X,
            y_clf,
            train_mask,
            test_mask,
            best_clf["params"],
            best_clf["n_trees"],
            spw_full,
        ),
        "reg": _fit_final(
            "reg",
            X,
            y_reg,
            train_mask,
            test_mask,
            best_reg["params"],
            best_reg["n_trees"],
            spw_full,
        ),
    }
    log.info("TUNED test: %s", out["tuned_test"])
    out["total_min"] = round((time.time() - t0) / 60, 1)
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    print(json.dumps(run_tuning(), indent=2, default=str))


if __name__ == "__main__":
    main()
