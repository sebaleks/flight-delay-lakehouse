"""Model-comparison harness — 'try different models', tracked with MLflow.

Compares alternative CLASSIFIERS on the SAME leak-free features and the SAME
time-based split as the production pipeline (``ml.train``), tracked in MLflow.
First alternative: LightGBM vs the XGBoost baseline.

The workflow is **compare on validation, confirm the winner on test** — the
same discipline as Stage 3 (``ml.tuning``). Each candidate is fit on the fit-set
and scored on a validation slice carved from INSIDE the training window; the
winner is retrained on the full training window and scored ONCE on the held-out
test. The harness never SELECTS on the test set, so its held-out numbers are not
optimistic. Only the LEARNER changes — identical ``is_training_row`` split,
identical ``FEATURES``, identical pre-departure boundary (CLAUDE.md §9), and the
same leakage self-audit hard-gate ``ml.train`` runs. The SHIPPED classifier
stays ``ml.train``'s XGBoost until a challenger genuinely BEATS it on test and
is adopted deliberately.

Reality check (see ml/README + blog_material.md ch. 5): the classifier sits on
a flat plateau — Stage 3's six tuned configs spanned val PR-AUC 0.514–0.518 and
the tuned candidate regressed on test. 0.7389 / 0.4652 is a signal ceiling of
leak-free pre-departure features, not a model-capacity limit, so expect a
model-family swap to reshuffle the last ~0.002, not to jump the number. The
levers that move it are new leak-free features or a different (real-time)
regime — not a bigger learner.

Run:  uv run --extra ml python -m ml.experiments
"""

from __future__ import annotations

import logging
import time

import xgboost as xgb
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from ml import features as f
from ml import tracking
from ml.audit import run_audit
from ml.data import load_mart
from ml.train import CLASSIFIER_PARAMS, split_report
from ml.tuning import carve

log = logging.getLogger("ml.experiments")
SEED = 0
N_ESTIMATORS = 300
LEARNING_RATE = 0.1
MAX_DEPTH = 8


def _clf_metrics(y, scores) -> dict:
    return {
        "clf_roc_auc": float(roc_auc_score(y, scores)),
        "clf_pr_auc": float(average_precision_score(y, scores)),
        "clf_accuracy": float(accuracy_score(y, scores >= 0.5)),
    }


def fit_xgboost(X, y, fit_mask, eval_mask, spw) -> dict:
    """The production baseline learner (same config as ml.train's classifier).
    Fits on fit_mask, scores on eval_mask — the caller decides whether that is
    (fit-set -> validation) for selection or (full-train -> test) for confirmation."""
    model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        **CLASSIFIER_PARAMS,
        tree_method="hist",
        enable_categorical=True,
        scale_pos_weight=spw,
        n_jobs=-1,
        eval_metric="aucpr",
    )
    model.fit(X[fit_mask], y[fit_mask])
    return _clf_metrics(y[eval_mask], model.predict_proba(X[eval_mask])[:, 1])


def fit_lightgbm(X, y, fit_mask, eval_mask, spw) -> dict:
    """LightGBM on the same features (native categoricals, same lr/rounds/depth).
    Leaf-wise growth differs from XGBoost's level-wise, so this compares model
    FAMILIES at matched lr/rounds/depth, not identical trees. deterministic +
    force_row_wise for reproducible fits; NaNs handled natively like XGBoost.
    Fits on fit_mask, scores on eval_mask (see fit_xgboost)."""
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        num_leaves=255,  # < 2**max_depth; leaf-wise capacity ~ depth-8
        objective="binary",
        scale_pos_weight=spw,
        random_state=SEED,
        deterministic=True,
        force_row_wise=True,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X[fit_mask], y[fit_mask], categorical_feature=list(f.CATEGORICAL_FEATURES))
    return _clf_metrics(y[eval_mask], model.predict_proba(X[eval_mask])[:, 1])


CANDIDATES = {"xgboost": fit_xgboost, "lightgbm": fit_lightgbm}


def compare_classifiers() -> dict:
    """SELECT a classifier family on a time-based VALIDATION slice, then confirm
    the winner ONCE on the held-out test — never selecting on test (the same
    discipline ml.tuning uses). Every fit is logged to MLflow.

    Each candidate is fit on the fit-set (train before VAL_START) and scored on
    the validation slice (last 8 weeks of train); the winner (max val PR-AUC) is
    retrained on the FULL training window and scored once on test. Because the
    summer val slice has a higher delay base rate (~0.260) than test (~0.197),
    val PR-AUC is not comparable in absolute terms to test PR-AUC — only the
    RELATIVE ranking within the validation slice drives selection."""
    t0 = time.time()
    df, bq, dataset = load_mart()
    # same guards as ml.train / ml.tuning: the leakage self-audit is a HARD GATE,
    # and split_report asserts the is_training_row partition is clean and
    # time-ordered (train_date_max < test_date_min) — a comparison that claims
    # the production boundary must enforce it, not just assume it.
    run_audit(bq, dataset)
    log.info("split: %s", split_report(df))
    # carve: full-train, fit (train before VAL_START), val (last 8 weeks of
    # train), test — the exact slice ml.tuning selects on.
    train, fit, val, test = carve(df)
    X = df[f.FEATURES]
    y = df["label_arr_del15"].to_numpy()
    spw_fit = float((fit.sum() - y[fit].sum()) / y[fit].sum())
    spw_full = float((train.sum() - y[train].sum()) / y[train].sum())
    log.info(
        "carve: fit=%d val=%d test=%d spw_fit=%.4f spw_full=%.4f",
        int(fit.sum()),
        int(val.sum()),
        int(test.sum()),
        spw_fit,
        spw_full,
    )

    logged: list[bool] = []  # ACTUAL MLflow success, not just the env switch

    # ---- SELECTION: fit on the FIT set, score on VALIDATION (never test) ----
    val_results: dict = {}
    for name, fitfn in CANDIDATES.items():
        ts = time.time()
        m = fitfn(X, y, fit, val, spw_fit)
        val_results[name] = m
        log.info("val %-9s %s (%.1f min)", name, m, (time.time() - ts) / 60)
        logged.append(
            tracking.log_run(
                run_name=f"select-{name}",
                params={
                    "model": name,
                    "n_estimators": N_ESTIMATORS,
                    "learning_rate": LEARNING_RATE,
                    "max_depth": MAX_DEPTH,
                    "scale_pos_weight": round(spw_fit, 4),
                    "n_fit": int(fit.sum()),
                    "n_val": int(val.sum()),
                },
                metrics=m,
                tags={
                    "comparison": "classifier",
                    "stage": "validation",
                    "model": name,
                    "split": "time-based",
                    "leakage_boundary": "pre-departure",
                },
            )
        )

    winner = max(val_results, key=lambda n: val_results[n]["clf_pr_auc"])

    # ---- CONFIRMATION: retrain the winner on FULL train, score ONCE on test ----
    test_metrics = CANDIDATES[winner](X, y, train, test, spw_full)
    logged.append(
        tracking.log_run(
            run_name=f"test-{winner}",
            params={
                "model": winner,
                "n_estimators": N_ESTIMATORS,
                "learning_rate": LEARNING_RATE,
                "max_depth": MAX_DEPTH,
                "scale_pos_weight": round(spw_full, 4),
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
            },
            metrics=test_metrics,
            tags={
                "comparison": "classifier",
                "stage": "test-confirmation",
                "model": winner,
                "split": "time-based",
                "leakage_boundary": "pre-departure",
            },
        )
    )

    print("\n===== CLASSIFIER SELECTION (validation slice; winner picked HERE) =====")
    print(f"{'model':10s} {'roc_auc':>10s} {'pr_auc':>10s} {'accuracy':>10s}")
    for name, m in val_results.items():
        star = "  <- winner" if name == winner else ""
        print(
            f"{name:10s} {m['clf_roc_auc']:>10.6f} {m['clf_pr_auc']:>10.6f} "
            f"{m['clf_accuracy']:>10.4f}{star}"
        )
    print(f"\nrecommendation: ship '{winner}' — it won the VALIDATION selection above.")
    print(
        "held-out TEST (a ONE-TIME confirmation, NOT a selection/adoption gate): "
        f"roc={test_metrics['clf_roc_auc']:.6f} pr={test_metrics['clf_pr_auc']:.6f} "
        f"acc={test_metrics['clf_accuracy']:.4f}"
    )
    if winner == "xgboost":
        print("this re-anchors the shipped XGBoost headline (roc 0.7389 / pr 0.4652).")
    else:
        print(
            f"'{winner}' beat XGBoost on VALIDATION. Do NOT adopt it merely because a "
            "test number beats 0.7389 / 0.4652 — that re-selects on test. To ship it, "
            "wire it into ml.train and re-run this validation selection with it included."
        )
    mins = (time.time() - t0) / 60
    n_ok = sum(logged)
    print(f"\ntotal {mins:.1f} min  (MLflow runs logged: {n_ok}/{len(logged)})")
    return {"validation": val_results, "winner": winner, "test": test_metrics}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    compare_classifiers()


if __name__ == "__main__":
    main()
