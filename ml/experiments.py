"""Model-comparison harness — 'try different models', tracked with MLflow.

Fits alternative CLASSIFIERS on the SAME leak-free features and the SAME
time-based split as the production pipeline (``ml.train``) and logs each to
MLflow so they compare apples-to-apples. First alternative: LightGBM vs the
XGBoost baseline.

Only the LEARNER changes — identical ``is_training_row`` split, identical
``FEATURES``, identical pre-departure boundary (CLAUDE.md §9). This is where you
try new models; the SHIPPED classifier stays ``ml.train``'s XGBoost until an
alternative genuinely BEATS it on the held-out TEST set and is adopted
deliberately. Selection discipline still applies: pick on a validation slice
(as Stage 3 did), never by re-selecting against the test set.

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


def fit_xgboost(X, y, train, test, spw) -> dict:
    """The production baseline learner (same config as ml.train's classifier)."""
    model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        **CLASSIFIER_PARAMS,
        tree_method="hist",
        enable_categorical=True,
        scale_pos_weight=spw,
        n_jobs=-1,
        eval_metric="aucpr",
    )
    model.fit(X[train], y[train])
    return _clf_metrics(y[test], model.predict_proba(X[test])[:, 1])


def fit_lightgbm(X, y, train, test, spw) -> dict:
    """LightGBM on the same features (native categoricals, same lr/rounds/depth).
    Leaf-wise growth differs from XGBoost's level-wise, so this compares model
    FAMILIES at matched lr/rounds/depth, not identical trees. deterministic +
    force_row_wise for reproducible fits; NaNs handled natively like XGBoost."""
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
    model.fit(X[train], y[train], categorical_feature=list(f.CATEGORICAL_FEATURES))
    return _clf_metrics(y[test], model.predict_proba(X[test])[:, 1])


CANDIDATES = {"xgboost": fit_xgboost, "lightgbm": fit_lightgbm}


def compare_classifiers() -> dict:
    """Fit every candidate on the identical split/features, log each to MLflow,
    and print a held-out-test comparison table."""
    t0 = time.time()
    df, bq, dataset = load_mart()
    # same guards as ml.train / ml.tuning: the leakage self-audit is a HARD GATE,
    # and split_report asserts the is_training_row partition is clean and
    # time-ordered (train_date_max < test_date_min) — a comparison that claims
    # the production boundary must enforce it, not just assume it.
    run_audit(bq, dataset)
    log.info("split: %s", split_report(df))
    train = df[f.SPLIT_COL].to_numpy()
    test = ~train
    X = df[f.FEATURES]
    y = df["label_arr_del15"].to_numpy()
    spw = float((train.sum() - y[train].sum()) / y[train].sum())
    log.info("comparison: n_train=%d n_test=%d spw=%.4f", int(train.sum()), int(test.sum()), spw)

    results: dict = {}
    for name, fit in CANDIDATES.items():
        ts = time.time()
        metrics = fit(X, y, train, test, spw)
        results[name] = metrics
        log.info("%-9s %s (%.1f min)", name, metrics, (time.time() - ts) / 60)
        tracking.log_run(
            run_name=f"classifier-{name}",
            params={
                "model": name,
                "n_estimators": N_ESTIMATORS,
                "learning_rate": LEARNING_RATE,
                "max_depth": MAX_DEPTH,
                "scale_pos_weight": round(spw, 4),
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
            },
            metrics=metrics,
            tags={
                "comparison": "classifier",
                "model": name,
                "split": "time-based",
                "leakage_boundary": "pre-departure",
            },
        )

    print("\n===== CLASSIFIER COMPARISON (held-out test; same split/features) =====")
    print(f"{'model':10s} {'roc_auc':>10s} {'pr_auc':>10s} {'accuracy':>10s}")
    for name, m in results.items():
        roc, pr, acc = m["clf_roc_auc"], m["clf_pr_auc"], m["clf_accuracy"]
        print(f"{name:10s} {roc:>10.6f} {pr:>10.6f} {acc:>10.4f}")
    mins = (time.time() - t0) / 60
    print(f"\ntotal {mins:.1f} min  (runs logged to MLflow: {tracking.enabled()})")
    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    compare_classifiers()


if __name__ == "__main__":
    main()
