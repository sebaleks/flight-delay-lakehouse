"""Train and evaluate the two flight-delay models (CLAUDE.md §9).

Reads gold ``ml_flight_features`` from BigQuery — the same gold layer the
dashboard consumes, no duplicated data or logic — runs the leakage self-audit
(hard gate), splits STRICTLY on the mart's ``is_training_row`` column (never
re-derived from dates, never shuffled across the boundary), trains:

* classification of ``label_arr_del15``:
  - logistic-regression baseline (class_weight='balanced'; numeric features +
    one-hot carrier; origin/dest/route identity enters via the hist_* rates)
  - XGBoost (native categoricals incl. origin/dest/route;
    scale_pos_weight = neg/pos from the TRAIN set)
* regression of ``label_arr_delay_minutes``: XGBoost, vs a predict-train-mean
  baseline.

All metrics are computed on the held-out ``is_training_row = false`` rows
only. Artifacts land in ml/artifacts/<run>/ (git-ignored).

Run:  uv run --extra ml python -m ml.train
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml import features as f
from ml.audit import run_audit
from ml.data import load_mart

log = logging.getLogger("ml.train")

ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"


def split_report(df: pd.DataFrame) -> dict:
    """Split strictly on is_training_row; prove the partition is clean."""
    train_mask = df[f.SPLIT_COL].to_numpy()
    test_mask = ~train_mask
    n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
    overlap = int((train_mask & test_mask).sum())  # complementary masks: 0
    report = {
        "n_train": n_train,
        "n_test": n_test,
        "n_total": int(len(df)),
        "partition_sums_to_total": bool(n_train + n_test == len(df)),
        "rows_in_both": overlap,
        "train_date_max": str(df.loc[train_mask, "flight_date"].max().date()),
        "test_date_min": str(df.loc[test_mask, "flight_date"].min().date()),
    }
    assert overlap == 0 and report["partition_sums_to_total"]
    assert report["train_date_max"] < report["test_date_min"]
    return report


LOGREG_INPUT_COLUMNS = [*f.NUMERIC_FEATURES, "carrier"]


def build_logreg_pipeline(max_iter: int) -> Pipeline:
    """Self-contained logreg artifact: imputation, standardization, one-hot
    encoding and the estimator live in ONE fitted sklearn Pipeline, persisted
    whole with joblib — no preprocessing metadata saved beside the estimator,
    so the artifact has no drift surface and carries its column contract
    internally.

    Train-only statistics hold structurally: Pipeline.fit(train) learns the
    imputer medians, scaler moments and carrier levels from the training
    frame alone; predict-time transforms reuse them on any new frame.

    Origin/dest/route identities are deliberately not one-hot here (7.6k+
    columns); their signal reaches the linear model through the hist_* rates.
    New-route TEST rows (NULL hist) impute to the train median — the linear
    baseline has no missingness signal by construction, because hist_* is
    never NULL on a training row. keep_empty_features guards the
    all-null-in-training edge (imputes 0, inert after scaling); unseen future
    carriers encode to all-zeros (handle_unknown='ignore').

    Metric-precision note (measured): refits of this model land within
    ~1e-4 ROC of each other, the lbfgs noise floor. The solver terminates on
    tolerance (n_iter 31-40 of 200, never the cap), so the stopping point is
    path-dependent; tightening tol to 1e-8 does NOT reconcile formulation
    variants (a ~3e-5 residual remains from float32-vs-float64 arithmetic
    and column ordering). The ddof-0-vs-1 scaling difference is 1 + 3e-8 at
    n = 16.7M — orders of magnitude too small to matter. Compare metrics at
    4 decimals; differences below ~1e-4 are numerically meaningless.
    """
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                f.NUMERIC_FEATURES,
            ),
            ("carrier", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), ["carrier"]),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("prep", preprocess),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=max_iter)),
        ]
    )


def xgb_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[f.FEATURES]


def classification_metrics(y_true, scores, threshold=0.5) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, scores >= threshold)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xgb-rounds", type=int, default=300)
    parser.add_argument("--logreg-max-iter", type=int, default=200)
    args = parser.parse_args()

    t0 = time.time()
    df, bq, dataset = load_mart()

    # ---- leakage audit: hard gate before anything touches the labels ----
    ordered_features = run_audit(bq, dataset)

    split = split_report(df)
    log.info("split: %s", json.dumps(split))

    train_mask = df[f.SPLIT_COL].to_numpy()
    test_mask = ~train_mask
    y_clf = df["label_arr_del15"].to_numpy()
    y_reg = df["label_arr_delay_minutes"].to_numpy(dtype="float32")

    base_rate_test = float(y_clf[test_mask].mean())
    # the deployable do-nothing baseline picks its class from TRAIN labels
    # (never from test), then is scored on test like any other model
    train_majority_class = int(y_clf[train_mask].mean() >= 0.5)
    baselines = {
        "test_delay_base_rate": base_rate_test,
        "train_majority_class": train_majority_class,
        "majority_class_accuracy": float((y_clf[test_mask] == train_majority_class).mean()),
        "random_roc_auc": 0.5,
        "random_pr_auc": base_rate_test,
        "mean_predictor_rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_reg[test_mask], np.full(test_mask.sum(), y_reg[train_mask].mean())
                )
            )
        ),
        "mean_predictor_mae": float(
            mean_absolute_error(
                y_reg[test_mask], np.full(test_mask.sum(), y_reg[train_mask].mean())
            )
        ),
    }

    results: dict = {"split": split, "baselines": baselines, "features": ordered_features}

    # ---- logistic-regression baseline (one self-contained Pipeline) ----
    x_lin = df[LOGREG_INPUT_COLUMNS]
    log.info(
        "fitting logreg pipeline on %s train rows (fit on train, transform on test) ...",
        f"{split['n_train']:,}",
    )
    logreg = build_logreg_pipeline(args.logreg_max_iter)
    logreg.fit(x_lin[train_mask], y_clf[train_mask])
    lin_scores = logreg.predict_proba(x_lin[test_mask])[:, 1]
    results["logreg_classifier"] = classification_metrics(y_clf[test_mask], lin_scores)
    lin_cols = list(logreg.named_steps["prep"].get_feature_names_out())
    coef = pd.Series(logreg.named_steps["clf"].coef_[0], index=lin_cols).sort_values(
        key=abs, ascending=False
    )
    results["logreg_top_coefficients"] = coef.head(15).round(4).to_dict()
    del x_lin, lin_scores

    # ---- XGBoost classifier ----
    pos = float(y_clf[train_mask].sum())
    spw = float((train_mask.sum() - pos) / pos)
    log.info("fitting xgboost classifier (scale_pos_weight=%.3f) ...", spw)
    x_xgb = xgb_frame(df)
    clf = xgb.XGBClassifier(
        n_estimators=args.xgb_rounds,
        learning_rate=0.1,
        max_depth=8,
        tree_method="hist",
        enable_categorical=True,
        scale_pos_weight=spw,
        n_jobs=-1,
        eval_metric="aucpr",
    )
    clf.fit(x_xgb[train_mask], y_clf[train_mask])
    clf_scores = clf.predict_proba(x_xgb[test_mask])[:, 1]
    results["xgb_classifier"] = classification_metrics(y_clf[test_mask], clf_scores)
    imp = pd.Series(clf.get_booster().get_score(importance_type="gain")).sort_values(
        ascending=False
    )
    results["xgb_classifier_importance_gain"] = imp.head(20).round(2).to_dict()

    # failure slices for the written reflection
    hist_missing = df["hist_route_arr_del15_rate"].isna().to_numpy()
    slices = {}
    for name, mask in {
        "route_hist_present": test_mask & ~hist_missing,
        "route_hist_missing": test_mask & hist_missing,
        "evening_dep_17_23": test_mask & df["crs_dep_hour"].between(17, 23).to_numpy(),
        "morning_dep_5_11": test_mask & df["crs_dep_hour"].between(5, 11).to_numpy(),
    }.items():
        if mask.sum() > 1000 and 0 < y_clf[mask].mean() < 1:
            sub_scores = clf.predict_proba(x_xgb[mask])[:, 1]
            base = float(y_clf[mask].mean())
            pr = float(average_precision_score(y_clf[mask], sub_scores))
            slices[name] = {
                "n": int(mask.sum()),
                "base_rate": round(base, 4),
                "pr_auc": round(pr, 4),
                # per-slice PR-AUC is meaningless without its base rate: the
                # comparable number across slices is the LIFT over prevalence
                "pr_auc_lift_vs_base": round(pr / base, 2),
                "roc_auc": round(float(roc_auc_score(y_clf[mask], sub_scores)), 4),
            }
    results["xgb_classifier_slices"] = slices
    del clf_scores

    # ---- XGBoost regressor ----
    log.info("fitting xgboost regressor ...")
    reg = xgb.XGBRegressor(
        n_estimators=args.xgb_rounds,
        learning_rate=0.1,
        max_depth=8,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=-1,
    )
    reg.fit(x_xgb[train_mask], y_reg[train_mask])
    reg_pred = reg.predict(x_xgb[test_mask])
    results["xgb_regressor"] = {
        "rmse": float(np.sqrt(mean_squared_error(y_reg[test_mask], reg_pred))),
        "mae": float(mean_absolute_error(y_reg[test_mask], reg_pred)),
    }
    # error by true-delay bucket: where the regressor actually fails
    buckets = pd.cut(
        y_reg[test_mask],
        [-0.01, 0, 15, 60, 180, np.inf],
        labels=["on_time_0", "1_15", "15_60", "60_180", "180_plus"],
    )
    err = pd.DataFrame({"bucket": buckets, "abs_err": np.abs(y_reg[test_mask] - reg_pred)})
    results["xgb_regressor_mae_by_true_delay"] = (
        err.groupby("bucket", observed=True)["abs_err"]
        .agg(["count", "mean"])
        .round(2)
        .rename(columns={"mean": "mae"})
        .to_dict("index")
    )
    imp_r = pd.Series(reg.get_booster().get_score(importance_type="gain")).sort_values(
        ascending=False
    )
    results["xgb_regressor_importance_gain"] = imp_r.head(20).round(2).to_dict()

    # ---- artifacts ----
    run_dir = ARTIFACT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    # UBJ artifacts persist the categorical LEVEL MAPPINGS and re-code by
    # name at prediction time (xgboost >= 3, pinned in pyproject) — verified
    # empirically: a fresh process scoring an independently-built frame with
    # slice-local category sets reproduced in-process predictions exactly
    # (130,354/130,354, max abs diff 0.0). Under xgboost 2.x the mapping was
    # NOT stored and scoring frames with different level sets mis-scored
    # silently; do not relax the pin.
    clf.save_model(run_dir / "xgb_classifier.ubj")
    reg.save_model(run_dir / "xgb_regressor.ubj")
    # the WHOLE fitted pipeline (preprocessing + estimator) is the artifact
    joblib.dump(logreg, run_dir / "logreg_pipeline.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    log.info("artifacts -> %s", run_dir)
    log.info("total wall time %.1f min", (time.time() - t0) / 60)

    print("\n===== RESULTS =====")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
