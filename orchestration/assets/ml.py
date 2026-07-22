"""ML training asset: wraps ml.train.run_training — the EXACT path the CLI
runs. The leakage audit gate, the is_training_row split with its assertions,
the canonical six-column determinism sort, and the self-contained artifact
contract (Pipeline joblib + UBJ with persisted category mappings) all apply
unchanged, because this wrapper calls the real function and nothing else.
"""

from __future__ import annotations

from dagster import AssetCheckResult, MaterializeResult, asset, asset_check
from dagster_dbt import get_asset_key_for_model

from orchestration.assets.dbt_layer import flight_delays_dbt_assets

ARTIFACT_CONTRACT = [
    "xgb_classifier.ubj",
    "xgb_regressor.ubj",
    "logreg_pipeline.joblib",
    "metrics.json",
]


@asset(
    group_name="ml",
    deps=[get_asset_key_for_model([flight_delays_dbt_assets], "ml_flight_features")],
    description="Train + evaluate both models via ml.train.run_training (the real path)",
)
def ml_training() -> MaterializeResult:
    import logging
    from pathlib import Path

    from ml.train import run_training

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )  # steps have no root logging config; ml progress INFO would be dropped

    results = run_training()
    # contract enforced in-asset so a violation fails the RUN, not just a check
    missing = [n for n in ARTIFACT_CONTRACT if not (Path(results["artifacts_dir"]) / n).exists()]
    if missing:
        raise RuntimeError(f"artifact contract violated: missing {missing}")
    return MaterializeResult(
        metadata={
            "artifacts_dir": results["artifacts_dir"],
            "n_train": results["split"]["n_train"],
            "n_test": results["split"]["n_test"],
            "xgb_roc_auc": results["xgb_classifier"]["roc_auc"],
            "xgb_pr_auc": results["xgb_classifier"]["pr_auc"],
            "logreg_roc_auc": results["logreg_classifier"]["roc_auc"],
            "logreg_pr_auc": results["logreg_classifier"]["pr_auc"],
            "reg_rmse": results["xgb_regressor"]["rmse"],
            "reg_mae": results["xgb_regressor"]["mae"],
        }
    )


@asset_check(asset=ml_training, description="This run's artifact dir satisfies the contract")
def ml_artifact_contract(context) -> AssetCheckResult:
    from pathlib import Path

    from dagster import AssetKey

    # check the dir the CURRENT materialization reported, never a
    # lexicographically-latest guess (stale/future-named dirs can exist)
    event = context.instance.get_latest_materialization_event(AssetKey("ml_training"))
    if event is None:
        return AssetCheckResult(passed=False, metadata={"reason": "no materialization found"})
    meta = event.asset_materialization.metadata
    run_dir = Path(str(meta["artifacts_dir"].value))
    missing = [n for n in ARTIFACT_CONTRACT if not (run_dir / n).exists()]
    return AssetCheckResult(
        passed=not missing,
        metadata={"artifacts_dir": str(run_dir), "missing": str(missing)},
    )
