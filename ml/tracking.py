"""MLflow experiment tracking — GCS-backed artifacts, local metadata backend.

WHY THIS SHAPE. MLflow has two stores:
  * backend store (run metadata: params, metrics, tags) — must be a SQL
    database or file store. GCS cannot hold metadata without a standing tracking
    server + SQL DB, which is out of scope for this batch pipeline. So the
    backend is a LOCAL SQLite database (``MLFLOW_TRACKING_URI``, default
    ``sqlite:///<repo>/mlflow.db``, git-ignored — MLflow 3 deprecated the bare
    file store). Point it at a tracking server later to move metadata to the
    cloud.
  * artifact store (models, metrics.json, plots) — ``gs://$GCS_BUCKET/mlflow``,
    so the heavy artifacts live in the cloud alongside the medallion layers.
    All GCP identifiers come from env vars (CLAUDE.md §2); nothing hardcoded.

DISCIPLINE. Tracking is a pure SIDE EFFECT: it reads what the pipeline already
computed and never changes model fits, metrics, or ``ml/artifacts/`` — so
determinism (bit-identical ``metrics.json``) is untouched. Every MLflow call is
wrapped so a tracking or GCS outage degrades to a warning and the run
completes untracked — a tracking failure must never fail a training run.
mlflow is imported LAZILY so importing this module (and ``ml.train``) never
requires mlflow to be installed. Disable entirely with ``MLFLOW_TRACKING=off``.
"""

from __future__ import annotations

import logging
import math
import numbers
import os
from pathlib import Path

log = logging.getLogger("ml.tracking")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OFF = {"off", "0", "false", "no"}


def enabled() -> bool:
    return os.environ.get("MLFLOW_TRACKING", "on").strip().lower() not in _OFF


def experiment_name() -> str:
    return os.environ.get("MLFLOW_EXPERIMENT", "flight-delay")


def tracking_uri() -> str:
    # Local SQLite backend (no server); MLflow 3 deprecated the bare file store.
    # Absolute path -> sqlite:/// + /abs = sqlite:////abs (SQLAlchemy convention).
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{_REPO_ROOT / 'mlflow.db'}"


def artifact_location() -> str | None:
    """GCS artifact root, or None to fall back to the backend's default (local
    under the file store) when no bucket is configured."""
    bucket = os.environ.get("GCS_BUCKET")
    return f"gs://{bucket}/mlflow" if bucket else None


def configure() -> bool:
    """Set the tracking URI and ensure the experiment exists with a GCS artifact
    location. Returns True if tracking is active. NEVER raises."""
    if not enabled():
        return False
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri())
        loc = artifact_location()
        exp = mlflow.get_experiment_by_name(experiment_name())
        if exp is None:
            mlflow.create_experiment(experiment_name(), artifact_location=loc)
            exp = mlflow.get_experiment_by_name(experiment_name())
        mlflow.set_experiment(experiment_name())
        # artifact_location is FIXED at experiment creation. A pre-existing
        # experiment keeps its original store even if GCS_BUCKET changed since,
        # so log the experiment's ACTUAL location (never a misleading gs://) and
        # warn loudly on a mismatch — switch by setting a new MLFLOW_EXPERIMENT.
        actual = exp.artifact_location if exp is not None else loc
        if loc and actual and not str(actual).startswith(loc):
            log.warning(
                "mlflow experiment %r already exists with artifact_location=%s; "
                "artifacts log THERE, not the configured %s (set a new "
                "MLFLOW_EXPERIMENT to use the GCS location)",
                experiment_name(),
                actual,
                loc,
            )
        log.info(
            "mlflow: uri=%s experiment=%s artifacts=%s",
            tracking_uri(),
            experiment_name(),
            actual or "(backend default)",
        )
        return True
    except Exception as e:  # noqa: BLE001 — tracking must never break a run
        log.warning("mlflow disabled (configure failed: %s)", e)
        return False


def log_run(
    run_name: str,
    params: dict,
    metrics: dict,
    tags: dict | None = None,
    artifact_dir: str | Path | None = None,
) -> bool:
    """Log one run (params, scalar metrics, tags, and — if given — the whole
    artifacts directory) to MLflow. Pure side effect; swallows ALL errors so a
    tracking/GCS outage never fails the caller. Returns True only if the run was
    ACTUALLY logged — False when tracking is disabled or logging failed — so
    callers can report real success rather than just the env switch."""
    if not configure():
        return False
    try:
        import mlflow

        # only FINITE, non-bool real scalars reach mlflow.log_metrics — numbers.Real
        # keeps numpy floats/ints (isinstance(np.float32, float) is False), bool is
        # excluded (it is an int subclass), and NaN/inf are dropped, not stored
        clean_metrics = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)
        }
        with mlflow.start_run(run_name=run_name):
            if tags:
                mlflow.set_tags(tags)
            mlflow.log_params(params)
            mlflow.log_metrics(clean_metrics)
            if artifact_dir is not None and Path(artifact_dir).is_dir():
                mlflow.log_artifacts(str(artifact_dir))
        log.info("mlflow: logged run '%s' to experiment %s", run_name, experiment_name())
        return True
    except Exception as e:  # noqa: BLE001 — tracking must never break a run
        log.warning("mlflow logging failed; '%s' completed untracked: %s", run_name, e)
        return False
