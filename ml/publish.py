"""Publish a trained artifact run to GCS for the predictor image to bake in.

I/O only — no modelling, no transforms (CLAUDE.md §5). ADC, no key file (§2).

WHY THIS EXISTS. `ml/artifacts/` is git-ignored and ~695 MB per run, so the
predictor container cannot COPY it from the repo. Nothing in the repo fetched
artifacts from GCS either: `ml/tracking.py` only PUSHES to
gs://$GCS_BUCKET/mlflow as a side effect of training, and MLflow's local SQLite
backend is not readable from a Cloud Run image. This closes that gap with the
smallest possible mechanism — a plain versioned prefix the build step copies
from.

    gs://$GCS_BUCKET/serving/<run>/

PINNED, NEVER "LATEST". The build passes the run id as a substitution, so an
image is tied to exactly one artifact set and a redeploy is reproducible. There
is deliberately no "latest" alias: `load_models` already picks the
lexicographically-newest COMPLETE local run, and having a second, differently
defined notion of "latest" in the deploy path is how a service ends up serving
a model nobody chose.

IMMUTABLE. Publishing over an existing prefix is refused. An artifact run is
identified by its timestamp; if the bytes under that name could change, the run
id would stop identifying a model, and `/health`'s `artifacts` field — the only
thing tying a prediction to a model version — would become a lie.

    uv run --extra ml --extra ingestion python -m ml.publish            # newest complete run
    uv run --extra ml --extra ingestion python -m ml.publish --run 20260730_145241
    uv run --extra ml --extra ingestion python -m ml.publish --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from google.cloud import storage

from ingestion.config import require_env
from ml.train import ARTIFACT_ROOT

log = logging.getLogger("ml.publish")

PREFIX = "serving"
# the four files load_models() requires — a run missing any of them must never
# reach a deploy, because the service would fail startup with a confusing load
# error rather than an obvious "incomplete run" one
REQUIRED = (
    "xgb_classifier.ubj",
    "xgb_regressor.ubj",
    "logreg_pipeline.joblib",
    "calibrator.joblib",
)
# reports the endpoints serve. Optional by design: an older run predates them
# and must still be publishable and servable (/calibration and /outcome-mix
# degrade to available=false), but a warning is worth printing.
OPTIONAL = ("metrics.json", "exceedance.json")


def newest_complete_run() -> Path:
    if not ARTIFACT_ROOT.is_dir():
        raise SystemExit(f"{ARTIFACT_ROOT} does not exist — train first")
    runs = sorted(
        d for d in ARTIFACT_ROOT.iterdir() if d.is_dir() and all((d / n).exists() for n in REQUIRED)
    )
    if not runs:
        raise SystemExit(f"no complete artifact runs under {ARTIFACT_ROOT}")
    return runs[-1]


def publish(run: str | None = None, dry_run: bool = False) -> str:
    run_dir = (ARTIFACT_ROOT / run) if run else newest_complete_run()
    if not run_dir.is_dir():
        raise SystemExit(f"{run_dir} is not a directory")
    missing = [n for n in REQUIRED if not (run_dir / n).exists()]
    if missing:
        raise SystemExit(f"{run_dir.name} is INCOMPLETE, missing {missing} — refusing to publish")
    for n in OPTIONAL:
        if not (run_dir / n).exists():
            log.warning(
                "%s has no %s — the matching endpoint will report available=false",
                run_dir.name,
                n,
            )

    bucket_name = require_env("GCS_BUCKET")
    client = storage.Client(project=require_env("GCP_PROJECT_ID"))
    bucket = client.bucket(bucket_name)
    dest = f"{PREFIX}/{run_dir.name}"

    existing = list(client.list_blobs(bucket, prefix=f"{dest}/", max_results=1))
    if existing:
        raise SystemExit(
            f"gs://{bucket_name}/{dest}/ already exists — artifact runs are immutable. "
            "Publish a new training run rather than overwriting one that a deployed "
            "image may be pinned to."
        )

    files = [p for p in sorted(run_dir.iterdir()) if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    log.info(
        "publishing %s (%d files, %.0f MB) -> gs://%s/%s/",
        run_dir.name,
        len(files),
        total / 1e6,
        bucket_name,
        dest,
    )
    if dry_run:
        for p in files:
            log.info("  [dry-run] %s (%.1f MB)", p.name, p.stat().st_size / 1e6)
    else:
        for p in files:
            bucket.blob(f"{dest}/{p.name}").upload_from_filename(str(p))
            log.info("  uploaded %s (%.1f MB)", p.name, p.stat().st_size / 1e6)

    print(f"\nrun published: {run_dir.name}")
    print("deploy it with:")
    print(
        "  gcloud builds submit --config cloudbuild.predictor.yaml "
        f"--substitutions=_RUN={run_dir.name},_BUCKET={bucket_name}"
    )
    return run_dir.name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None, help="artifact run id; default is the newest complete")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        publish(args.run, args.dry_run)
    except SystemExit as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
