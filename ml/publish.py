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

IMMUTABLE ONCE COMPLETE, RETRYABLE BEFORE THAT. A COMPLETION MARKER
(`_PUBLISHED.json`) is written LAST, after every file lands. Publishing over a
prefix that has the marker is refused: an artifact run is identified by its
timestamp, and if the bytes under that name could change, the run id would stop
identifying a model and `/health`'s `artifacts` field — the only thing tying a
prediction to a model version — would become a lie.

A prefix WITHOUT the marker was never a published run; it is the wreckage of an
interrupted upload, and it is safe to overwrite. Without this distinction a
single network timeout on the 438 MB booster strands the run forever: the
existence check refuses to retry, and the build refuses the half-populated
prefix. (This is not hypothetical — it happened on the first real run.)

    uv run --extra ml --extra ingestion python -m ml.publish            # newest complete run
    uv run --extra ml --extra ingestion python -m ml.publish --run 20260730_145241
    uv run --extra ml --extra ingestion python -m ml.publish --dry-run
"""

from __future__ import annotations

import argparse
import json
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
# written LAST; its presence is what makes a prefix an immutable published run
MARKER = "_PUBLISHED.json"
# 8 MiB chunks turn these into RESUMABLE uploads, and the default 120s deadline
# is not enough for a 438 MB object on a home connection — the first real run
# died on exactly that.
CHUNK_BYTES = 8 * 1024 * 1024
UPLOAD_TIMEOUT_S = 900


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

    # COMPLETE (marker present) -> immutable, refuse. PARTIAL (no marker) -> the
    # remains of a failed upload, safe to overwrite.
    if bucket.blob(f"{dest}/{MARKER}").exists(client):
        raise SystemExit(
            f"gs://{bucket_name}/{dest}/ is already published — artifact runs are "
            "immutable. Publish a new training run rather than overwriting one a "
            "deployed image may be pinned to."
        )
    stale = [b.name for b in client.list_blobs(bucket, prefix=f"{dest}/")]
    if stale:
        log.warning(
            "gs://%s/%s/ has %d object(s) but no %s — treating it as an INTERRUPTED "
            "publish and overwriting. A completed run would have the marker.",
            bucket_name,
            dest,
            len(stale),
            MARKER,
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
        log.info("  [dry-run] %s (completion marker, written last)", MARKER)
    else:
        for p in files:
            blob = bucket.blob(f"{dest}/{p.name}")
            blob.chunk_size = CHUNK_BYTES  # resumable
            blob.upload_from_filename(str(p), timeout=UPLOAD_TIMEOUT_S)
            log.info("  uploaded %s (%.1f MB)", p.name, p.stat().st_size / 1e6)
        # LAST, and only now is the prefix a published run
        bucket.blob(f"{dest}/{MARKER}").upload_from_string(
            json.dumps(
                {
                    "run": run_dir.name,
                    "files": {p.name: p.stat().st_size for p in files},
                    "required_present": sorted(REQUIRED),
                },
                indent=2,
            ),
            content_type="application/json",
            timeout=UPLOAD_TIMEOUT_S,
        )
        log.info("  wrote %s — run is now published and immutable", MARKER)

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
