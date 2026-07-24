"""Pre-training leakage self-audit (CLAUDE.md §9). Fails hard — training must
not proceed if any check fails.

Checks:
1. No ``label_`` column and no forbidden post-departure/arrival outcome column
   in the feature list (local, against ml.features).
2. The live BigQuery mart schema equals the audited column set exactly — so a
   mart change cannot silently feed the models something un-audited. (In the
   warehouse, ``assert_ml_features_no_leakage`` pins the mart schema to the
   audited allowlist and ``assert_ml_weather_obs_before_departure`` pins
   every weather observation at or before SCHEDULED departure; the hist_*
   smoothed-rate semantics live in the mart SQL itself. This audit re-asserts
   the schema contract at train time.)
3. The split column and both labels exist and features/labels/keys are
   disjoint.

Run standalone:  uv run --extra ml python -m ml.audit
"""

from __future__ import annotations

import logging

from google.cloud import bigquery

from ingestion.config import require_env
from ml import features as f

log = logging.getLogger("ml.audit")

MART_TABLE = "ml_flight_features"


class LeakageAuditError(AssertionError):
    pass


def run_audit(bq: bigquery.Client, dataset: str) -> list[str]:
    """Return the ordered feature list; raise LeakageAuditError on any failure."""
    problems: list[str] = []

    # 1. local: labels / forbidden names can never be features
    label_like = [c for c in f.FEATURES if c.startswith("label_") or c in f.LABELS]
    if label_like:
        problems.append(f"label columns in feature list: {label_like}")
    forbidden = sorted(set(c.lower() for c in f.FEATURES) & f.FORBIDDEN_FEATURES)
    if forbidden:
        problems.append(f"forbidden post-departure columns in feature list: {forbidden}")
    overlap = set(f.FEATURES) & set(f.EXCLUDED)
    if overlap:
        problems.append(f"feature list overlaps excluded/bookkeeping columns: {sorted(overlap)}")

    # 2. remote: live mart schema must equal the audited set exactly
    rows = bq.query(
        f"""
        select column_name
        from `{bq.project}.{dataset}`.INFORMATION_SCHEMA.COLUMNS
        where table_name = '{MART_TABLE}'
        """
    ).result()
    live = {r.column_name for r in rows}
    if not live:
        problems.append(f"mart {dataset}.{MART_TABLE} not found — nothing to train on")
    else:
        expected = set(f.MART_COLUMNS)
        if live != expected:
            problems.append(
                f"mart schema drifted from audited set — un-audited in mart: "
                f"{sorted(live - expected)}; audited but missing: {sorted(expected - live)}"
            )
        live_forbidden = sorted(c for c in live if c.lower() in f.FORBIDDEN_FEATURES)
        if live_forbidden:
            problems.append(f"forbidden outcome columns present in mart: {live_forbidden}")

    # 3. split + labels wired
    for col in (f.SPLIT_COL, *f.LABELS):
        if col not in f.MART_COLUMNS:
            problems.append(f"required column missing from audited schema: {col}")

    if problems:
        raise LeakageAuditError("; ".join(problems))

    log.info("leakage audit PASSED: %d features, 0 label/forbidden columns", len(f.FEATURES))
    log.info(
        "provenance (enforced by dbt standing guards on the mart): hist_* = "
        "training-window rates smoothed toward the global (constant within "
        "an entity, train and test alike); origin weather = last hourly ISD "
        "observation at or before scheduled departure; holiday flags = "
        "generated calendar"
    )
    return list(f.FEATURES)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    project = require_env("GCP_PROJECT_ID")
    dataset = require_env("BQ_GOLD_DATASET")
    feats = run_audit(bigquery.Client(project=project), dataset)
    print(f"ordered feature list ({len(feats)}):")
    for i, name in enumerate(feats, 1):
        kind = "categorical" if name in f.CATEGORICAL_FEATURES else "numeric"
        print(f"  {i:2d}. {name}  [{kind}]")


if __name__ == "__main__":
    main()
