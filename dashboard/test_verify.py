"""Pytest wrapper around the dashboard correctness harness.

Asserts every dashboard rate matches its direct-BigQuery ground truth. This is
an INTEGRATION test — it needs ADC + BigQuery, so it SKIPS (never fails) when
credentials or connectivity are absent, keeping CI credential-free per CLAUDE.md.
A genuine rate mismatch (a returned check with ok=False) still fails loudly.

Run:  uv run --extra dashboard pytest dashboard/test_verify.py
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit", reason="dashboard extra not installed")


def test_dashboard_rates_match_bigquery() -> None:
    from google.api_core.exceptions import GoogleAPIError
    from google.auth.exceptions import GoogleAuthError

    from dashboard.verify import verify

    try:
        rows = verify()
    except (GoogleAuthError, GoogleAPIError, OSError) as exc:
        pytest.skip(f"BigQuery/ADC unavailable — integration test skipped: {exc}")

    failed = [r for r in rows if not r["ok"]]
    assert not failed, f"dashboard rates disagree with BigQuery: {failed}"
    assert len(rows) >= 9, f"expected the full check set, got {len(rows)}"


def test_schemas_match_bigquery() -> None:
    """``dashboard.schemas.SCHEMAS`` must still describe the real views.

    ``test_pages_render.py`` renders every page against frames built from
    SCHEMAS, with no credentials, so it is the guard that runs in CI. That guard
    is only as good as the column lists behind it: if a dbt change renames a
    column and SCHEMAS is not updated, the fixtures keep the OLD name, every
    page test keeps passing, and the dashboard breaks in production anyway.
    This test is what stops that — it needs BigQuery, so it skips like its
    neighbour above rather than failing a credential-free CI run.
    """
    from google.api_core.exceptions import GoogleAPIError
    from google.auth.exceptions import GoogleAuthError

    from dashboard import schemas
    from dashboard.config import gcp_project, gold_dataset

    try:
        from google.cloud import bigquery

        project, dataset = gcp_project(), gold_dataset()
        sql = (
            "SELECT table_name, column_name "
            f"FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS` "
            "WHERE table_name LIKE 'dash_%' ORDER BY table_name, ordinal_position"
        )
        rows = list(bigquery.Client(project=project).query(sql))
    except (GoogleAuthError, GoogleAPIError, OSError, SystemExit) as exc:
        pytest.skip(f"BigQuery/ADC unavailable — integration test skipped: {exc}")

    live: dict[str, list[str]] = {}
    for r in rows:
        live.setdefault(r.table_name, []).append(r.column_name)

    for view, cols in schemas.SCHEMAS.items():
        assert view in live, f"{view} is in SCHEMAS but not in BigQuery"
        assert list(cols) == live[view], (
            f"{view} drifted.\n  SCHEMAS: {list(cols)}\n  BigQuery: {live[view]}\n"
            "Update dashboard/schemas.py — the page render tests are trusting it."
        )
