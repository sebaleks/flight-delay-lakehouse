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
