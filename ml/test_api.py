"""Unit tests for /replay/airport-day's guards — no BigQuery, no artifacts.

The endpoint's whole job before touching the warehouse is refusing the wrong
requests: an unknown origin, a training-window date, a date with no held-out
rows — and proving after the fact that nothing from the training window came
back. Those refusals are pinned here with a faked serving context; the happy
path against real data is exercised by the deployed service.

Skipped when fastapi is not installed (the documented gate runs with only the
`dashboard` + `ml` extras; add `--extra serve` to run these).

    uv run --extra ml --extra serve --group dev pytest ml/test_api.py
"""

from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")

import ml.api as api  # noqa: E402
import ml.replay as replay  # noqa: E402


def _fake_ctx(tmp_path, metrics: dict | None = None):
    run = tmp_path / "20260730_145241"
    run.mkdir()
    if metrics is not None:
        (run / "metrics.json").write_text(json.dumps(metrics))
    return SimpleNamespace(
        airports=pd.DataFrame({"tz": ["America/Chicago"]}, index=pd.Index(["ORD"], name="iata")),
        models=SimpleNamespace(artifacts_dir=run),
    )


def test_unknown_origin_is_404(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_ctx", _fake_ctx(tmp_path))
    with pytest.raises(fastapi.HTTPException) as exc:
        api.replay_airport_day("XXX", flight_date=dt.date(2024, 9, 13))
    assert exc.value.status_code == 404


def test_training_window_date_is_404_using_the_runs_own_split(monkeypatch, tmp_path):
    """The boundary comes from the run's metrics.json when present — the
    artifacts define their test window, not a constant in the code."""
    ctx = _fake_ctx(tmp_path, metrics={"split": {"test_date_min": "2024-07-01"}})
    monkeypatch.setattr(api, "_ctx", ctx)
    with pytest.raises(fastapi.HTTPException) as exc:
        api.replay_airport_day("ORD", flight_date=dt.date(2024, 6, 30))
    assert exc.value.status_code == 404
    assert "training window" in exc.value.detail


def test_training_window_falls_back_to_holdout_floor(monkeypatch, tmp_path):
    """A run that predates metrics.json still refuses training-window dates."""
    monkeypatch.setattr(api, "_ctx", _fake_ctx(tmp_path, metrics=None))
    with pytest.raises(fastapi.HTTPException) as exc:
        api.replay_airport_day("ord", flight_date=dt.date(2023, 9, 13))  # lowercase must also work
    assert exc.value.status_code == 404
    assert "training window" in exc.value.detail


def test_no_held_out_rows_is_404(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_ctx", _fake_ctx(tmp_path))
    monkeypatch.setattr(
        replay,
        "score_airport_day",
        lambda *a, **k: (_ for _ in ()).throw(replay.NoHeldOutRows("empty")),
    )
    with pytest.raises(fastapi.HTTPException) as exc:
        api.replay_airport_day("ORD", flight_date=dt.date(2024, 12, 25))
    assert exc.value.status_code == 404
    assert "no held-out flights" in exc.value.detail


def test_a_leaked_training_row_raises_not_returns(monkeypatch, tmp_path):
    """Belt behind the WHERE clause: if a returned row somehow predates the
    held-out floor, the endpoint must blow up, never serve it as a demo."""
    monkeypatch.setattr(api, "_ctx", _fake_ctx(tmp_path))
    leaked = pd.DataFrame({"flight_date": pd.to_datetime(["2024-01-01"])})
    monkeypatch.setattr(replay, "score_airport_day", lambda *a, **k: leaked)
    with pytest.raises(RuntimeError, match="training row"):
        api.replay_airport_day("ORD", flight_date=dt.date(2024, 9, 13))
