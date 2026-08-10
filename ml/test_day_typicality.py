"""Pure unit tests for the day-typicality math — no BigQuery, no artifacts.

uv run --extra ml --group dev pytest ml/test_day_typicality.py
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd
import pytest

from ml.day_typicality import daily_moments, verdict


def _scored(days: dict[str, list[tuple[float, bool]]]) -> pd.DataFrame:
    """{date: [(p, actual), ...]} -> a frame shaped like ml/replay.score's output."""
    rows = [
        {"flight_date": pd.Timestamp(day), "delay_probability": p, "label_arr_del15": actual}
        for day, flights in days.items()
        for p, actual in flights
    ]
    return pd.DataFrame(rows)


def test_daily_moments_are_poisson_binomial():
    daily = daily_moments(
        _scored({"2024-09-13": [(0.8, True), (0.1, False)], "2024-09-14": [(0.5, True)]})
    )
    d1 = daily.loc[dt.date(2024, 9, 13)]
    assert d1["n"] == 2 and d1["actual"] == 1
    assert d1["expected"] == pytest.approx(0.9)
    assert d1["sd"] == pytest.approx(math.sqrt(0.8 * 0.2 + 0.1 * 0.9))
    assert d1["z"] == pytest.approx((1 - 0.9) / math.sqrt(0.25))
    d2 = daily.loc[dt.date(2024, 9, 14)]
    assert d2["z"] == pytest.approx(0.5 / 0.5)


def test_verdict_flags_both_tails_as_atypical():
    """A day the model nailed suspiciously well is as cherry-picked as a day
    it missed — the typical band excludes BOTH tails."""
    # 21 days, each with 20 p=0.5 flights and i of them delayed (i = 0..20):
    # z runs evenly from -4.47 to +4.47, so the tails are unambiguous
    days = {f"2024-07-{i + 1:02d}": [(0.5, j < i) for j in range(20)] for i in range(21)}
    daily = daily_moments(_scored(days))
    zmax_day = daily["z"].idxmax()
    zmin_day = daily["z"].idxmin()
    assert not verdict(daily, zmax_day)["typical"]
    assert not verdict(daily, zmin_day)["typical"]
    # a median day is typical
    median_day = daily["z"].sort_values().index[len(daily) // 2]
    v = verdict(daily, median_day)
    assert v["typical"] and v["band"][0] <= v["z"] <= v["band"][1]


def test_verdict_unknown_day_raises_keyerror():
    daily = daily_moments(_scored({"2024-09-13": [(0.5, True)]}))
    with pytest.raises(KeyError):
        verdict(daily, dt.date(2024, 9, 14))
