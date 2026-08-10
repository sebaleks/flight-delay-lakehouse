"""Tests for the fair-comparison helpers: share and mix adjustment.

The mix adjustment is the one worth pinning — it is the difference between
"this airline is worse" and "this airline flies harder routes".
"""

from __future__ import annotations

import pandas as pd
import pytest

from dashboard.metrics import mix_adjusted, share_of


def _frame(rows):
    return pd.DataFrame(
        rows, columns=["route", "carrier_key", "n_flights", "n_with_arr_outcome", "n_arr_del15"]
    )


def test_share_sums_to_one():
    df = _frame([("A-B", "AA", 100, 100, 10), ("A-B", "UA", 300, 300, 30)])
    s = share_of(df)
    assert s.sum() == pytest.approx(1.0)
    assert s.tolist() == pytest.approx([0.25, 0.75])


def test_share_of_empty_is_nan_not_a_crash():
    df = _frame([("A-B", "AA", 0, 0, 0)])
    assert share_of(df).isna().all()


def test_mix_adjustment_is_neutral_when_everyone_flies_the_same_routes():
    """Same route mix -> the index is just the ratio of rates, and an airline
    matching the field sits at 1.0."""
    df = _frame(
        [
            ("A-B", "AA", 100, 100, 30),  # 30%
            ("A-B", "UA", 100, 100, 10),  # 10%
        ]
    )
    out = mix_adjusted(df, "carrier_key", "route").set_index("carrier_key")
    # route average is 20%; AA is 1.5x that, UA is 0.5x
    assert out.loc["AA", "index"] == pytest.approx(1.5)
    assert out.loc["UA", "index"] == pytest.approx(0.5)


def test_mix_adjustment_exonerates_an_airline_that_only_flies_a_hard_route():
    """The whole point. Raw rates say HARD is worse; adjusted says they perform
    exactly as their routes predict."""
    df = _frame(
        [
            ("EASY", "GOOD", 1000, 1000, 100),  # 10% on an easy route
            ("HARD", "GOOD", 1000, 1000, 400),  # 40% on a hard route
            ("HARD", "ONLYHARD", 1000, 1000, 400),  # 40%, hard route only
        ]
    )
    out = mix_adjusted(df, "carrier_key", "route").set_index("carrier_key")
    # raw: ONLYHARD 40% vs GOOD 25% — looks much worse
    assert out.loc["ONLYHARD", "rate"] == pytest.approx(0.40)
    assert out.loc["GOOD", "rate"] == pytest.approx(0.25)
    # adjusted: both perform exactly at their routes' average
    assert out.loc["ONLYHARD", "index"] == pytest.approx(1.0)
    assert out.loc["GOOD", "index"] == pytest.approx(1.0)


def test_mix_adjustment_still_catches_a_genuinely_worse_airline():
    """Adjustment must not launder real underperformance away."""
    df = _frame(
        [
            ("HARD", "OK", 1000, 1000, 300),
            ("HARD", "BAD", 1000, 1000, 500),
        ]
    )
    out = mix_adjusted(df, "carrier_key", "route").set_index("carrier_key")
    assert out.loc["BAD", "index"] > 1.2
    assert out.loc["OK", "index"] < 0.9


def test_mix_adjustment_survives_a_zero_denominator():
    df = _frame([("A-B", "AA", 5, 0, 0), ("A-B", "UA", 100, 100, 20)])
    out = mix_adjusted(df, "carrier_key", "route").set_index("carrier_key")
    assert pd.isna(out.loc["AA", "rate"])
    assert out.loc["UA", "index"] == pytest.approx(1.0)
