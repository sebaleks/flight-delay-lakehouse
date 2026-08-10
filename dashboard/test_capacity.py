"""Unit tests for the ops-capacity math — the page's claims, pinned.

uv run --extra dashboard --group dev pytest dashboard/test_capacity.py
"""

from __future__ import annotations

import math

import pytest

from dashboard.capacity import (
    comms_ranking,
    day_summary,
    fragile_banks,
    hourly_banks,
    remaining_legs,
)


def _fl(p, hour, actual, pos=None, legs=None, tight=None, **kw):
    return {
        "delay_probability": p,
        "dep_hour": hour,
        "actual_delayed": actual,
        "rotation_position": pos,
        "legs_today": legs,
        "is_tight_turnaround": tight,
        **kw,
    }


FLIGHTS = [
    _fl(0.8, 18, True, pos=2, legs=5, tight=True, dep_time="18:10", carrier="UA"),
    _fl(0.1, 18, False, pos=1, legs=1, tight=False, dep_time="18:30", carrier="AA"),
    # swap-shaped linkage: rotation nulls must contribute ZERO downstream
    _fl(0.5, 7, True, dep_time="07:05", carrier="WN"),
    _fl(0.2, 7, False, pos=3, legs=3, tight=False, dep_time="07:40", carrier="DL"),
]


def test_remaining_legs_is_schedule_linkage_only():
    assert remaining_legs({"legs_today": 5, "rotation_position": 2}) == 3
    # swap-shaped / unknown linkage: zero, never a guess
    assert remaining_legs({"legs_today": None, "rotation_position": None}) == 0
    assert remaining_legs({"legs_today": 3, "rotation_position": None}) == 0
    # last leg of the day carries nothing downstream
    assert remaining_legs({"legs_today": 3, "rotation_position": 3}) == 0


def test_day_summary_poisson_binomial():
    s = day_summary(FLIGHTS)
    assert s["n_flights"] == 4 and s["actual"] == 2
    assert s["expected"] == pytest.approx(0.8 + 0.1 + 0.5 + 0.2)
    assert s["sd"] == pytest.approx(math.sqrt(0.8 * 0.2 + 0.1 * 0.9 + 0.5 * 0.5 + 0.2 * 0.8))
    # 0.8*(5-2) + 0.1*0 + 0.5*0 (swap: null) + 0.2*0 (last leg)
    assert s["expected_downstream"] == 0.8 * 3


def test_hourly_banks_grouping_and_tight_share():
    banks = hourly_banks(FLIGHTS)
    assert [b["hour"] for b in banks] == [7, 18]
    b18 = banks[1]
    assert b18["n_flights"] == 2 and b18["actual"] == 1
    assert b18["expected"] == 0.9
    assert b18["tight_share"] == 0.5  # absent flag (swap row) counts as not-tight
    assert banks[0]["tight_share"] == 0.0
    assert b18["downstream_legs"] == 3 and b18["expected_downstream"] == 0.8 * 3


def test_fragile_banks_ranked_by_downstream_stake():
    ranked = fragile_banks(hourly_banks(FLIGHTS))
    assert [b["hour"] for b in ranked] == [18, 7]  # 2.4 downstream beats 0
    assert ranked[0]["fragile"] is True  # tight_share 0.5 >= 0.25
    assert ranked[1]["fragile"] is False


def test_comms_ranking_orders_by_probability_and_truncates():
    top = comms_ranking(FLIGHTS, top_n=2)
    assert [t["delay_probability"] for t in top] == [0.8, 0.5]
    assert top[0]["flight"].startswith("UA")
    assert top[0]["remaining_legs"] == 3
    # labels ride along as reported outcomes — the ranking itself is p only
    assert top[0]["actual_delayed"] is True


def test_empty_day_degrades_to_zeros():
    s = day_summary([])
    assert s == {
        "n_flights": 0,
        "expected": 0,
        "sd": 0.0,
        "actual": 0,
        "expected_downstream": 0,
    }
    assert hourly_banks([]) == []
    assert comms_ranking([]) == []
