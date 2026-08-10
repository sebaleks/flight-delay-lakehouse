"""Pure unit tests for the airport-day replay payload — no BigQuery, no
artifacts, no ADC. The loading/scoring half is exercised against the warehouse
by the deployed endpoint and ml/replay.py's own CLI assert; what is pinned
here is the JSON contract the ops page consumes: the Poisson-binomial
summary math, NaN -> null on the schedule-linkage columns, ordering, and
JSON-serializability of every value (numpy scalars must not leak through).

    uv run --extra ml --group dev pytest ml/test_replay.py
"""

from __future__ import annotations

import json
import math

import pandas as pd

from ml.replay import airport_day_payload


def _scored() -> pd.DataFrame:
    """Three flights shaped like score_airport_day's output frame."""
    return pd.DataFrame(
        {
            "flight_date": pd.to_datetime(["2024-09-13"] * 3),
            "carrier": pd.Categorical(["UA", "AA", "UA"]),
            "flight_number": ["100", "200", "300"],
            "origin": ["ORD"] * 3,
            "dest": ["DEN", "ATL", "SFO"],
            "dep_time": ["09:15", "07:30", "18:05"],
            "label_arr_del15": [True, False, False],
            "label_arr_delay_minutes": [42.0, -5.0, 3.0],
            "delay_probability": [0.8, 0.1, 0.3],
            "expected_delay_minutes": [25.3, 1.2, 8.9],
            "has_origin_weather": [1.0, 1.0, 0.0],
            "crs_dep_hour": [9.0, 7.0, 18.0],
            # the AA row is a swap-shaped linkage: rotation columns NULL
            "rotation_position": [2.0, float("nan"), 1.0],
            "legs_today": [4.0, float("nan"), 5.0],
            "is_tight_turnaround": [1.0, float("nan"), 0.0],
        }
    )


def _payload() -> dict:
    return airport_day_payload("ORD", "2024-09-13", _scored(), "20260730_145241")


def test_summary_is_poisson_binomial():
    """Σp and √Σp(1−p) — exact for a sum of independent Bernoullis, which is
    precisely the claim the calibrated per-flight probabilities make about a
    day. actual_delayed counts the labels, not any prediction."""
    s = _payload()["summary"]
    assert s["expected_delayed"] == round(0.8 + 0.1 + 0.3, 2)
    assert s["expected_delayed_sd"] == round(math.sqrt(0.8 * 0.2 + 0.1 * 0.9 + 0.3 * 0.7), 2)
    assert s["actual_delayed"] == 1


def test_flights_sorted_by_departure():
    times = [fl["dep_time"] for fl in _payload()["flights"]]
    assert times == sorted(times)
    assert times[0] == "07:30"


def test_swap_shaped_rotation_serializes_as_null():
    """The tail-swap restriction NULLs rotation columns; those must reach the
    page as null, never the string 'nan' or a crash."""
    by_carrier = {fl["carrier"]: fl for fl in _payload()["flights"]}
    assert by_carrier["AA"]["rotation_position"] is None
    assert by_carrier["AA"]["legs_today"] is None
    assert by_carrier["AA"]["is_tight_turnaround"] is None
    assert by_carrier["UA"]["rotation_position"] in (1, 2)
    assert isinstance(by_carrier["UA"]["rotation_position"], int)


def test_labels_ride_alongside_predictions():
    fl = next(f for f in _payload()["flights"] if f["dest"] == "DEN")
    assert fl["delay_probability"] == 0.8
    assert fl["actual_delayed"] is True
    assert fl["actual_delay_minutes"] == 42.0
    assert fl["dep_hour"] == 9 and isinstance(fl["dep_hour"], int)


def test_payload_is_json_serializable_and_honest():
    """json.dumps is the catch-all against numpy scalars leaking through; the
    basis block must say these are held-out rows under observed weather."""
    p = _payload()
    json.dumps(p)  # raises on any np.float32 / np.bool_ leak
    assert p["n_flights"] == 3
    assert p["prediction_basis"]["mode"] == "held_out_replay"
    assert p["prediction_basis"]["weather"] == "observed_at_scheduled_departure"
