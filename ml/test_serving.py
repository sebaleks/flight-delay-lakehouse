"""Pure unit tests for the serving helpers — no BigQuery, no artifacts, no ADC.

Scope is deliberately the logic that has non-obvious behaviour and no other
guard: the vectorised response-record builder, the holiday-flag cache's
immutability, and the turnaround-band mirror. Everything that needs the
warehouse is covered by ml/parity.py (golden vectors) and the dbt tests.

    uv run --extra ml --group dev pytest ml/test_serving.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml import features as f
from ml.serving import _feature_records, _holiday_flags, _turnaround_band


def _frame(n: int = 3) -> pd.DataFrame:
    """A frame shaped exactly like the assembled feature frame."""
    data: dict[str, object] = {}
    for c in f.CATEGORICAL_FEATURES:
        # a fixed vocabulary with an unused level, as the serving path builds it
        data[c] = pd.Categorical(["AA"] * n, categories=["AA", "UA", "WN"])
    for c in f.NUMERIC_FEATURES:
        data[c] = np.arange(n, dtype="float32")
    return pd.DataFrame(data)[list(f.FEATURES)]


def test_feature_records_shape_and_keys():
    recs = _feature_records(_frame(3))
    assert len(recs) == 3
    for r in recs:
        assert list(r) == list(f.FEATURES)


def test_feature_records_types_match_the_json_contract():
    r = _feature_records(_frame(1))[0]
    for c in f.CATEGORICAL_FEATURES:
        assert isinstance(r[c], str), f"{c} must serialize as a string"
    for c in f.NUMERIC_FEATURES:
        assert isinstance(r[c], float), f"{c} must serialize as a float, not np.float32"


def test_feature_records_maps_nan_to_none():
    """The bug this pins: numeric columns hold np.float32, which is NOT a Python
    float, so an isinstance(v, float) check never fires and NaN leaks into the
    JSON response on the missing-data path."""
    x = _frame(2)
    x.loc[0, "distance"] = np.nan
    recs = _feature_records(x)
    assert recs[0]["distance"] is None
    assert recs[1]["distance"] == 1.0


def test_feature_records_maps_missing_category_to_none():
    x = _frame(2)
    # an unseen value becomes a missing category (the training NULL path), not a
    # crash and not the literal string "nan"
    x["carrier"] = pd.Categorical([None, "AA"], categories=["AA", "UA", "WN"])
    recs = _feature_records(x)
    assert recs[0]["carrier"] is None
    assert recs[1]["carrier"] == "AA"


def test_feature_records_rows_are_independent():
    x = _frame(3)
    x.loc[1, "crs_dep_hour"] = 17.0
    recs = _feature_records(x)
    assert [r["crs_dep_hour"] for r in recs] == [0.0, 17.0, 2.0]


def test_holiday_flags_are_read_only():
    """Cached, so every caller for a date shares one object — a mutation would
    silently corrupt the feature for every later request in the process."""
    import datetime

    flags = _holiday_flags(datetime.date(2024, 12, 25))
    assert flags["is_holiday"] == 1.0
    with pytest.raises(TypeError):
        flags["is_holiday"] = 0.0  # type: ignore[index]


def test_holiday_flags_neighbours():
    import datetime

    assert _holiday_flags(datetime.date(2024, 12, 24))["is_day_before_holiday"] == 1.0
    assert _holiday_flags(datetime.date(2024, 12, 26))["is_day_after_holiday"] == 1.0
    assert _holiday_flags(datetime.date(2024, 3, 12))["is_holiday"] == 0.0


@pytest.mark.parametrize(
    ("has_inbound", "turnaround", "expected"),
    [
        (False, None, "no_inbound"),
        (False, 50.0, "no_inbound"),
        (True, None, "no_inbound"),
        (True, 34.9, "lt_35"),
        (True, 35.0, "35_60"),  # boundaries are >=, mirroring int_aircraft_rotation
        (True, 59.9, "35_60"),
        (True, 60.0, "60_120"),
        (True, 119.9, "60_120"),
        (True, 120.0, "ge_120"),
    ],
)
def test_turnaround_band_mirrors_the_sql(has_inbound, turnaround, expected):
    """Pins the Python mirror of int_aircraft_rotation.sql's band CASE. The SQL
    side is pinned by assert_ml_rotation_schedule_only; this is the other half."""
    assert _turnaround_band(has_inbound, turnaround) == expected


def test_departure_utc_precision(monkeypatch):
    """The past/future decision must keep MINUTES; the weather bucket truncates.

    The bug: at 17:05 local, a flight scheduled 17:30 truncated to 17:00 and came
    back flight_in_past=true — so a consumer UI told to hard-gate on that value
    would hide a valid pre-departure prediction for up to 59 minutes before every
    single departure.
    """
    import datetime as dt

    import pandas as pd

    from ml.serving import ServingContext, _departure_utc

    ctx = ServingContext(
        models=None,  # type: ignore[arg-type]
        bq=None,  # type: ignore[arg-type]
        gold="g",
        airports=pd.DataFrame(
            {"latitude": [41.98], "longitude": [-87.9], "tz": ["America/Chicago"]},
            index=pd.Index(["ORD"], name="iata"),
        ),
    )
    d = dt.date(2026, 8, 12)
    exact = _departure_utc(ctx, "ORD", d, "17:30")
    bucket = _departure_utc(ctx, "ORD", d, "17:30", hour_only=True)
    assert exact is not None and bucket is not None
    assert exact - bucket == dt.timedelta(minutes=30)
    assert bucket.minute == 0
    # and the whole point: at 17:05 the 17:30 flight is still in the future
    now_1705 = _departure_utc(ctx, "ORD", d, "17:05")
    assert now_1705 is not None and now_1705 < exact


def test_departure_utc_unknown_airport_is_none():
    import datetime as dt

    import pandas as pd

    from ml.serving import ServingContext, _departure_utc

    ctx = ServingContext(
        models=None,  # type: ignore[arg-type]
        bq=None,  # type: ignore[arg-type]
        gold="g",
        airports=pd.DataFrame(
            {"latitude": [], "longitude": [], "tz": []}, index=pd.Index([], name="iata")
        ),
    )
    assert _departure_utc(ctx, "XXX", dt.date(2026, 8, 12), "17:30") is None
