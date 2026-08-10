"""Pure tests for the flight picker and the connection arithmetic.

No creds, no network, no Streamlit. The connection math is the part worth
pinning hardest: it is the only place the app turns two model outputs into
advice a traveller might act on.

    uv run --extra dashboard --group dev pytest dashboard/test_flights.py
"""

from __future__ import annotations

import pytest

from dashboard.flights import (
    DEFAULT_MCT_MIN,
    airport_label,
    code_from_label,
    connection_risk,
    filter_flights,
    layover_minutes,
)

BOARD = [
    {"carrier": "AA", "flight_number": "2842", "dest": "CLT", "dep_time": "05:00"},
    {"carrier": "AA", "flight_number": "370", "dest": "DFW", "dep_time": "05:00"},
    {"carrier": "NK", "flight_number": "19", "dest": "FLL", "dep_time": "05:00"},
    {"carrier": "UA", "flight_number": "1900", "dest": "SFO", "dep_time": "17:30"},
    {"carrier": "UA", "flight_number": "284", "dest": "LAX", "dep_time": "21:15"},
]

# shaped like /outcome-mix: 10 bands, P(delay >= t) per band
BINS = [
    {
        "lo": round(i / 10, 1),
        "hi": round((i + 1) / 10, 1),
        "n": 1000,
        "exceedance": {
            "15": 0.05 + i * 0.09,
            "30": 0.03 + i * 0.07,
            "60": 0.02 + i * 0.05,
            "90": 0.01 + i * 0.035,
            "120": 0.005 + i * 0.025,
        },
    }
    for i in range(10)
]


def test_airport_label_prefers_name_then_city_then_code():
    assert airport_label("SFO", "San Francisco International Airport", "San Francisco") == (
        "San Francisco International Airport (SFO)"
    )
    assert airport_label("XYZ", None, "Somewhere") == "Somewhere (XYZ)"
    assert airport_label("XYZ", "", "") == "XYZ"
    assert airport_label("XYZ") == "XYZ"


def test_code_round_trips_out_of_a_label():
    for code in ("SFO", "ORD", "LAX"):
        assert code_from_label(airport_label(code, f"{code} International Airport")) == code
    assert code_from_label("ORD") == "ORD"


def test_filters_are_optional_and_compose():
    assert len(filter_flights(BOARD)) == len(BOARD)  # nothing set -> everything
    assert len(filter_flights(BOARD, carrier="AA")) == 2
    assert len(filter_flights(BOARD, carrier="AA", dest="CLT")) == 1
    assert filter_flights(BOARD, carrier="AA", dest="FLL") == []


def test_empty_filter_values_are_ignored_not_matched():
    """A half-filled form must still return rows — '' must not mean 'match nothing'."""
    assert len(filter_flights(BOARD, carrier=None, dest=None, flight_number="")) == len(BOARD)


def test_flight_number_matches_as_a_prefix():
    # typing "19" finds both 19 and 1900, which is what a half-typed booking does
    assert {f["flight_number"] for f in filter_flights(BOARD, flight_number="19")} == {"19", "1900"}
    # prefix, deliberately: "284" keeps 2842 too. The user picks from the
    # remaining list, so a superset is helpful while a premature exact match
    # would hide their flight if they mistyped one digit.
    assert {f["flight_number"] for f in filter_flights(BOARD, flight_number="284")} == {
        "284",
        "2842",
    }
    assert {f["flight_number"] for f in filter_flights(BOARD, flight_number="2842")} == {"2842"}


def test_flight_number_ignores_leading_zeros_on_both_sides():
    """Boarding passes print '0019'; the mart stores '19'. Neither must miss.

    Zeros are stripped on both sides before the prefix compare, so '0019' still
    reaches flight 19 (and, being a prefix, 1900 as well — the user picks from
    what is left).
    """
    padded_board = [{"carrier": "NK", "flight_number": "0019", "dest": "FLL", "dep_time": "05:00"}]
    assert len(filter_flights(padded_board, flight_number="19")) == 1
    found = {f["flight_number"] for f in filter_flights(BOARD, flight_number="0019")}
    assert "19" in found


def test_departure_window_filters_inclusively():
    assert len(filter_flights(BOARD, dep_from="00:00", dep_to="12:00")) == 3
    assert len(filter_flights(BOARD, dep_from="17:00", dep_to="24:00")) == 2
    assert len(filter_flights(BOARD, dep_from="05:00", dep_to="05:00")) == 3


def test_layover_wraps_past_midnight():
    assert layover_minutes("14:00", "16:30") == 150
    # the bug this pins: 23:40 -> 00:25 must be 45 minutes, not -1395
    assert layover_minutes("23:40", "00:25") == 45
    assert layover_minutes("22:00", "06:00") == 480


def test_connection_risk_uses_the_largest_threshold_at_or_below_slack():
    """P(delay >= t) decreases in t, so choosing t <= slack OVERSTATES risk —
    the safe direction for someone judging a tight connection."""
    # layover 120 -> slack 90 -> threshold 90
    r = connection_risk("12:00", "14:00", BINS, 0.35)
    assert r.layover_min == 120
    assert r.slack_min == 120 - DEFAULT_MCT_MIN
    assert r.threshold_min == 90
    assert r.conservative is False  # threshold == slack exactly
    assert r.probability == pytest.approx(BINS[3]["exceedance"]["90"])


def test_connection_risk_flags_when_it_overstates():
    # layover 110 -> slack 80 -> largest usable threshold is 60, below slack
    r = connection_risk("12:00", "13:50", BINS, 0.35)
    assert r.slack_min == 80
    assert r.threshold_min == 60
    assert r.conservative is True
    assert "overstates" in r.note


def test_connection_risk_says_so_when_slack_is_below_every_threshold():
    # layover 40 -> slack 10, under the 15-minute floor: the real risk is HIGHER
    r = connection_risk("12:00", "12:40", BINS, 0.35)
    assert r.slack_min == 10
    assert r.threshold_min == 15
    assert "HIGHER" in r.note


def test_connection_risk_refuses_an_impossible_layover():
    """A layover shorter than the connect time is not a risk estimate — it is a
    broken itinerary, and must not render as a probability."""
    r = connection_risk("12:00", "12:20", BINS, 0.35)
    assert r.slack_min <= 0
    assert r.probability is None
    assert "cannot" in r.note or "not a connection you can rely on" in r.note


def test_connection_risk_rises_with_leg1_risk():
    """Monotonicity: a riskier first leg can never produce a safer connection."""
    low = connection_risk("12:00", "14:00", BINS, 0.05)
    high = connection_risk("12:00", "14:00", BINS, 0.95)
    assert low.probability is not None and high.probability is not None
    assert high.probability > low.probability


def test_connection_risk_falls_as_layover_grows():
    """More slack must never look worse."""
    tight = connection_risk("12:00", "13:00", BINS, 0.35)
    loose = connection_risk("12:00", "15:00", BINS, 0.35)
    assert loose.probability < tight.probability


def test_connection_risk_handles_a_band_with_no_data():
    empty = [{"lo": 0.0, "hi": 1.0, "n": 0, "exceedance": {"15": None}}]
    r = connection_risk("12:00", "14:00", empty, 0.35)
    assert r.probability is None


def test_long_layover_is_an_upper_bound_not_an_estimate():
    """The bug this pins: 209 minutes of slack scored against the 120-minute
    threshold reported 24.9% — telling a traveller a comfortable 3.5-hour
    layover was a one-in-four risk. It must come back flagged as a bound."""
    r = connection_risk("12:00", "16:00", BINS, 0.85)  # layover 240 -> slack 210
    assert r.slack_min == 210
    assert r.threshold_min == 120  # the largest we measured
    assert r.upper_bound is True
    assert "BELOW" in r.note


def test_slack_inside_the_measured_range_is_not_an_upper_bound():
    r = connection_risk("12:00", "14:00", BINS, 0.85)  # slack 90, measured exactly
    assert r.upper_bound is False


def test_carrier_label_names_the_airline_and_marks_regionals():
    from dashboard.flights import carrier_label

    assert carrier_label("UA", "United Airlines") == "United Airlines (UA)"
    # a passenger books "American" and flies Envoy — the dropdown must say so
    assert carrier_label("MQ", "Envoy Air", True) == "Envoy Air (MQ) - regional"
    # missing name must never render an empty option
    assert carrier_label("ZZ", None) == "ZZ"
    assert carrier_label("ZZ", "") == "ZZ"


def test_carrier_code_round_trips_including_the_regional_suffix():
    from dashboard.flights import carrier_label, code_from_label

    for code, name, reg in [("UA", "United Airlines", False), ("MQ", "Envoy Air", True)]:
        assert code_from_label(carrier_label(code, name, reg)) == code
