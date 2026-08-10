"""Pure tests for how a probability is spoken. No creds, no network, no Streamlit.

The consumer page's whole job is to not mislead, so the wording rules and the
field allowlist are the things worth pinning.

    uv run --extra dashboard --group dev pytest dashboard/test_uncertainty.py
"""

from __future__ import annotations

import pytest

from dashboard.uncertainty import (
    BASE_RATE,
    basis_notes,
    natural_frequency,
    phrase,
    vs_base_rate,
)


@pytest.mark.parametrize(
    ("p", "expected_k"),
    [
        (0.0, 0),  # exact zero passes through
        (0.01, 1),  # never rounds a real risk down to "0 in 10"
        (0.04, 1),
        (0.14, 1),
        (0.16, 2),
        (0.3273, 3),
        (0.5, 5),
        (0.96, 9),  # never rounds up to "10 in 10"
        (1.0, 10),  # exact one passes through
    ],
)
def test_natural_frequency_rounding(p, expected_k):
    k, d = natural_frequency(p)
    assert (k, d) == (expected_k, 10)


def test_natural_frequency_never_promises_certainty():
    """0 in 10 reads as 'cannot happen' and 10 in 10 as 'will happen'. A model
    with held-out ECE 0.017 has not earned either sentence."""
    for p in (0.001, 0.02, 0.049):
        assert natural_frequency(p)[0] >= 1
    for p in (0.999, 0.98, 0.951):
        assert natural_frequency(p)[0] <= 9


def test_natural_frequency_rejects_out_of_range():
    for bad in (-0.01, 1.01, 42.0):
        with pytest.raises(ValueError):
            natural_frequency(bad)


def test_vs_base_rate_bands_are_ordered():
    """Wording must move monotonically with risk — a higher probability can
    never be described as closer to typical than a lower one."""
    order = [
        "well below",
        "somewhat below",
        "about the same",
        "somewhat above",
        "well above",
        "far above",
    ]
    seen = [vs_base_rate(p) for p in (0.05, 0.15, 0.20, 0.28, 0.40, 0.90)]
    idx = [next(i for i, o in enumerate(order) if s.startswith(o)) for s in seen]
    assert idx == sorted(idx), seen


def test_vs_base_rate_at_the_base_rate_says_typical():
    assert "about the same" in vs_base_rate(BASE_RATE)


def test_phrase_shows_both_sides():
    """A lone 'X% delayed' reads as a verdict; the pair reads as a distribution."""
    ph = phrase(0.3273)
    assert "3 in 10" in ph.headline
    assert "7 in 10" in ph.complement
    assert "33%" in ph.precise or "32%" in ph.precise
    assert "typical" in ph.comparison
    assert "cannot tell you" in ph.caveat


def test_phrase_complement_always_sums():
    for p in (0.02, 0.2, 0.5, 0.77, 0.98):
        ph = phrase(p)
        k = natural_frequency(p)[0]
        assert f"{k} in 10" in ph.headline
        assert f"{10 - k} in 10" in ph.complement


def test_basis_notes_past_flight_is_an_error():
    notes = basis_notes({"flight_in_past": True, "weather_horizon": "past"})
    assert notes[0][0] == "error"
    assert "already departed" in notes[0][1]


def test_basis_notes_beyond_horizon_warns_and_says_when_to_return():
    notes = basis_notes({"flight_in_past": False, "weather_horizon": "beyond_horizon"})
    assert notes[0][0] == "warning"
    assert "check back within 7 days" in notes[0][1].lower()


def test_basis_notes_forecast_path_has_no_weather_warning():
    notes = basis_notes(
        {
            "flight_in_past": False,
            "weather_horizon": "forecast",
            "rotation_context": "provided",
            "origin_density_source": "provided",
        }
    )
    assert notes == []


def test_basis_notes_typical_rotation_is_standard_copy_not_an_error():
    """This is the NORMAL consumer state — a consumer never knows the aircraft's
    rotation — so it must not be styled as something going wrong."""
    notes = basis_notes(
        {
            "flight_in_past": False,
            "weather_horizon": "forecast",
            "rotation_context": "typical_estimate",
        }
    )
    assert [lvl for lvl, _ in notes] == ["info"]


def test_consumer_projection_drops_the_dangerous_fields():
    """The two fields that must never reach a page: the uncalibrated logreg
    score, and the regressor point estimate whose error bar dwarfs it."""
    from dataclasses import asdict

    from dashboard.predict_client import ConsumerPrediction

    cp = ConsumerPrediction(
        delay_probability=0.33,
        probability_calibration="platt",
        has_origin_weather=True,
        basis={"flight_in_past": False},
    )
    keys = set(asdict(cp))
    assert "logreg_baseline_probability" not in keys
    assert "expected_delay_minutes" not in keys
    assert "features" not in keys
