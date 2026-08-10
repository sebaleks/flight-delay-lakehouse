"""HTTP client for the predictor service.

`dashboard/` must NEVER `import ml` — the BI image carries only the `dashboard`
extra, and pulling in xgboost plus ~695 MB of artifacts would put a model load
on every cold start for every visitor who only wanted the delay map. All
inference goes over HTTPS through this module.

AUTH. The predictor is deployed private (--no-allow-unauthenticated): an open
/predict/batch is a free compute amplifier anyone could point at thousands of
flights. We mint a Google-signed ID token from ADC — the dashboard's Cloud Run
runtime service account, which holds roles/run.invoker. No key file, per
CLAUDE.md §2. Running locally, ADC is your `gcloud auth application-default
login` identity, so the same code path works.

THE CONSUMER PROJECTION. predict_one() returns a narrow dataclass rather than
the raw response, and deliberately drops two fields:

  * logreg_baseline_probability — UNCALIBRATED and class_weight-balanced, so
    systematically inflated. It is a comparison anchor for model work, and a
    number a consumer would read as a second opinion. It must not reach a page.
  * expected_delay_minutes — the regressor's held-out MAE is 18.99 and RMSE
    49.26. As a point estimate to a person it is the most misleading number the
    system can produce; the /outcome-mix table answers "how late?" honestly.
    (The ops page sums it across a whole airport-hour, where per-leg errors
    partially cancel — that path uses predict_batch, not this one.)

A unit test asserts neither key survives the projection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urlencode

import requests

log = logging.getLogger("dashboard.predict_client")

# A cold predictor deserializes ~695 MB of boosters. min-instances=0 keeps the
# project's zero-idle-cost posture, so the FIRST request after a quiet period
# genuinely takes this long — the page shows a "waking the model" state rather
# than a spinner that looks broken.
COLD_START_TIMEOUT_S = 90
WARM_TIMEOUT_S = 30


class PredictorUnavailable(RuntimeError):
    """The predictor could not be reached or refused the request.

    Raised rather than returning a sentinel so a page cannot accidentally render
    a missing prediction as a real one.
    """


@dataclass(frozen=True)
class ConsumerPrediction:
    """The safe subset of a prediction for a person to see."""

    delay_probability: float
    probability_calibration: str
    has_origin_weather: bool
    basis: dict = field(default_factory=dict)

    @property
    def flight_in_past(self) -> bool:
        return bool(self.basis.get("flight_in_past"))


def _id_token(audience: str) -> str | None:
    """A Google-signed ID token for the predictor, or None if ADC cannot mint one.

    None is not fatal: a locally-run predictor started with `uvicorn` has no
    auth at all, which is the normal development path. The request is simply
    sent unauthenticated and the server decides.
    """
    try:
        import google.auth.transport.requests
        import google.oauth2.id_token

        return google.oauth2.id_token.fetch_id_token(
            google.auth.transport.requests.Request(), audience
        )
    except Exception as exc:  # noqa: BLE001 - any ADC failure means "no token"
        log.info("no ID token available (%s) — calling the predictor unauthenticated", exc)
        return None


def _post(base_url: str, path: str, payload: dict, timeout: float) -> object:
    url = base_url.rstrip("/") + path
    headers = {"Content-Type": "application/json"}
    token = _id_token(base_url.rstrip("/"))
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.Timeout as exc:
        raise PredictorUnavailable(f"the predictor did not respond within {timeout:.0f}s") from exc
    except requests.RequestException as exc:
        raise PredictorUnavailable(f"could not reach the predictor: {exc}") from exc
    if r.status_code == 422:
        # the API's own validation message is the useful one (e.g. the
        # complete-or-absent rotation rule) — surface it rather than a generic
        raise PredictorUnavailable(f"the predictor rejected the request: {r.text[:300]}")
    if r.status_code >= 400:
        raise PredictorUnavailable(f"predictor returned HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


def _get(base_url: str, path: str, timeout: float) -> dict:
    url = base_url.rstrip("/") + path
    headers = {}
    token = _id_token(base_url.rstrip("/"))
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
    except requests.Timeout as exc:
        raise PredictorUnavailable(f"the predictor did not respond within {timeout:.0f}s") from exc
    except requests.RequestException as exc:
        raise PredictorUnavailable(f"could not reach the predictor: {exc}") from exc
    if r.status_code >= 400:
        # the server's own detail is the useful part (e.g. the replay
        # endpoint's "training window starts ..." refusal) — surface it,
        # exactly as _post does, rather than a bare status line
        raise PredictorUnavailable(f"predictor returned HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def predict_one(
    base_url: str,
    *,
    origin: str,
    dest: str,
    carrier: str,
    flight_date: date,
    dep_time: str,
    arr_time: str,
    distance: float | None = None,
    timeout: float = COLD_START_TIMEOUT_S,
) -> ConsumerPrediction:
    """Score one flight and return only what a person should see."""
    payload = {
        "origin": origin,
        "dest": dest,
        "carrier": carrier,
        "flight_date": flight_date.isoformat(),
        "dep_time": dep_time,
        "arr_time": arr_time,
    }
    if distance is not None:
        payload["distance"] = distance
    raw = _post(base_url, "/predict", payload, timeout)
    if not isinstance(raw, dict):
        raise PredictorUnavailable(f"unexpected response shape: {type(raw).__name__}")
    # explicit allowlist, not a blocklist: a field ADDED to the API later cannot
    # leak to a consumer page by default
    return ConsumerPrediction(
        delay_probability=float(raw["delay_probability"]),
        probability_calibration=str(raw.get("probability_calibration", "unknown")),
        has_origin_weather=bool(raw.get("has_origin_weather", False)),
        basis=dict(raw.get("prediction_basis") or {}),
    )


def replay_airport_day(
    base_url: str, *, origin: str, flight_date: date, timeout: float = COLD_START_TIMEOUT_S
) -> dict:
    """One airport's HELD-OUT day: per-flight predictions with labels alongside.

    The ops page's data source — replay mode, not live scoring: the server
    reads the held-out mart rows and 404s any training-window date. No
    consumer projection here: the ops page aggregates the raw probabilities
    (Σp per bank) and shows labels as reported outcomes, and its per-flight
    minutes are never rendered as point estimates.
    """
    q = urlencode({"origin": origin, "date": flight_date.isoformat()})
    raw = _get(base_url, f"/replay/airport-day?{q}", timeout)
    if not isinstance(raw, dict) or not isinstance(raw.get("flights"), list):
        raise PredictorUnavailable("unexpected replay response shape")
    return raw


def calibration(base_url: str, timeout: float = COLD_START_TIMEOUT_S) -> dict:
    """The held-out reliability table — the evidence behind the probability."""
    return _get(base_url, "/calibration", timeout)


def outcome_mix(base_url: str, timeout: float = COLD_START_TIMEOUT_S) -> dict:
    """Held-out P(arrival delay >= t) per probability band."""
    return _get(base_url, "/outcome-mix", timeout)


def health(base_url: str, timeout: float = 10) -> dict:
    """Liveness + which artifact run is being served. Short timeout on purpose:
    this is used to decide whether to show the 'waking the model' state."""
    return _get(base_url, "/health", timeout)
