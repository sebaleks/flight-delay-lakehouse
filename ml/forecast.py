"""NWS/NDFD forecast client for serving-time origin weather.

Source choice: api.weather.gov (National Weather Service / NDFD gridded
forecast). Justification: official government forecasts, no API key, free,
and its coverage (CONUS + Alaska + Hawaii + Puerto Rico + Guam) matches the
BTS domestic network — the same territory our airports seed spans.

TIME REFERENCE — matches training exactly: the models are trained on the
last hourly ISD observation AT OR BEFORE the scheduled departure hour, so
serving fetches the gridded forecast value valid AT the scheduled departure
hour (the layer interval containing it). Training and serving now share one
reference point — the scheduled departure hour — so the remaining train/serve
gap is ONLY forecast-vs-observed error (see ml/serving.py).

Coverage/missing handling: any failure — a point outside NWS grids, a date
beyond the ~7-day forecast horizon, an API error, no layer value at the hour
— returns ``has_origin_weather = 0.0`` with every weather value NaN, which is
EXACTLY the training NULL path (the mart leaves weather NULL with the flag
false when no observation lands in the 3-hour lookback; XGBoost consumes NaN
natively, the logreg pipeline imputes train medians).

Field mapping (point-in-time training features -> NDFD layers), units
converted to the mart's (F, knots, statute miles, inches):
    origin_temp_f          temperature at the departure hour (C -> F)
    origin_dewpoint_f      dewpoint at the departure hour (C -> F)
    origin_wind_speed_kn   windSpeed at the hour (km/h -> kn)
    origin_gust_kn         windGust at the hour; NO gust value forecast ->
                           0.0 with origin_gust_reported = 0.0 — the same
                           absent-as-calm + indicator encoding training used
                           for observations without a gust group
    origin_visibility_mi   visibility at the hour (m -> mi), right-censored
                           at 10.0 exactly like the mart; NDFD publishes the
                           layer primarily during restriction, so no value at
                           a resolved hour -> 10.0 (unrestricted)
    origin_precip_1h_in    quantitativePrecipitation: the QPF interval
                           containing the hour, apportioned per hour when the
                           interval is longer than PT1H (mm -> in). Training
                           used strictly 1-hour accumulations; a forecast QPF
                           spread uniformly over its window is the honest
                           hourly expectation — a forecast-side approximation,
                           part of the forecast-vs-observed gap, not a new
                           time-reference mismatch. No QPF interval -> 0.0
                           (forecasts omit dry periods; mirrors the training
                           no-precip-groups -> 0.0 convention).
    origin_had_*           phenomena from the `weather` layer interval(s)
                           covering the departure hour
    has_origin_weather     1.0 when the grid + hour resolve; else NULL path
"""

from __future__ import annotations

import logging
import math
import os
import re
from datetime import datetime, timedelta

import requests

log = logging.getLogger("ml.forecast")

API_ROOT = "https://api.weather.gov"


def _user_agent() -> str:
    """NWS asks for a User-Agent that identifies the client AND gives them a way
    to reach you if it misbehaves; they use it to contact operators before
    throttling. A bare app name with no contact does not meet that ask, which is
    fine for a laptop but not for a deployed service making bulk requests.

    Set NWS_CONTACT_EMAIL. Unset falls back to the old string with an explicit
    'no contact' marker and a warning, so a misconfigured deploy is visible in
    the logs rather than silently rude.
    """
    contact = os.environ.get("NWS_CONTACT_EMAIL", "").strip()
    if not contact:
        log.warning(
            "NWS_CONTACT_EMAIL is unset — api.weather.gov asks for a contactable "
            "User-Agent. Set it before running this as a deployed service."
        )
        return "flight-delay-lakehouse (no contact configured)"
    return f"flight-delay-lakehouse ({contact})"


HEADERS = {
    "User-Agent": _user_agent(),
    "Accept": "application/geo+json",
}
TIMEOUT = (10, 30)


def _session() -> requests.Session:
    """One retry-wrapped session: a single transient 5xx must not silently
    degrade a flight to the no-weather path."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    s = requests.Session()
    retry = Retry(total=2, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update(HEADERS)
    return s


SESSION = _session()

WEATHER_FEATURE_DEFAULTS: dict[str, float] = {
    "origin_temp_f": math.nan,
    "origin_dewpoint_f": math.nan,
    "origin_wind_speed_kn": math.nan,
    "origin_gust_kn": math.nan,
    "origin_gust_reported": math.nan,
    "origin_visibility_mi": math.nan,
    "origin_precip_1h_in": math.nan,
    "origin_had_fog": math.nan,
    "origin_had_rain_drizzle": math.nan,
    "origin_had_snow_ice_pellets": math.nan,
    "origin_had_thunder": math.nan,
}

_DURATION = re.compile(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?")

# NDFD layer units this client's conversions assume. A layer served in any
# other unit is treated as MISSING (fail toward the NULL path) rather than
# converted wrongly.
EXPECTED_UOM = {
    "temperature": "wmoUnit:degC",
    "dewpoint": "wmoUnit:degC",
    "windSpeed": "wmoUnit:km_h-1",
    "windGust": "wmoUnit:km_h-1",
    "visibility": "wmoUnit:m",
    "quantitativePrecipitation": "wmoUnit:mm",
}


def _parse_valid_time(valid: str) -> tuple[datetime, timedelta]:
    """Parse NDFD 'start/ISO-duration' strings like 2026-07-30T18:00:00+00:00/PT6H.
    Days/hours/minutes are handled; an unrecognized or zero duration yields
    timedelta(0), so the interval can never contain an instant — the value is
    skipped (fail closed toward missing), never mis-attributed to the hour."""
    start_s, dur_s = valid.split("/")
    start = datetime.fromisoformat(start_s)
    m = _DURATION.fullmatch(dur_s)
    if not m:
        return start, timedelta(0)
    days = int(m.group(1) or 0)
    hours = int(m.group(2) or 0)
    minutes = int(m.group(3) or 0)
    return start, timedelta(days=days, hours=hours, minutes=minutes)


def _interval_at(props: dict, name: str, t: datetime) -> tuple[float, timedelta] | None:
    """The named layer's value whose validity interval contains t, with its
    duration. When the layer's uom does not match the unit this client
    converts from, the layer is treated as MISSING — never converted with
    the wrong factor."""
    layer = props.get(name) or {}
    expected = EXPECTED_UOM.get(name)
    if expected is not None and layer.get("uom") not in (None, expected):
        log.warning(
            "%s uom %r != expected %r — treating layer as missing",
            name,
            layer.get("uom"),
            expected,
        )
        return None
    for v in layer.get("values", []):
        if v.get("value") is None:
            continue
        start, dur = _parse_valid_time(v["validTime"])
        if start <= t < start + dur:
            return float(v["value"]), dur
    return None


def _value_at(props: dict, name: str, t: datetime) -> float | None:
    hit = _interval_at(props, name, t)
    return hit[0] if hit else None


def _phenomena_at(layer: dict, t: datetime) -> set[str]:
    seen: set[str] = set()
    for v in layer.get("values", []):
        start, dur = _parse_valid_time(v["validTime"])
        if not start <= t < start + dur:
            continue
        for cond in v.get("value") or []:
            if cond.get("weather"):
                seen.add(str(cond["weather"]))
    return seen


NULL_PATH = {**WEATHER_FEATURE_DEFAULTS, "has_origin_weather": 0.0}


def fetch_grid(lat: float, lon: float) -> dict | None:
    """The full gridded-forecast properties for one point (covers every hour
    in the horizon — fetch once per airport, extract many hours). None on any
    failure; retried once on transient 5xx by the session."""
    try:
        point = SESSION.get(f"{API_ROOT}/points/{lat:.4f},{lon:.4f}", timeout=TIMEOUT)
        point.raise_for_status()
        grid_url = point.json()["properties"]["forecastGridData"]
        grid = SESSION.get(grid_url, timeout=TIMEOUT)
        grid.raise_for_status()
        return grid.json()["properties"]
    except Exception as exc:
        log.warning("grid fetch failed for (%.3f, %.3f): %s", lat, lon, exc)
        return None


def features_at_hour(props: dict | None, dep_utc: datetime) -> dict:
    """Pure extraction: the feature dict for one UTC instant from a fetched
    grid. Any failure or unresolved hour -> a FRESH all-NaN NULL path (never
    a partially populated dict — an exception midway must not leak the
    values assigned before it)."""
    if props is None:
        return dict(NULL_PATH)
    features = dict(WEATHER_FEATURE_DEFAULTS)
    try:
        temp_c = _value_at(props, "temperature", dep_utc)
        if temp_c is None:
            log.info("no forecast value at %s (beyond horizon?)", dep_utc)
            return dict(NULL_PATH)

        features["origin_temp_f"] = temp_c * 9 / 5 + 32
        dew_c = _value_at(props, "dewpoint", dep_utc)
        if dew_c is not None:
            features["origin_dewpoint_f"] = dew_c * 9 / 5 + 32
        wind = _value_at(props, "windSpeed", dep_utc)
        if wind is not None:
            features["origin_wind_speed_kn"] = wind / 1.852  # km/h -> kn
        gust = _value_at(props, "windGust", dep_utc)
        # absent-as-calm + indicator: the training encoding for an observation
        # that reports no gust group
        features["origin_gust_kn"] = gust / 1.852 if gust is not None else 0.0
        features["origin_gust_reported"] = 1.0 if gust is not None else 0.0
        vis = _value_at(props, "visibility", dep_utc)
        # NDFD publishes visibility primarily during RESTRICTION; absent at
        # the hour with the grid resolved means unrestricted -> 10.0, the
        # mart's right-censored clear-day value (training rows with an
        # observation almost never carry NaN visibility, so NaN here would
        # be a serving-only pattern)
        features["origin_visibility_mi"] = min(vis / 1609.344, 10.0) if vis is not None else 10.0
        qpf = _interval_at(props, "quantitativePrecipitation", dep_utc)
        if qpf is None:
            features["origin_precip_1h_in"] = 0.0
        else:
            mm, dur = qpf
            hours = max(dur.total_seconds() / 3600, 1.0)
            features["origin_precip_1h_in"] = (mm / hours) / 25.4

        phenomena = _phenomena_at(props.get("weather", {}), dep_utc)
        features["origin_had_fog"] = float(any("fog" in p for p in phenomena))
        # freezing LIQUID precip counts as rain — the FRSHTT convention the
        # training decoder applies (silver_isd_hourly: MW 56-57/66-67 and
        # AW 47-48 land in had_rain_drizzle, never the snow flag)
        features["origin_had_rain_drizzle"] = float(
            any(
                p in ("rain", "drizzle", "rain_showers", "freezing_rain", "freezing_drizzle")
                for p in phenomena
            )
        )
        features["origin_had_snow_ice_pellets"] = float(
            any(p in ("snow", "sleet", "ice_pellets", "snow_showers") for p in phenomena)
        )
        features["origin_had_thunder"] = float(any("thunder" in p for p in phenomena))
        features["has_origin_weather"] = 1.0
        return features
    except Exception as exc:  # any failure -> a FRESH all-NaN NULL path
        log.warning("forecast extraction failed at %s: %s", dep_utc, exc)
        return dict(NULL_PATH)


def fetch_hourly_forecast(lat: float, lon: float, dep_utc: datetime) -> dict:
    """One-shot convenience: fetch the grid and extract one hour."""
    return features_at_hour(fetch_grid(lat, lon), dep_utc)
