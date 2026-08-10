"""Forecast-driven inference over the trained artifacts: score FUTURE flights.

WHY THIS IS LEAK-FREE (CLAUDE.md §9): the pre-departure boundary requires
every predictor to be knowable before the flight departs. For a flight that
has not yet departed, a weather FORECAST issued now predates departure by
construction — it is legitimately pre-departure information. Training uses
the last OBSERVED hourly reading at or before the scheduled departure hour;
serving substitutes the FORECAST valid at that same hour. Every other
feature (schedule attributes, smoothed training-window historical rates,
holiday flags) is knowable arbitrarily far in advance.

TRAIN/SERVE MISMATCH (honest, enumerated): training and serving reference
the SAME instant — the scheduled departure hour — so the daily-weather
era's time misalignment is GONE. The remaining gaps: (a) weather is
forecast-vs-observed (NDFD forecasts of the observations training used,
plus the documented QPF apportionment); (b) rotation context depends on
what the caller knows — with a planning feed it is the same schedule data
training used; without one the flight takes the flagged typical-profile
estimate, and origin departure density falls back to a historical
(origin, hour, weekday) median. None of these gaps changes the held-out test
metrics — those were computed entirely on observed data and stand as
reported — and they are the price of scoring flights that have not
happened yet.

Feature parity with training:
  * hist_* rates are read from ml_flight_features itself — they are constant
    within an entity (verified property), so ANY_VALUE per entity reproduces
    the training values byte-exactly with zero formula duplication. Entities
    absent from the mart (new route etc.) stay NaN, the training NULL path.
  * cascade/rotation features are SCHEDULE-derived (knowable at booking):
    callers with the aircraft's planned rotation pass it per request (the
    demo passes its proxy schedule's historical rotation; production would
    use the airline's planning feed); callers WITHOUT it get the TYPICAL
    rotation profile — training medians, flagged per response. Why not
    NULL: under the tail-swap restriction, all-NULL rotation is
    in-distribution but MEANS "operated linkage was swap-restructured" —
    a merely-unknown future plan is not swap-shaped, so NULL would
    misclassify it; the typical profile estimates the real unknown plan.
    The band derivation mirrors int_aircraft_rotation (pinned by the dbt
    guard on the SQL side); band/position hist values come from the mart
    byte-exactly.
    origin_dep_density_hour without a caller value is ESTIMATED as the
    (origin, hour, weekday) median over the mart — a published-schedule
    quantity we lack a future feed for; part of the serve-side gap like
    forecast-vs-observed weather.
  * holiday flags use the same `holidays` library + one-year padding as the
    seed generator.
  * day_of_week uses the BTS convention (isoweekday: 1 = Monday).
  * The assembled frame's columns are asserted equal to the model's stored
    feature schema before any prediction — a mismatch raises, never scores.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import xgboost as xgb
from google.api_core.exceptions import NotFound
from google.cloud import bigquery

from ingestion.config import require_env
from ml import features as f
from ml.forecast import NULL_PATH, features_at_hour, fetch_grid
from ml.train import ARTIFACT_ROOT, LOGREG_INPUT_COLUMNS

log = logging.getLogger("ml.serving")

# the serving lookup layer (dbt/models/gold/ml/serving_*.sql) — tiny tables that
# materialize what the request path used to query per call. Serving no longer
# reads ml_flight_features at all.
ENTITY_PROFILE = "serving_entity_profile"
DENSITY_PROFILE = "serving_density_profile"
TYPICAL_ROTATION = "serving_typical_rotation"
# the four entity grains carrying hist_* triples. A frozenset, not a mapping:
# the values used to be the mart column name interpolated into the per-grain
# SQL, and that query is gone — every remaining use is membership.
HIST_GRAINS = frozenset({"route", "carrier", "origin", "dest"})


class SchemaMismatchError(RuntimeError):
    """The assembled features do not match the model's stored schema."""


@dataclass
class FlightRequest:
    origin: str
    dest: str
    carrier: str
    flight_date: date
    dep_time: str  # "HH:MM" local
    arr_time: str  # "HH:MM" local
    distance: float | None = None
    # OPTIONAL aircraft-rotation context — all SCHEDULE-derived (knowable at
    # booking; CLAUDE.md §9): which leg of the day this is for the aircraft,
    # its scheduled turnaround, and the inbound leg's schedule. In production
    # this comes from the airline's planned-rotation feed; the demo passes
    # its proxy schedule's historical rotation. Callers WITHOUT the context
    # get the TYPICAL rotation profile (training medians) — flagged in the
    # response as an estimate; see _load_rotation_hist for why not NULL.
    # The FastAPI FlightIn model enforces this as COMPLETE-or-absent (once
    # rotation_position is given, legs_today is required and — for position >= 2
    # — the inbound triple too), and rotation_context="provided" reports that
    # guarantee. A hand-built FlightRequest bypasses that validation: assembly
    # still degrades to a coherent vector (legs floored at the position, missing
    # attributes filled from the typical profile), but such a partial request
    # should honor the same completeness for the response flag to be meaningful.
    rotation_position: int | None = None
    legs_today: int | None = None
    sched_turnaround_min: float | None = None
    inbound_distance: float | None = None
    inbound_crs_elapsed_min: float | None = None
    origin_dep_density_hour: float | None = None


@dataclass
class Models:
    clf: xgb.XGBClassifier
    reg: xgb.XGBRegressor
    logreg: object
    # probability calibrator (ml.calibration.Calibrator): the Platt map that
    # turns the classifier's raw, recall-inflated scores into calibrated
    # frequencies. Strictly monotonic -> leaves ranking (ROC/PR-AUC) unchanged.
    calibrator: object
    artifacts_dir: Path


@dataclass
class ServingContext:
    models: Models
    bq: bigquery.Client
    gold: str
    airports: pd.DataFrame  # iata -> latitude, longitude, tz
    forecast_cache: dict = field(default_factory=dict)
    # (band -> 3 hist values), (position key -> 3 hist values), plus the
    # "typical" rotation profile: read once at startup from
    # serving_entity_profile / serving_typical_rotation (constant within
    # entity — byte-exact training values), see _load_rotation_hist
    rotation_hist: dict = field(default_factory=dict)
    # PRELOADED SERVING LOOKUPS — the request path issues ZERO BigQuery
    # queries. Each is read whole at startup from a tiny gold table that
    # materializes the query this used to run per request:
    #   hist            {grain: {entity_key: {3 hist values}}}   ~8.3k entries
    #   route_distance  {route: distance}                        7.5k entries
    #   density         {(origin, hour, weekday): median}        ~35k entries
    # They change only when the mart is rebuilt, which is exactly when the
    # process is redeployed — so a cache with a TTL would buy nothing.
    hist: dict = field(default_factory=dict)
    route_distance: dict = field(default_factory=dict)
    density: dict = field(default_factory=dict)
    # training category vocabulary per categorical feature: unseen values
    # must become MISSING before prediction — xgboost >= 3 hard-errors on a
    # category absent from the trained encoder instead of routing it to the
    # default direction; see assemble_features
    category_vocab: dict = field(default_factory=dict)
    # the same vocabulary pre-sorted, because pd.Categorical needs an ORDER and
    # re-sorting 7,539 routes on every request is pure waste once the
    # vocabulary is fixed at startup. Set for membership, list for categories.
    category_order: dict = field(default_factory=dict)


def load_models(artifacts_dir: Path | None = None) -> Models:
    """Load the self-contained artifacts (no retraining, no side metadata) and
    assert their stored feature schemas match the canonical registry."""
    if artifacts_dir is None:
        if not ARTIFACT_ROOT.is_dir():
            raise FileNotFoundError(f"artifact root {ARTIFACT_ROOT} does not exist — train first")
        # only COMPLETE runs are candidates — all FOUR artifacts, the same
        # contract training writes: a run interrupted between saves must not
        # win selection and crash startup with a confusing load error
        required = (
            "xgb_classifier.ubj",
            "xgb_regressor.ubj",
            "logreg_pipeline.joblib",
            "calibrator.joblib",
        )
        runs = sorted(
            d
            for d in ARTIFACT_ROOT.iterdir()
            if d.is_dir() and all((d / name).exists() for name in required)
        )
        if not runs:
            raise FileNotFoundError(f"no complete artifact runs under {ARTIFACT_ROOT}")
        artifacts_dir = runs[-1]
    clf = xgb.XGBClassifier()
    clf.load_model(artifacts_dir / "xgb_classifier.ubj")
    reg = xgb.XGBRegressor()
    reg.load_model(artifacts_dir / "xgb_regressor.ubj")
    logreg = joblib.load(artifacts_dir / "logreg_pipeline.joblib")
    calibrator = joblib.load(artifacts_dir / "calibrator.joblib")

    for name, booster in (("classifier", clf), ("regressor", reg)):
        stored = booster.get_booster().feature_names
        if stored != list(f.FEATURES):
            raise SchemaMismatchError(
                f"xgb {name} stored schema != canonical FEATURES; "
                f"stored={stored} expected={list(f.FEATURES)}"
            )
    logreg_cols = list(logreg.feature_names_in_)
    if logreg_cols != LOGREG_INPUT_COLUMNS:
        raise SchemaMismatchError(
            f"logreg pipeline schema != LOGREG_INPUT_COLUMNS; stored={logreg_cols}"
        )
    log.info("artifacts loaded from %s; schemas verified", artifacts_dir.name)
    return Models(
        clf=clf, reg=reg, logreg=logreg, calibrator=calibrator, artifacts_dir=artifacts_dir
    )


def build_context(artifacts_dir: Path | None = None) -> ServingContext:
    bq = bigquery.Client(project=require_env("GCP_PROJECT_ID"))
    gold = require_env("BQ_GOLD_DATASET")
    airports = (
        bq.query(
            f"select airport_key as iata, latitude, longitude, tz "
            f"from `{bq.project}.{gold}.dim_airport`"
        )
        .to_dataframe()
        .set_index("iata")
    )
    ctx = ServingContext(models=load_models(artifacts_dir), bq=bq, gold=gold, airports=airports)
    # the rotation levels are RETURNED rather than written onto ctx, so the
    # dependency between these two loads is a visible argument instead of an
    # undeclared ordering contract on a mutable field
    rotation_levels = _load_serving_lookups(ctx)
    ctx.rotation_hist = _load_rotation_hist(ctx, rotation_levels)
    return ctx


def _missing_lookup_error(table: str) -> RuntimeError:
    return RuntimeError(
        f"{table} is missing or empty — build the serving lookups BEFORE deploying "
        "a serving image against this dataset: dbt build -s serving_entity_profile "
        "serving_density_profile serving_typical_rotation"
    )


def _load_serving_lookups(ctx: ServingContext) -> dict:
    """Read the whole serving lookup layer once, into plain dicts.

    Replaces the five per-request queries and the startup vocab union. Those
    scanned the 20.2M-row mart without partition pruning — ~2.7 GB per
    /predict call — to fetch a few thousand constants that only change on a
    dbt rebuild. serving_entity_profile is ~8.3k rows.

    The category vocabulary falls out of the same table for free: the training
    vocabulary at a level IS the set of entity keys present at that level.

    Returns the two rotation levels (band / position) read on the same pass,
    for _load_rotation_hist to combine with the typical profile.
    """
    try:
        rows = list(
            ctx.bq.query(
                f"select entity_level, entity_key, hist_arr_del15_rate, "
                f"hist_avg_arr_delay_minutes, hist_n_flights, distance "
                f"from `{ctx.bq.project}.{ctx.gold}.{ENTITY_PROFILE}`"
            ).result()
        )
    except NotFound as exc:
        # a gold dataset built before the lookup layer existed: surface the
        # actionable message, not a bare 404 from three frames down
        raise _missing_lookup_error(ENTITY_PROFILE) from exc
    if not rows:
        raise _missing_lookup_error(ENTITY_PROFILE)
    ctx.hist = {g: {} for g in HIST_GRAINS}
    ctx.route_distance = {}
    ctx.category_vocab = {}
    # the two rotation levels live in the same table; collect them on this one
    # pass rather than re-querying it (see _load_rotation_hist)
    rotation: dict = {"band": {}, "pos": {}}
    rotation_levels = {"turnaround_band": "band", "rotation_position": "pos"}
    for r in rows:
        level, key = r["entity_level"], r["entity_key"]
        # keyed by the mart's own column names so the assembled row dict is
        # identical to what the old per-grain query produced
        if level in HIST_GRAINS:
            ctx.hist[level][key] = {
                f"hist_{level}_arr_del15_rate": r["hist_arr_del15_rate"],
                f"hist_{level}_avg_arr_delay_minutes": r["hist_avg_arr_delay_minutes"],
                f"hist_{level}_n_flights": r["hist_n_flights"],
            }
            ctx.category_vocab.setdefault(level, set()).add(key)
        elif level in rotation_levels:
            rotation[rotation_levels[level]][key] = {
                "rate": r["hist_arr_del15_rate"],
                "avg_min": r["hist_avg_arr_delay_minutes"],
                "n": r["hist_n_flights"],
            }
        if level == "route" and r["distance"] is not None:
            ctx.route_distance[key] = float(r["distance"])
    missing = [g for g in HIST_GRAINS if not ctx.hist[g]]
    missing += [lv for lv, k in rotation_levels.items() if not rotation[k]]
    if missing:
        raise RuntimeError(f"{ENTITY_PROFILE} has no rows for level(s) {sorted(missing)}")

    # pd.Categorical needs the categories in a stable ORDER; sorting the 7,539
    # routes on every request would be pure waste now that the vocabulary is
    # fixed at startup. Sort once here, keep the set for the membership test.
    ctx.category_order = {c: sorted(v) for c, v in ctx.category_vocab.items()}

    try:
        ctx.density = {
            (r["origin"], int(r["crs_dep_hour"]), int(r["day_of_week"])): float(r["density_median"])
            for r in ctx.bq.query(
                f"select origin, crs_dep_hour, day_of_week, density_median "
                f"from `{ctx.bq.project}.{ctx.gold}.{DENSITY_PROFILE}`"
            ).result()
        }
    except NotFound as exc:
        raise _missing_lookup_error(DENSITY_PROFILE) from exc
    if not ctx.density:
        raise _missing_lookup_error(DENSITY_PROFILE)
    log.info(
        "serving lookups loaded: hist %s, %d route distances, %d density keys",
        {g: len(v) for g, v in ctx.hist.items()},
        len(ctx.route_distance),
        len(ctx.density),
    )
    return rotation


# Mirrors the band derivation in int_aircraft_rotation.sql (a small necessary
# duplicate, like the holiday-flag mirror; the dbt guard pins the SQL side).
def _turnaround_band(has_inbound: bool, turnaround: float | None) -> str:
    if not has_inbound or turnaround is None:
        return "no_inbound"
    if turnaround < 35:
        return "lt_35"
    if turnaround < 60:
        return "35_60"
    if turnaround < 120:
        return "60_120"
    return "ge_120"


def _load_rotation_hist(ctx: ServingContext, rotation_levels: dict) -> dict:
    """The turnaround-band and rotation-position hist triples, plus the
    'typical rotation profile' (training medians of the rotation schedule
    attributes) used when a caller provides no context.

    Both now come from the serving lookup tables rather than from aggregates
    issued at startup. The values are unchanged in kind — the band and
    position levels of serving_entity_profile are the SAME any_value(...)
    group-by this used to run — but the medians are now EXACT
    (percentile_disc) instead of approx_quantiles. That is a deliberate fix,
    not a port: the approximation was measured returning different answers on
    identical data across runs, so the typical profile — and therefore every
    prediction made without rotation context — depended on which process
    served it. See serving_typical_rotation.sql for the measurements.

    Why medians and not NaN: the mart has essentially no tail-unknown rows
    (completed flights carry tails), so NaN in these columns sits OUTSIDE the
    training distribution and empirically produces garbage scores;
    unknown-but-knowable schedule facts are instead estimated with training
    medians — the same epistemic move as the density estimator — and the
    response flags the estimate.
    """
    # band/pos were collected on the single entity-profile read and handed in;
    # only the one-row median table is left to fetch
    out: dict = {"band": rotation_levels["band"], "pos": rotation_levels["pos"]}
    try:
        med = list(
            ctx.bq.query(
                f"select typical_rotation_position as pos, typical_legs_today as legs, "
                f"typical_sched_turnaround_min as turn, typical_inbound_distance as dist, "
                f"typical_inbound_crs_elapsed_min as elapsed, typical_density as density "
                f"from `{ctx.bq.project}.{ctx.gold}.{TYPICAL_ROTATION}`"
            ).result()
        )
    except NotFound as exc:
        raise _missing_lookup_error(TYPICAL_ROTATION) from exc
    if len(med) > 1:
        # the model is contractually one row and _load_rotation_hist pins every
        # context-less prediction to med[0]; more than one row would make that
        # choice depend on scan order — the exact class of bug the exact-median
        # change removed. A dbt singular test pins this on the SQL side too.
        raise RuntimeError(f"{TYPICAL_ROTATION} returned {len(med)} rows; it must be exactly 1")
    if not med or any(
        med[0][k] is None for k in ("pos", "legs", "turn", "dist", "elapsed", "density")
    ):
        raise RuntimeError(
            f"{TYPICAL_ROTATION} is empty or has NULL medians - build the serving "
            "lookups (dbt build -s serving_typical_rotation) before serving"
        )
    out["typical"] = {
        k: float(med[0][k]) for k in ("pos", "legs", "turn", "dist", "elapsed", "density")
    }
    log.info(
        "rotation hist loaded: %d bands, %d position keys; typical profile %s",
        len(out["band"]),
        len(out["pos"]),
        out["typical"],
    )
    return out


def _needs_density_estimate(v: float | None) -> bool:
    """A caller-supplied origin_dep_density_hour is usable only when finite and
    positive — training density is never NULL and is >= 1 by definition. Any
    other value (None, NaN, inf, <= 0) must fall back to the (origin, hour,
    weekday) estimate, never assemble NaN. Used at BOTH the key-collection and
    the lookup site so the two predicates can never drift (a value gated out at
    lookup but not collected as a key would silently become NaN)."""
    return not (v is not None and math.isfinite(v) and v > 0)


def _density_estimates(ctx: ServingContext, keys: list[tuple[str, int, int]]) -> dict:
    """Serve-time ESTIMATE of origin_dep_density_hour, from the preloaded
    lookup: the TRAINING-window median over distinct schedule-hours (not
    flight rows — a flight-row median would overweight busy banks) for
    (origin, hour, weekday).

    Fallback chain is unchanged from the query version it replaces: an unknown
    airport, or a known airport with no training rows at that hour/weekday,
    takes the global training median — always an in-distribution value, never
    NaN. The airports-index check is kept so an unknown airport resolves to the
    global median for the same reason it always did, rather than incidentally
    because the lookup missed.
    """
    default = ctx.rotation_hist.get("typical", {}).get("density", math.nan)
    return {
        k: (ctx.density.get(k, default) if k[0] in ctx.airports.index else default) for k in keys
    }


def _rotation_features(fl: FlightRequest, ctx: ServingContext, density: float) -> dict[str, float]:
    """The 15 cascade features for one request. With rotation context: the
    schedule values as provided, band/position keys derived exactly as the
    mart derives them (first-leg context -> the no_inbound band, matching
    training). Without context: the TYPICAL rotation profile — training
    medians for the schedule attributes, band/position keys derived from
    them — NULL would mislabel an unknown plan as swap-shaped (see
    _load_rotation_hist); the response flags the estimate."""
    has_context = fl.rotation_position is not None
    typ = ctx.rotation_hist.get("typical", {})
    if has_context:
        pos = float(fl.rotation_position)
        # legs_today >= rotation_position in every training row (COUNT(*) vs
        # ROW_NUMBER over the same tail/flight_date partition). The API enforces
        # this (422 on legs_today < rotation_position or on a missing
        # legs_today); serving FLOORS at the position on EVERY branch so a direct
        # dataclass caller (bypassing pydantic) can never assemble the
        # legs_today < rotation_position shape training never produces — one that
        # omits legs_today degrades to the typical median, one that supplies a
        # value below the position is clamped up to it.
        legs = (
            max(float(fl.legs_today), pos)
            if fl.legs_today is not None
            else max(typ.get("legs", pos), pos)
        )
        t = fl.sched_turnaround_min
        has_inbound = t is not None and math.isfinite(t) and 0 <= t <= 840
        # all-or-nothing inbound (the API enforces complete-or-absent context,
        # and requires the inbound triple for position >= 2; direct dataclass
        # constructors degrade to typical values): has_inbound rows in training
        # ALWAYS carry inbound distance/duration — never NaN them
        dist = fl.inbound_distance if fl.inbound_distance is not None else typ.get("dist")
        elapsed = (
            fl.inbound_crs_elapsed_min
            if fl.inbound_crs_elapsed_min is not None
            else typ.get("elapsed")
        )
    else:  # typical-profile estimate
        pos = typ.get("pos", math.nan)
        legs = typ.get("legs", math.nan)
        t = typ.get("turn", math.nan)
        has_inbound = math.isfinite(t)
        dist = typ.get("dist", math.nan)
        elapsed = typ.get("elapsed", math.nan)
    band = _turnaround_band(has_inbound, t if has_inbound else None)
    pos_key = str(min(int(pos), 6)) if math.isfinite(pos) else "3"

    def _fin(v: float | None) -> float:
        return float(v) if v is not None and math.isfinite(v) and v > 0 else math.nan

    row: dict[str, float] = {
        "rotation_position": pos,
        "legs_today": legs,
        "origin_dep_density_hour": density,
        "has_inbound_leg": 1.0 if has_inbound else 0.0,
        "sched_turnaround_min": float(t) if has_inbound else math.nan,
        "sched_turnaround_slack_min": float(t) - 35.0 if has_inbound else math.nan,
        "is_tight_turnaround": 1.0 if (has_inbound and t < 35) else 0.0,
        "inbound_distance": _fin(dist) if has_inbound else math.nan,
        "inbound_crs_elapsed_min": _fin(elapsed) if has_inbound else math.nan,
    }
    for kind, key, grain in (
        ("band", band, "turnaround_band"),
        ("pos", pos_key, "rotation_position"),
    ):
        entity = ctx.rotation_hist.get(kind, {}).get(key, {})
        row[f"hist_{grain}_arr_del15_rate"] = float(entity.get("rate") or math.nan)
        row[f"hist_{grain}_avg_arr_delay_minutes"] = float(entity.get("avg_min") or math.nan)
        row[f"hist_{grain}_n_flights"] = float(entity.get("n") or math.nan)
    return row


def _hist_lookup(ctx: ServingContext, grain: str, keys: list[str]) -> dict[str, dict]:
    """Constant-within-entity hist values, from the preloaded lookup.

    Same contract as the per-request query it replaces: absent entities simply
    do not appear, so the caller's .get(...) leaves NaN — the training NULL
    path for an entity first seen after the cutoff.
    """
    table = ctx.hist.get(grain, {})
    return {k: table[k] for k in keys if k in table}


def _route_distance(ctx: ServingContext, routes: list[str]) -> dict[str, float]:
    return {r: ctx.route_distance[r] for r in routes if r in ctx.route_distance}


@lru_cache(maxsize=512)
def _holiday_flags(d: date) -> Mapping[str, float]:
    """Holiday flags for one date, using the same library the training calendar
    was generated with.

    Cached: a whole-airport batch is one or two distinct dates across hundreds
    of flights, and constructing a 3-year holidays calendar per flight was pure
    waste.

    Returns a READ-ONLY mapping, not a dict. Every caller for a given date gets
    the same object, so a caller that mutated it would silently corrupt the
    feature for every subsequent request in the process — across all FastAPI
    threadpool workers, with no traceback. MappingProxyType makes that a
    TypeError instead of a promise in a docstring. `dict.update(proxy)` works,
    which is how the one caller consumes it.
    """
    import holidays  # same library the training calendar was generated with

    us = holidays.country_holidays("US", years=range(d.year - 1, d.year + 2))
    return MappingProxyType(
        {
            "is_holiday": float(d in us),
            "is_day_before_holiday": float(d + timedelta(days=1) in us),
            "is_day_after_holiday": float(d - timedelta(days=1) in us),
        }
    )


# NDFD grids refresh roughly hourly: cache each airport's fetched GRID for 30
# minutes (bounded by the airport count — never unbounded growth), and pin a
# FAILED fetch for only 5 minutes so a transient error or beyond-horizon
# request can never permanently lock a flight onto the NULL path. Per-hour
# feature extraction is pure and cheap, so it is not cached at all.
GRID_TTL_S = 1800
GRID_FAIL_TTL_S = 300


def _grid_for(ctx: ServingContext, origin: str) -> dict | None:
    now = time.monotonic()
    hit = ctx.forecast_cache.get(origin)
    if hit is not None:
        fetched_at, props = hit
        if now - fetched_at < (GRID_TTL_S if props is not None else GRID_FAIL_TTL_S):
            return props
    a = ctx.airports.loc[origin]
    props = fetch_grid(float(a["latitude"]), float(a["longitude"]))
    ctx.forecast_cache[origin] = (now, props)
    return props


def _departure_utc(
    ctx: ServingContext, origin: str, d: date, dep_time: str, *, hour_only: bool = False
) -> datetime | None:
    """The scheduled departure instant in UTC, or None if we cannot place it.

    Local wall clock -> UTC via the airport's IANA tz, exactly as the mart's
    weather join does with observations. Returns None for an unknown airport or
    a dim_airport row without a timezone — both take the NULL weather path, and
    for both we cannot say whether the flight is in the past.

    TWO CALLERS, TWO PRECISIONS, and the difference matters:

      hour_only=True  — the WEATHER bucket. Training joins the last hourly
                        observation at or before the scheduled departure HOUR,
                        so the grid lookup keeps using the truncated hour.
      hour_only=False — the PAST/FUTURE decision. Truncating here is a bug: at
                        17:05 a flight scheduled 17:30 becomes 17:00, comes back
                        flight_in_past=true, and a consumer UI told to hard-gate
                        on that value hides a perfectly valid pre-departure
                        prediction for up to 59 minutes before every departure.
    """
    if origin not in ctx.airports.index:
        return None
    tz = ctx.airports.loc[origin]["tz"]
    if not isinstance(tz, str) or not tz:
        log.warning("%s has no timezone in dim_airport; NULL weather path", origin)
        return None
    hh, mm = dep_time.split(":")
    minute = 0 if hour_only else int(mm)
    return datetime(d.year, d.month, d.day, int(hh), minute, tzinfo=ZoneInfo(tz)).astimezone(UTC)


def _origin_weather(ctx: ServingContext, origin: str, d: date, dep_time: str) -> dict[str, float]:
    """Forecast at the SCHEDULED departure hour — the training time reference."""
    # hour_only: the grid lookup keeps the training time reference (the
    # scheduled departure HOUR). The past check below deliberately uses the
    # full-precision instant instead — see _departure_utc.
    dep_utc = _departure_utc(ctx, origin, d, dep_time, hour_only=True)
    exact = _departure_utc(ctx, origin, d, dep_time)
    if dep_utc is None or exact is None:
        return dict(NULL_PATH)  # unknown airport / no tz: the training NULL path
    if exact <= datetime.now(UTC):
        # scoring an already-departed flight is a legitimate debugging use,
        # but the CURRENT grid may postdate departure — the output is not
        # "pre-departure knowable" and must not be presented as such. The
        # response now says so in prediction_basis.flight_in_past, rather than
        # only in a server log the caller never sees.
        log.warning(
            "%s %s %s already departed: forecast grid may postdate departure; "
            "not a pre-departure prediction",
            origin,
            d,
            dep_time,
        )
    return features_at_hour(_grid_for(ctx, origin), dep_utc)


def _prediction_basis(ctx: ServingContext, fl: FlightRequest, has_weather: bool) -> dict:
    """What this prediction actually rests on — surfaced, not buried in a log.

    `weather_horizon` distinguishes the three ways the twelve weather features
    can be absent, which a caller otherwise cannot tell apart from a bare
    has_origin_weather=false:

      forecast        an NDFD forecast for the scheduled hour was used
      beyond_horizon  future flight, but no forecast value — past the ~7-day
                      NDFD horizon, off-grid, or the fetch failed
      past            scheduled departure is at or before now, so this is NOT a
                      pre-departure prediction and a UI must refuse to present
                      it as one
      unavailable     the airport is unknown or has no timezone, so the instant
                      cannot even be placed

    `flight_in_past` is the one a consumer UI must hard-gate on. The API still
    scores such a request (debugging is a legitimate use) but the caller has to
    be told, because the number looks exactly like a real forecast.
    """
    dep_utc = _departure_utc(ctx, fl.origin, fl.flight_date, fl.dep_time)
    in_past = dep_utc is not None and dep_utc <= datetime.now(UTC)
    if dep_utc is None:
        horizon = "unavailable"
    elif in_past:
        horizon = "past"
    elif has_weather:
        horizon = "forecast"
    else:
        horizon = "beyond_horizon"
    return {
        "flight_in_past": in_past,
        "weather_horizon": horizon,
        # repeated from the top level so the block is a complete, self-contained
        # answer to "what does this rest on?" that a UI can render wholesale;
        # the top-level fields stay for the existing consumers
        "rotation_context": (
            "provided" if fl.rotation_position is not None else "typical_estimate"
        ),
        "origin_density_source": (
            "estimated" if _needs_density_estimate(fl.origin_dep_density_hour) else "provided"
        ),
    }


def assemble_features(ctx: ServingContext, flights: list[FlightRequest]) -> pd.DataFrame:
    routes = sorted({fl.origin + "-" + fl.dest for fl in flights})
    hist = {
        g: _hist_lookup(ctx, g, sorted({getattr(fl, a) for fl in flights}))
        for g, a in (("carrier", "carrier"), ("origin", "origin"), ("dest", "dest"))
    }
    hist["route"] = _hist_lookup(ctx, "route", routes)
    distances = _route_distance(ctx, routes)
    density_keys = sorted(
        {
            (fl.origin, int(fl.dep_time.split(":")[0]), fl.flight_date.isoweekday())
            for fl in flights
            if _needs_density_estimate(fl.origin_dep_density_hour)
        }
    )
    densities = _density_estimates(ctx, density_keys) if density_keys else {}

    rows = []
    for fl in flights:
        route = f"{fl.origin}-{fl.dest}"
        row: dict[str, object] = {
            "carrier": fl.carrier,
            "origin": fl.origin,
            "dest": fl.dest,
            "route": route,
            # caller override kept ONLY for finite-positive values (it is the
            # sole distance source for routes absent from the mart); anything
            # else falls back to the mart lookup — an invalid vector can never
            # assemble regardless of entry path (the API additionally 422s)
            "distance": (
                fl.distance
                if fl.distance is not None and math.isfinite(fl.distance) and fl.distance > 0
                else distances.get(route, math.nan)
            ),
            "crs_dep_hour": float(int(fl.dep_time.split(":")[0])),
            "crs_arr_hour": float(int(fl.arr_time.split(":")[0])),
            "day_of_week": float(fl.flight_date.isoweekday()),  # BTS: 1 = Monday
            "month": float(fl.flight_date.month),
        }
        for grain, attr in (
            ("route", None),
            ("carrier", "carrier"),
            ("origin", "origin"),
            ("dest", "dest"),
        ):
            key = route if grain == "route" else getattr(fl, attr)
            entity = hist[grain].get(key, {})
            for s in ("arr_del15_rate", "avg_arr_delay_minutes", "n_flights"):
                row[f"hist_{grain}_{s}"] = float(
                    entity.get(f"hist_{grain}_{s}", math.nan) or math.nan
                )
        row.update(_origin_weather(ctx, fl.origin, fl.flight_date, fl.dep_time))
        row.update(_holiday_flags(fl.flight_date))
        density = (
            densities.get(
                (fl.origin, int(fl.dep_time.split(":")[0]), fl.flight_date.isoweekday()),
                math.nan,
            )
            if _needs_density_estimate(fl.origin_dep_density_hour)
            else float(fl.origin_dep_density_hour)
        )
        row.update(_rotation_features(fl, ctx, density))
        rows.append(row)

    # gate BEFORE frame construction: pd.DataFrame(columns=...) would silently
    # create an all-NaN column for any feature the row dicts failed to
    # populate — a registry addition unmatched in serving must raise, never
    # score as a serving-only all-missing pattern
    unpopulated = [c for c in f.FEATURES if c not in rows[0]]
    if unpopulated:
        raise SchemaMismatchError(f"assembly did not populate features: {unpopulated}")
    return coerce_feature_frame(ctx, pd.DataFrame(rows, columns=list(f.FEATURES)))


def coerce_feature_frame(ctx: ServingContext, x: pd.DataFrame) -> pd.DataFrame:
    """Coerce an assembled feature frame to the trained dtypes and gate it.

    Split out of assemble_features so that any path scoring the shipped models
    — request assembly here, mart-row replay in ml/replay.py — goes through
    the SAME categorical vocabulary and the SAME schema gates. Behaviour is
    unchanged; this is the extraction, not a new policy.
    """
    for c in f.CATEGORICAL_FEATURES:
        # categorical dtype built on the TRAINING vocabulary: xgboost >= 3
        # recodes by name and raises on categories absent from the trained
        # encoder, so unseen values must become MISSING (default-direction
        # routing — the graceful analog of training's new-entity treatment,
        # whose hist_* values are already NaN via the lookup misses). Using
        # the fixed vocab also keeps the category set stable and non-empty
        # (an inferred all-missing column crashes xgboost on size-0 cats).
        vocab = ctx.category_vocab.get(c)
        if vocab:
            unseen = ~x[c].isin(vocab)
            if unseen.any():
                log.warning(
                    "%s: %d unseen value(s) -> missing category (e.g. %s)",
                    c,
                    int(unseen.sum()),
                    x.loc[unseen, c].iloc[0],
                )
            # pre-sorted at startup (ctx.category_order); sorted(vocab) here
            # would re-sort the 7,539-route list on every request
            x[c] = pd.Categorical(x[c], categories=ctx.category_order[c])
        else:
            x[c] = x[c].astype("category")
    for c in f.NUMERIC_FEATURES:
        x[c] = pd.to_numeric(x[c]).astype("float32")

    # HARD GATE: the assembled frame must match the model's stored schema
    if list(x.columns) != list(f.FEATURES):
        raise SchemaMismatchError(f"assembled columns {list(x.columns)} != FEATURES")
    stored = ctx.models.clf.get_booster().feature_names
    if list(x.columns) != stored:
        raise SchemaMismatchError(f"assembled columns != booster schema {stored}")
    return x


def predict(
    ctx: ServingContext, flights: list[FlightRequest], include_features: bool = True
) -> list[dict]:
    """Score a batch. Model calls are vectorised across the whole batch.

    include_features=False drops the per-flight 51-key `features` block. Bulk
    callers (a whole airport-day) do not want 51 floats per flight in the
    payload, and building it was the dominant per-flight cost — see
    _feature_records.
    """
    if not flights:
        return []
    x = assemble_features(ctx, flights)
    # RAW classifier scores are recall-inflated by scale_pos_weight; the
    # calibrator (Platt, strictly monotonic) remaps them onto the frequency
    # scale WITHOUT changing their order — so delay_probability is a calibrated
    # probability while ranking-based metrics are unchanged. TreeSHAP/margin
    # attribution explains the RAW margin upstream of this map, not p_cal.
    p_xgb = ctx.models.clf.predict_proba(x)[:, 1]
    p_cal = ctx.models.calibrator.transform(p_xgb)
    minutes = ctx.models.reg.predict(x)
    p_logreg = ctx.models.logreg.predict_proba(x[LOGREG_INPUT_COLUMNS])[:, 1]
    # vectorised once, not N x 51 scalar .iloc lookups inside the loop
    feature_records = _feature_records(x) if include_features else None
    has_weather = (x["has_origin_weather"].to_numpy() == 1.0).tolist()
    out = []
    for i, fl in enumerate(flights):
        out.append(
            {
                "flight": f"{fl.carrier} {fl.origin}->{fl.dest} {fl.flight_date} {fl.dep_time}",
                # CALIBRATED probability (Platt) — a delay FREQUENCY, not the
                # raw recall-inflated score; calibration method reported for
                # transparency. The logreg baseline stays raw (a comparison
                # anchor, itself class_weight-balanced and uncalibrated).
                "delay_probability": round(float(p_cal[i]), 4),
                "probability_calibration": ctx.models.calibrator.method,
                "expected_delay_minutes": round(float(minutes[i]), 1),
                "logreg_baseline_probability": round(float(p_logreg[i]), 4),
                "has_origin_weather": has_weather[i],
                # what this prediction rests on: whether it is pre-departure at
                # all, and which of the three ways weather can be absent applies
                "prediction_basis": _prediction_basis(ctx, fl, has_weather[i]),
                # whether the rotation LINKAGE (position/legs/inbound) was
                # caller-provided or the typical-median estimate. The API
                # enforces complete-or-absent, so "provided" means the whole
                # linkage came from the caller. Density is reported separately
                # (origin_density_source) because it is an independently
                # optional feature, estimated when omitted in BOTH modes.
                "rotation_context": (
                    "provided" if fl.rotation_position is not None else "typical_estimate"
                ),
                "origin_density_source": (
                    "estimated"
                    if _needs_density_estimate(fl.origin_dep_density_hour)
                    else "provided"
                ),
            }
        )
        if feature_records is not None:
            out[-1]["features"] = feature_records[i]
    return out


def _feature_records(x: pd.DataFrame) -> list[dict]:
    """The per-flight `features` block for a whole frame, built column-wise.

    Same output as the previous per-row comprehension, which did one pandas
    scalar .iloc lookup per feature per flight (51 x N — 76,500 of them for the
    benchmark's 1,500-flight batch, and the dominant per-flight cost in the
    response path). Here each column is converted once with a vectorised call
    and the rows are zipped together. Bulk callers should pass
    include_features=False and skip this entirely.

    NaN -> None is preserved exactly: numeric columns hold np.float32, which is
    NOT a Python float, so the isinstance check this logic originally used
    never fired and NaN leaked into the JSON on the NULL path. isna() is the
    check that actually works, applied per column.
    """
    columns: dict[str, list] = {}
    for k in f.FEATURES:
        col = x[k]
        na = col.isna().to_numpy()
        if k in f.CATEGORICAL_FEATURES:
            vals = [None if n else str(v) for v, n in zip(col.to_numpy(), na, strict=True)]
        else:
            vals = [None if n else float(v) for v, n in zip(col.to_numpy(), na, strict=True)]
        columns[k] = vals
    keys = list(f.FEATURES)
    return [dict(zip(keys, row, strict=True)) for row in zip(*columns.values(), strict=True)]
