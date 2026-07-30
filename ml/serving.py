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
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import xgboost as xgb
from google.cloud import bigquery

from ingestion.config import require_env
from ml import features as f
from ml.forecast import NULL_PATH, features_at_hour, fetch_grid
from ml.train import ARTIFACT_ROOT, LOGREG_INPUT_COLUMNS

log = logging.getLogger("ml.serving")

MART = "ml_flight_features"
HIST_GRAINS = {"route": "route", "carrier": "carrier", "origin": "origin", "dest": "dest"}


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
    # (band -> 3 hist values), (position key -> 3 hist values): loaded once at
    # startup from the mart (constant within entity — byte-exact training
    # values), see _load_rotation_hist
    rotation_hist: dict = field(default_factory=dict)
    density_cache: dict = field(default_factory=dict)
    # training category vocabulary per categorical feature: unseen values
    # must become MISSING before prediction — xgboost >= 3 hard-errors on a
    # category absent from the trained encoder instead of routing it to the
    # default direction; see assemble_features
    category_vocab: dict = field(default_factory=dict)


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
    ctx.rotation_hist = _load_rotation_hist(ctx)
    vocab_rows = ctx.bq.query(
        f"select 'carrier' as col, carrier as v from `{bq.project}.{gold}.{MART}` group by v "
        f"union all select 'origin', origin from `{bq.project}.{gold}.{MART}` group by 2 "
        f"union all select 'dest', dest from `{bq.project}.{gold}.{MART}` group by 2 "
        f"union all select 'route', route from `{bq.project}.{gold}.{MART}` group by 2"
    ).result()
    ctx.category_vocab = {}
    for r in vocab_rows:
        ctx.category_vocab.setdefault(r["col"], set()).add(r["v"])
    log.info(
        "category vocab loaded: %s",
        {k: len(v) for k, v in ctx.category_vocab.items()},
    )
    return ctx


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


def _load_rotation_hist(ctx: ServingContext) -> dict:
    """The turnaround-band and rotation-position hist triples, read once at
    startup FROM THE MART (constant within entity — byte-exact training
    values, zero rates-formula duplication; the band is reconstructed from
    mart columns exactly as the standing dbt guard reconstructs it), PLUS
    the training MEDIANS of the rotation schedule attributes — the 'typical
    rotation profile' used when a caller provides no context. Why medians
    and not NaN: the mart has essentially no tail-unknown rows (completed
    flights carry tails), so NaN in these columns sits OUTSIDE the training
    distribution and empirically produces garbage scores; unknown-but-
    knowable schedule facts are instead estimated with training medians —
    the same epistemic move as the density estimator — and the response
    flags the estimate."""
    band_expr = """
        case
            when not has_inbound_leg then 'no_inbound'
            when sched_turnaround_min < 35 then 'lt_35'
            when sched_turnaround_min < 60 then '35_60'
            when sched_turnaround_min < 120 then '60_120'
            else 'ge_120'
        end"""
    pos_expr = "cast(least(rotation_position, 6) as string)"
    out: dict = {"band": {}, "pos": {}}
    for kind, expr, grain in (
        ("band", band_expr, "turnaround_band"),
        ("pos", pos_expr, "rotation_position"),
    ):
        rows = ctx.bq.query(
            f"select {expr} as k, "
            f"any_value(hist_{grain}_arr_del15_rate) as rate, "
            f"any_value(hist_{grain}_avg_arr_delay_minutes) as avg_min, "
            f"any_value(hist_{grain}_n_flights) as n "
            f"from `{ctx.bq.project}.{ctx.gold}.{MART}` "
            f"where rotation_position is not null group by k"
        ).result()
        out[kind] = {r["k"]: dict(r) for r in rows}
    med = list(
        ctx.bq.query(
            f"select approx_quantiles(rotation_position, 2)[offset(1)] as pos, "
            f"approx_quantiles(legs_today, 2)[offset(1)] as legs, "
            f"approx_quantiles(sched_turnaround_min, 2)[offset(1)] as turn, "
            f"approx_quantiles(inbound_distance, 2)[offset(1)] as dist, "
            f"approx_quantiles(inbound_crs_elapsed_min, 2)[offset(1)] as elapsed, "
            # last-resort density (misses + unknown airports): TRAINING median
            # over distinct schedule-hours, not flight rows
            f"(select approx_quantiles(d, 2)[offset(1)] from (select distinct origin, "
            f"flight_date, crs_dep_hour, origin_dep_density_hour as d "
            f"from `{ctx.bq.project}.{ctx.gold}.{MART}` where is_training_row)) as density "
            # TRAINING-window medians: the fallback must sit inside the
            # distribution the models were fit on, not a full-mart blend
            f"from `{ctx.bq.project}.{ctx.gold}.{MART}` "
            f"where has_inbound_leg and is_training_row"
        ).result()
    )[0]
    if any(med[k] is None for k in ("pos", "legs", "turn", "dist", "elapsed", "density")):
        raise RuntimeError(
            "ml_flight_features is empty or missing rotation columns - "
            "build the mart (dbt build -s ml_flight_features) before serving"
        )
    out["typical"] = {
        k: float(med[k]) for k in ("pos", "legs", "turn", "dist", "elapsed", "density")
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
    """Serve-time ESTIMATE of origin_dep_density_hour: the TRAINING-window
    median over distinct schedule-hours (not flight rows — a flight-row
    median would overweight busy banks) for (origin, hour, weekday). Only
    KNOWN airports are queried and cached (bounded: airports x 24 x 7);
    unknown airports and empty groups fall back to the global training
    median — always an in-distribution value, never NaN. Parameterized —
    no request-derived string ever enters the SQL text."""
    default = ctx.rotation_hist.get("typical", {}).get("density", math.nan)
    known = [k for k in keys if k[0] in ctx.airports.index]
    missing = [k for k in known if k not in ctx.density_cache]
    if missing:
        os_, hs, ds = (list(v) for v in zip(*missing, strict=True))
        rows = ctx.bq.query(
            f"select origin, h, d, approx_quantiles(density, 2)[offset(1)] as med "
            f"from (select distinct origin, cast(crs_dep_hour as int64) h, "
            f"cast(day_of_week as int64) d, flight_date, origin_dep_density_hour as density "
            f"from `{ctx.bq.project}.{ctx.gold}.{MART}` where is_training_row) "
            f"where (origin, h, d) in (select (o[offset(i)], hh[offset(i)], dd[offset(i)]) "
            f"from unnest([struct(@origins as o, @hours as hh, @dows as dd)]), "
            f"unnest(generate_array(0, array_length(@origins) - 1)) as i) "
            f"group by origin, h, d",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("origins", "STRING", os_),
                    bigquery.ArrayQueryParameter("hours", "INT64", hs),
                    bigquery.ArrayQueryParameter("dows", "INT64", ds),
                ]
            ),
        ).result()
        found = {(r["origin"], r["h"], r["d"]): float(r["med"]) for r in rows}
        for k in missing:
            ctx.density_cache[k] = found.get(k, default)
    return {k: ctx.density_cache.get(k, default) for k in keys}


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
    """Constant-within-entity hist values straight from the mart (byte-exact
    parity with training); absent entities simply do not appear (NaN path)."""
    if not keys:
        return {}
    cols = ", ".join(
        f"any_value(hist_{grain}_{s}) as hist_{grain}_{s}"
        for s in ("arr_del15_rate", "avg_arr_delay_minutes", "n_flights")
    )
    rows = ctx.bq.query(
        f"select {HIST_GRAINS[grain]} as k, {cols} "
        f"from `{ctx.bq.project}.{ctx.gold}.{MART}` "
        f"where {HIST_GRAINS[grain]} in unnest(@keys) group by k",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("keys", "STRING", keys)]
        ),
    ).result()
    return {r["k"]: dict(r) for r in rows}


def _route_distance(ctx: ServingContext, routes: list[str]) -> dict[str, float]:
    if not routes:
        return {}
    rows = ctx.bq.query(
        f"select route, any_value(distance) as distance "
        f"from `{ctx.bq.project}.{ctx.gold}.{MART}` where route in unnest(@r) group by route",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("r", "STRING", routes)]
        ),
    ).result()
    return {r["route"]: float(r["distance"]) for r in rows}


def _holiday_flags(d: date) -> dict[str, float]:
    import holidays  # same library the training calendar was generated with

    us = holidays.country_holidays("US", years=range(d.year - 1, d.year + 2))
    return {
        "is_holiday": float(d in us),
        "is_day_before_holiday": float(d + timedelta(days=1) in us),
        "is_day_after_holiday": float(d - timedelta(days=1) in us),
    }


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


def _origin_weather(ctx: ServingContext, origin: str, d: date, dep_time: str) -> dict[str, float]:
    """Forecast at the SCHEDULED departure hour — the training time reference.
    Local wall clock -> UTC via the airport's IANA tz, exactly as the mart's
    join does with observations."""
    if origin not in ctx.airports.index:
        return dict(NULL_PATH)  # unknown airport: the training NULL path
    a = ctx.airports.loc[origin]
    tz = a["tz"]
    if not isinstance(tz, str) or not tz:
        # a dim_airport row without a timezone cannot place the departure
        # hour — NULL weather path, never a 500
        log.warning("%s has no timezone in dim_airport; NULL weather path", origin)
        return dict(NULL_PATH)
    hour = int(dep_time.split(":")[0])
    dep_local = datetime(d.year, d.month, d.day, hour, tzinfo=ZoneInfo(tz))
    dep_utc = dep_local.astimezone(UTC)
    if dep_utc <= datetime.now(UTC):
        # scoring an already-departed flight is a legitimate debugging use,
        # but the CURRENT grid may postdate departure — the output is not
        # "pre-departure knowable" and must not be presented as such
        log.warning(
            "%s %s %s already departed: forecast grid may postdate departure; "
            "not a pre-departure prediction",
            origin,
            d,
            dep_time,
        )
    return features_at_hour(_grid_for(ctx, origin), dep_utc)


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
    x = pd.DataFrame(rows, columns=list(f.FEATURES))
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
            x[c] = pd.Categorical(x[c], categories=sorted(vocab))
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


def predict(ctx: ServingContext, flights: list[FlightRequest]) -> list[dict]:
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
                "has_origin_weather": bool(x["has_origin_weather"].iloc[i] == 1.0),
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
                # pd.isna, not isinstance(float): numeric columns hold
                # np.float32, which is NOT a Python float — the old check
                # never fired and NaN leaked into the JSON on the NULL path
                "features": {
                    k: (
                        None
                        if pd.isna(v := x[k].iloc[i])
                        else (str(v) if k in f.CATEGORICAL_FEATURES else float(v))
                    )
                    for k in f.FEATURES
                },
            }
        )
    return out
