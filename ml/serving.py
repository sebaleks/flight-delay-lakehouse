"""Forecast-driven inference over the trained artifacts: score FUTURE flights.

WHY THIS IS LEAK-FREE (CLAUDE.md §9): the pre-departure boundary requires
every predictor to be knowable before the flight departs. For a flight that
has not yet departed, a weather FORECAST issued now predates departure by
construction — it is legitimately pre-departure information. Training uses
the last OBSERVED hourly reading at or before the scheduled departure hour;
serving substitutes the FORECAST valid at that same hour. Every other
feature (schedule attributes, smoothed training-window historical rates,
holiday flags) is knowable arbitrarily far in advance.

TRAIN/SERVE MISMATCH (honest, now down to ONE gap): training and serving
reference the SAME instant — the scheduled departure hour — so the
prior-day-vs-flight-day time misalignment of the daily-weather era is GONE.
What remains is forecast-vs-observed error: training features are
observations, serving features are NDFD forecasts of those observations
(plus the documented QPF-apportionment approximation). That gap does not
change the held-out test metrics — those were computed entirely on observed
data and stand as reported — and it is the unavoidable price of scoring
flights that have not happened yet.

Feature parity with training:
  * hist_* rates are read from ml_flight_features itself — they are constant
    within an entity (verified property), so ANY_VALUE per entity reproduces
    the training values byte-exactly with zero formula duplication. Entities
    absent from the mart (new route etc.) stay NaN, the training NULL path.
  * holiday flags use the same `holidays` library + one-year padding as the
    seed generator.
  * day_of_week uses the BTS convention (isoweekday: 1 = Monday).
  * The assembled frame's columns are asserted equal to the model's stored
    feature schema before any prediction — a mismatch raises, never scores.
"""

from __future__ import annotations

import logging
import math
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
from ml.forecast import WEATHER_FEATURE_DEFAULTS, fetch_hourly_forecast
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


@dataclass
class Models:
    clf: xgb.XGBClassifier
    reg: xgb.XGBRegressor
    logreg: object
    artifacts_dir: Path


@dataclass
class ServingContext:
    models: Models
    bq: bigquery.Client
    gold: str
    airports: pd.DataFrame  # iata -> latitude, longitude, tz
    forecast_cache: dict = field(default_factory=dict)


def load_models(artifacts_dir: Path | None = None) -> Models:
    """Load the self-contained artifacts (no retraining, no side metadata) and
    assert their stored feature schemas match the canonical registry."""
    if artifacts_dir is None:
        runs = sorted(d for d in ARTIFACT_ROOT.iterdir() if d.is_dir())
        if not runs:
            raise FileNotFoundError(f"no artifact runs under {ARTIFACT_ROOT}")
        artifacts_dir = runs[-1]
    clf = xgb.XGBClassifier()
    clf.load_model(artifacts_dir / "xgb_classifier.ubj")
    reg = xgb.XGBRegressor()
    reg.load_model(artifacts_dir / "xgb_regressor.ubj")
    logreg = joblib.load(artifacts_dir / "logreg_pipeline.joblib")

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
    return Models(clf=clf, reg=reg, logreg=logreg, artifacts_dir=artifacts_dir)


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
    return ServingContext(models=load_models(artifacts_dir), bq=bq, gold=gold, airports=airports)


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


def _origin_weather(ctx: ServingContext, origin: str, d: date, dep_time: str) -> dict[str, float]:
    """Forecast at the SCHEDULED departure hour — the training time reference.
    Local wall clock -> UTC via the airport's IANA tz, exactly as the mart's
    join does with observations. Cache key includes the hour."""
    hour = int(dep_time.split(":")[0])
    key = (origin, d, hour)
    if key not in ctx.forecast_cache:
        if origin in ctx.airports.index:
            a = ctx.airports.loc[origin]
            dep_local = datetime(d.year, d.month, d.day, hour, tzinfo=ZoneInfo(str(a["tz"])))
            ctx.forecast_cache[key] = fetch_hourly_forecast(
                float(a["latitude"]), float(a["longitude"]), dep_local.astimezone(UTC)
            )
        else:  # unknown airport: the training NULL path
            ctx.forecast_cache[key] = {**WEATHER_FEATURE_DEFAULTS, "has_origin_weather": 0.0}
    return ctx.forecast_cache[key]


def assemble_features(ctx: ServingContext, flights: list[FlightRequest]) -> pd.DataFrame:
    routes = sorted({fl.origin + "-" + fl.dest for fl in flights})
    hist = {
        g: _hist_lookup(ctx, g, sorted({getattr(fl, a) for fl in flights}))
        for g, a in (("carrier", "carrier"), ("origin", "origin"), ("dest", "dest"))
    }
    hist["route"] = _hist_lookup(ctx, "route", routes)
    distances = _route_distance(ctx, routes)

    rows = []
    for fl in flights:
        route = f"{fl.origin}-{fl.dest}"
        row: dict[str, object] = {
            "carrier": fl.carrier,
            "origin": fl.origin,
            "dest": fl.dest,
            "route": route,
            "distance": fl.distance if fl.distance is not None else distances.get(route, math.nan),
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
        rows.append(row)

    x = pd.DataFrame(rows, columns=list(f.FEATURES))
    for c in f.CATEGORICAL_FEATURES:
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
    x = assemble_features(ctx, flights)
    p_xgb = ctx.models.clf.predict_proba(x)[:, 1]
    minutes = ctx.models.reg.predict(x)
    p_logreg = ctx.models.logreg.predict_proba(x[LOGREG_INPUT_COLUMNS])[:, 1]
    out = []
    for i, fl in enumerate(flights):
        out.append(
            {
                "flight": f"{fl.carrier} {fl.origin}->{fl.dest} {fl.flight_date} {fl.dep_time}",
                "delay_probability": round(float(p_xgb[i]), 4),
                "expected_delay_minutes": round(float(minutes[i]), 1),
                "logreg_baseline_probability": round(float(p_logreg[i]), 4),
                "has_origin_weather": bool(x["has_origin_weather"].iloc[i] == 1.0),
                "features": {
                    k: (
                        None
                        if isinstance(v := x[k].iloc[i], float) and math.isnan(v)
                        else (str(v) if k in f.CATEGORICAL_FEATURES else float(v))
                    )
                    for k in f.FEATURES
                },
            }
        )
    return out
