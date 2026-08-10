"""FastAPI inference endpoint over the trained artifacts (the ML stretch goal).

Leak-free by construction: scores FUTURE flights with the weather FORECAST
for the flight date — pre-departure information in exactly the sense the
training boundary (CLAUDE.md §9) requires. See ml/serving.py for the full
leakage reasoning and the honest train/serve mismatch statement.

Run locally:
    uv run --extra ml --extra serve --extra ingestion uvicorn ml.api:app --port 8000
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from google.cloud import bigquery
from pydantic import BaseModel, Field, field_validator, model_validator

from ml.serving import FlightRequest, ServingContext, build_context, predict

log = logging.getLogger("ml.api")

_ctx: ServingContext | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _ctx
    _ctx = build_context()
    yield
    _ctx = None  # a torn-down app must not mask the next startup's failures


app = FastAPI(title="flight-delay inference", lifespan=lifespan)


class FlightIn(BaseModel):
    origin: str = Field(pattern=r"^[A-Za-z0-9]{3}$", examples=["ORD"])
    dest: str = Field(pattern=r"^[A-Za-z0-9]{3}$", examples=["ATL"])
    carrier: str = Field(pattern=r"^[A-Za-z0-9]{2,3}$", examples=["AA"])
    flight_date: date
    # valid clock times only — 99:99 must 422 here, not 500 in assembly
    dep_time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$", examples=["17:30"])
    arr_time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$", examples=["20:45"])
    # finite-positive only: zero/negative/NaN/inf must 422, never assemble a
    # physically invalid vector; bound comfortably above the longest US
    # domestic leg (~5,000 mi)
    distance: float | None = Field(default=None, gt=0, lt=20000)
    # OPTIONAL aircraft-rotation context, all SCHEDULE-derived (see
    # ml/serving.py): provide it if you know the planned rotation; omitting
    # it yields the TYPICAL-profile estimate, flagged in the response. It is
    # complete-or-absent: once rotation_position is given, legs_today is
    # required and (for position >= 2) the inbound leg is too — a partial
    # context 422s rather than assemble a shape training never produced
    # (see _rotation_coherence).
    rotation_position: int | None = Field(default=None, ge=1, le=30)
    legs_today: int | None = Field(default=None, ge=1, le=30)
    sched_turnaround_min: float | None = Field(default=None, ge=0, le=840)
    inbound_distance: float | None = Field(default=None, gt=0, lt=20000)
    inbound_crs_elapsed_min: float | None = Field(default=None, gt=0, le=900)
    origin_dep_density_hour: float | None = Field(default=None, gt=0, le=500)

    @field_validator("origin", "dest", "carrier")
    @classmethod
    def _uppercase(cls, v: str) -> str:
        # BTS codes are uppercase; lowercase input would silently miss the
        # airport/hist lookups and score as unseen categories
        return v.upper()

    @field_validator("sched_turnaround_min", "inbound_crs_elapsed_min", "origin_dep_density_hour")
    @classmethod
    def _whole_number(cls, v: float | None) -> float | None:
        # these are integer quantities in training — minute-granularity
        # TIMESTAMP_DIFFs (turnaround, inbound elapsed) and a COUNT(*) (density)
        # in int_aircraft_rotation.sql. A fractional input (35.5, a count of
        # 7.5) assembles an off-grid value training never produced, so reject
        # sub-integer floats; an integer-valued float (45.0) passes. (Distances
        # are also whole BTS miles but stay float, matching the pre-existing
        # `distance` field's convention — impact is equally negligible and out
        # of this endpoint's rotation scope.)
        if v is not None and v % 1 != 0:
            raise ValueError("must be a whole number (integer minutes / count)")
        return v

    @model_validator(mode="after")
    def _rotation_coherence(self) -> FlightIn:
        # Rotation context is COMPLETE-and-coherent OR wholly absent — the same
        # binary the serving layer documents (ml/serving.py): a caller supplies
        # the full rotation shape training co-populates, or supplies none of it
        # and takes the flagged typical-profile estimate. A PARTIAL context 422s
        # here rather than silently assembling a vector training never produced
        # (and being mislabeled rotation_context="provided" in the response).
        inbound = (self.sched_turnaround_min, self.inbound_distance, self.inbound_crs_elapsed_min)
        if self.rotation_position is None:
            if any(v is not None for v in (*inbound, self.legs_today)):
                raise ValueError("rotation fields require rotation_position")
            return self
        # legs_today and rotation_position are co-populated in every training
        # row (COUNT(*) vs ROW_NUMBER over the same tail/flight_date partition),
        # so legs_today is required and can never be below the position.
        if self.legs_today is None:
            raise ValueError("legs_today is required when rotation_position is provided")
        if self.legs_today < self.rotation_position:
            raise ValueError("legs_today cannot be less than rotation_position")
        # an inbound is all-or-nothing: training rows with has_inbound_leg
        # always carry the inbound's distance and duration together — a partial
        # inbound would assemble a pattern training never produced
        if any(v is not None for v in inbound) and not all(v is not None for v in inbound):
            raise ValueError(
                "sched_turnaround_min, inbound_distance and inbound_crs_elapsed_min "
                "must be provided together"
            )
        # Only a FIRST leg (rotation_position == 1) may omit the inbound: that is
        # the clean first-of-service-date shape (class-b) training produces. A
        # position >= 2 leg has an inbound within the service day by
        # construction; omitting it would assemble the no_inbound band at a
        # mid-rotation position — a shape with essentially zero training support
        # that also discards the strongest cascade signal (the turnaround band).
        # A genuine >14h same-day gap can't even be expressed here anyway
        # (sched_turnaround_min is capped at 840). rotation_position == 1 keeps
        # the inbound OPTIONAL: a clean first leg omits it; an
        # overnight-turnaround first leg supplies the full triple.
        if self.rotation_position >= 2 and not all(v is not None for v in inbound):
            raise ValueError(
                "rotation_position >= 2 requires the inbound leg (sched_turnaround_min, "
                "inbound_distance, inbound_crs_elapsed_min); omit rotation_position "
                "entirely for the typical-profile estimate"
            )
        return self


class BatchIn(BaseModel):
    flights: list[FlightIn] = Field(min_length=1)


def _ctx_or_503() -> ServingContext:
    if _ctx is None:
        raise HTTPException(503, "serving context not initialized")
    return _ctx


@lru_cache(maxsize=8)
def _artifact_json(artifacts_dir: str, name: str) -> dict | None:
    """Read a side-metadata file out of the artifacts run, or None.

    These are REPORTS, not part of the model contract: load_models requires
    exactly the four model files, and a run written before this metadata
    existed must still serve predictions. So a missing file degrades to null
    rather than failing startup. Cached — the files never change for a run.
    """
    path = Path(artifacts_dir) / name
    if not path.is_file():
        log.warning("%s not found in %s — endpoint will return null", name, artifacts_dir)
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning(
            "%s in %s is unreadable (%s) — endpoint will return null", name, artifacts_dir, exc
        )
        return None


@app.get("/calibration")
def calibration() -> dict:
    """Held-out reliability: when the model says p, how often is it delayed?

    Straight out of the shipped run's metrics.json — 10 equal-width bands over
    the full 3,561,782-row held-out test set, computed at training time by
    ml/calibration.py. A consumer UI renders this as the evidence behind its
    probability; it is not recomputed per request and never touches the mart.
    """
    ctx = _ctx_or_503()
    m = _artifact_json(str(ctx.models.artifacts_dir), "metrics.json")
    if m is None:
        return {"available": False, "reason": "metrics.json absent from the artifacts run"}
    platt = m.get("calibration", {}).get("test", {}).get("platt", {})
    split = m.get("split", {})
    return {
        "available": True,
        "artifacts": ctx.models.artifacts_dir.name,
        "calibration_method": m.get("calibration", {}).get("serving_method"),
        "n_test": split.get("n_test"),
        # metrics.json records the split BOUNDARY, not the mart's end date, so
        # only the start is reported here rather than inventing a window end.
        # /outcome-mix carries the observed [min, max] of the same rows.
        "test_start": split.get("test_date_min"),
        "train_end": split.get("train_date_max"),
        "base_rate": m.get("baselines", {}).get("test_delay_base_rate"),
        "roc_auc": m.get("xgb_classifier", {}).get("roc_auc"),
        "pr_auc": m.get("xgb_classifier", {}).get("pr_auc"),
        "brier": platt.get("brier"),
        "ece": platt.get("ece"),
        # [{lo, hi, n, mean_pred, frac_pos}] — predicted vs actual per band
        "reliability": platt.get("reliability"),
    }


@app.get("/outcome-mix")
def outcome_mix() -> dict:
    """Held-out P(arrival delay >= t) per probability band.

    The honest answer to "how late might I be?" — empirical, from flights the
    model never trained on. A consumer UI shows this INSTEAD of the regressor's
    point estimate, whose held-out MAE (18.99 min) and RMSE (49.26) make a
    single number misleading. Bands match /calibration exactly, so a flight
    sits in the same band on both panels.

    Built by ml/exceedance.py; see its header.
    """
    ctx = _ctx_or_503()
    table = _artifact_json(str(ctx.models.artifacts_dir), "exceedance.json")
    if table is None:
        return {
            "available": False,
            "reason": "exceedance.json absent — run `python -m ml.exceedance` for this run",
        }
    return {"available": True, **table}


@app.get("/health")
def health() -> dict:
    ctx = _ctx_or_503()
    return {"status": "ok", "artifacts": ctx.models.artifacts_dir.name}


@app.post("/predict")
def predict_one(flight: FlightIn) -> dict:
    ctx = _ctx_or_503()
    return predict(ctx, [FlightRequest(**flight.model_dump())])[0]


@app.post("/predict/batch")
def predict_batch(batch: BatchIn) -> list[dict]:
    ctx = _ctx_or_503()
    return predict(ctx, [FlightRequest(**fl.model_dump()) for fl in batch.flights])


@app.get("/demo/airport-departures")
@app.get("/demo/ord-departures")  # original path, kept so existing callers work
def demo_airport(origin: str = "ORD", target_date: date | None = None) -> list[dict]:
    """Batch demo: 'all of tomorrow's scheduled departures' from one airport.

    HONEST PROXY: we hold no source of FUTURE airline schedules (that is an
    OAG/airline-feed product). The demo re-dates a historical schedule from the
    same weekday to the target date — a realistic schedule shape, clearly
    labeled a proxy. A production deployment would swap in a real schedule
    feed; the prediction path is identical either way. The proxy includes its
    historical ROTATION structure (positions, turnarounds, density) — all
    schedule-derived, passed through as the rotation context a planning feed
    would provide.

    The proxy day is LOOKED UP, not computed from a fixed offset. The original
    version subtracted a literal 104 weeks, which silently stops landing inside
    the 2022-2024 mart once target dates pass ~2026-12-30 — every call would
    404. Selecting the most recent in-mart date with the same weekday AT THIS
    ORIGIN is self-healing, always has flights, and preserves the weekday shape
    that makes the schedule realistic.
    """
    ctx = _ctx_or_503()
    origin = origin.upper()
    if origin not in ctx.airports.index:
        raise HTTPException(404, f"unknown origin {origin!r}")
    # "tomorrow" in the ORIGIN's own timezone — a UTC deployment queried
    # overnight must not target the wrong calendar day
    tz = ctx.airports.loc[origin]["tz"]
    target = target_date or (
        datetime.now(ZoneInfo(tz if isinstance(tz, str) and tz else "UTC")).date()
        + timedelta(days=1)
    )
    params = [
        bigquery.ScalarQueryParameter("origin", "STRING", origin),
        bigquery.ScalarQueryParameter("target", "DATE", target),
    ]
    proxy_rows = list(
        ctx.bq.query(
            f"select max(flight_date) as proxy_day "
            f"from `{ctx.bq.project}.{ctx.gold}.ml_flight_features` "
            f"where origin = @origin "
            # same weekday as the target; both sides use BigQuery's own
            # convention so no 1=Monday/1=Sunday mismatch is possible
            f"and extract(dayofweek from flight_date) = extract(dayofweek from @target)",
            job_config=bigquery.QueryJobConfig(query_parameters=params),
        ).result()
    )
    proxy_day = proxy_rows[0]["proxy_day"] if proxy_rows else None
    if proxy_day is None:
        raise HTTPException(404, f"no in-mart schedule for {origin} on that weekday")
    rows = ctx.bq.query(
        f"select carrier, dest, format_time('%H:%M', crs_dep_time) as dep_time, "
        f"cast(crs_arr_hour as int64) as arr_hour, any_value(distance) as distance, "
        f"any_value(struct(rotation_position, legs_today, sched_turnaround_min, "
        f"inbound_distance, inbound_crs_elapsed_min, origin_dep_density_hour)) as rot "
        f"from `{ctx.bq.project}.{ctx.gold}.ml_flight_features` "
        f"where origin = @origin and flight_date = @d "
        f"group by carrier, dest, dep_time, arr_hour",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("origin", "STRING", origin),
                bigquery.ScalarQueryParameter("d", "DATE", proxy_day),
            ]
        ),
    ).result()

    def _opt(v, cast=float):
        return cast(v) if v is not None else None

    flights = [
        FlightRequest(
            origin=origin,
            dest=r["dest"],
            carrier=r["carrier"],
            flight_date=target,
            dep_time=r["dep_time"],
            arr_time=f"{int(r['arr_hour']):02d}:30",
            distance=float(r["distance"]),
            rotation_position=_opt(r["rot"]["rotation_position"], int),
            legs_today=_opt(r["rot"]["legs_today"], int),
            sched_turnaround_min=_opt(r["rot"]["sched_turnaround_min"]),
            inbound_distance=_opt(r["rot"]["inbound_distance"]),
            inbound_crs_elapsed_min=_opt(r["rot"]["inbound_crs_elapsed_min"]),
            origin_dep_density_hour=_opt(r["rot"]["origin_dep_density_hour"]),
        )
        for r in rows
    ]
    if not flights:
        raise HTTPException(404, f"no proxy schedule found for {proxy_day}")
    # include_features=False, not a post-hoc pop: building the 51-key block for
    # ~850 flights and then discarding it was the dominant per-flight cost in
    # the response path
    return predict(ctx, flights, include_features=False)
