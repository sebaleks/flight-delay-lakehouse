"""FastAPI inference endpoint over the trained artifacts (the ML stretch goal).

Leak-free by construction: scores FUTURE flights with the weather FORECAST
for the flight date — pre-departure information in exactly the sense the
training boundary (CLAUDE.md §9) requires. See ml/serving.py for the full
leakage reasoning and the honest train/serve mismatch statement.

Run locally:
    uv run --extra ml --extra serve --extra ingestion uvicorn ml.api:app --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from google.cloud import bigquery
from pydantic import BaseModel, Field, field_validator

from ml.serving import FlightRequest, ServingContext, build_context, predict

_ctx: ServingContext | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _ctx
    _ctx = build_context()
    yield


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

    @field_validator("origin", "dest", "carrier")
    @classmethod
    def _uppercase(cls, v: str) -> str:
        # BTS codes are uppercase; lowercase input would silently miss the
        # airport/hist lookups and score as unseen categories
        return v.upper()


class BatchIn(BaseModel):
    flights: list[FlightIn] = Field(min_length=1)


def _ctx_or_503() -> ServingContext:
    if _ctx is None:
        raise HTTPException(503, "serving context not initialized")
    return _ctx


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


@app.get("/demo/ord-departures")
def demo_ord(target_date: date | None = None) -> list[dict]:
    """Batch demo: 'all of tomorrow's scheduled ORD departures'.

    HONEST PROXY: we hold no source of FUTURE airline schedules (that is an
    OAG/airline-feed product). The demo re-dates the historical ORD schedule
    from the same weekday 104 weeks earlier (inside the 2022-2024 mart
    window) to the target date — a realistic schedule shape, clearly labeled
    a proxy. A production deployment would swap in a real schedule feed; the
    prediction path is identical either way.
    """
    ctx = _ctx_or_503()
    # "tomorrow" in ORD's own timezone — a UTC deployment queried overnight
    # must not target the wrong calendar day
    target = target_date or (datetime.now(ZoneInfo("America/Chicago")).date() + timedelta(days=1))
    proxy_day = target - timedelta(weeks=104)  # same weekday, in-mart
    rows = ctx.bq.query(
        f"select carrier, dest, format_time('%H:%M', crs_dep_time) as dep_time, "
        f"cast(crs_arr_hour as int64) as arr_hour, any_value(distance) as distance "
        f"from `{ctx.bq.project}.{ctx.gold}.ml_flight_features` "
        f"where origin = 'ORD' and flight_date = @d "
        f"group by carrier, dest, dep_time, arr_hour",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("d", "DATE", proxy_day)]
        ),
    ).result()
    flights = [
        FlightRequest(
            origin="ORD",
            dest=r["dest"],
            carrier=r["carrier"],
            flight_date=target,
            dep_time=r["dep_time"],
            arr_time=f"{int(r['arr_hour']):02d}:30",
            distance=float(r["distance"]),
        )
        for r in rows
    ]
    if not flights:
        raise HTTPException(404, f"no proxy schedule found for {proxy_day}")
    results = predict(ctx, flights)
    for r in results:
        r.pop("features", None)  # keep the batch response compact
    return results
