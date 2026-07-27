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
from datetime import date, timedelta

from fastapi import FastAPI, HTTPException
from google.cloud import bigquery
from pydantic import BaseModel, Field

from ml.serving import FlightRequest, ServingContext, build_context, predict

_ctx: ServingContext | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _ctx
    _ctx = build_context()
    yield


app = FastAPI(title="flight-delay inference", lifespan=lifespan)


class FlightIn(BaseModel):
    origin: str = Field(examples=["ORD"])
    dest: str = Field(examples=["ATL"])
    carrier: str = Field(examples=["AA"])
    flight_date: date
    dep_time: str = Field(pattern=r"^\d{2}:\d{2}$", examples=["17:30"])
    arr_time: str = Field(pattern=r"^\d{2}:\d{2}$", examples=["20:45"])
    distance: float | None = None


class BatchIn(BaseModel):
    flights: list[FlightIn]


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
    target = target_date or (date.today() + timedelta(days=1))
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
