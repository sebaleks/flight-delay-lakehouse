{{
    config(
        materialized='table',
        partition_by={'field': 'date_key', 'data_type': 'date', 'granularity': 'month'},
        cluster_by=['origin_airport_key', 'dest_airport_key', 'carrier_key'],
    )
}}

-- Fact table: one row per flight leg (cancelled and diverted legs included —
-- they are measures, not filters). Foreign keys are the natural keys of the
-- three dimensions; flight_number and route ride along as degenerate
-- dimensions. Delay-outcome measures live here by design: the leakage
-- boundary is the ML feature mart, not analytics (CLAUDE.md §9).

select
    -- foreign keys
    flight_date as date_key,
    carrier as carrier_key,
    origin as origin_airport_key,
    dest as dest_airport_key,

    -- degenerate dimensions
    flight_number,
    route,
    crs_dep_time,
    crs_dep_hour,
    crs_arr_time,
    crs_arr_hour,
    day_of_week,

    -- measures: schedule & distance
    distance,
    crs_elapsed_time,

    -- measures: outcomes
    dep_delay_minutes,
    dep_del15,
    taxi_out,
    taxi_in,
    air_time,
    actual_elapsed_time,
    arr_delay_minutes,
    arr_del15,
    cancelled,
    cancellation_code,
    diverted,
    carrier_delay,
    weather_delay,
    nas_delay,
    security_delay,
    late_aircraft_delay
from {{ ref('stg_gold__flights') }}
