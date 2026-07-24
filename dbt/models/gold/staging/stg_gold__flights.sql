{{
    config(
        materialized='table',
        partition_by={'field': 'flight_date', 'data_type': 'date', 'granularity': 'month'},
        cluster_by=['origin', 'dest', 'carrier'],
    )
}}

-- Gold staging over silver_flights: the column subset gold consumes, plus the
-- derived keys/buckets every downstream model shares (route, scheduled-hour
-- buckets) so they are defined exactly once. One row per flight leg,
-- including cancelled/diverted legs (fact keeps them; the ML mart filters).

select
    flight_date,
    year,
    month,
    day_of_week,  -- BTS convention: 1 = Monday .. 7 = Sunday
    reporting_airline as carrier,
    dot_id_reporting_airline,
    iata_code_reporting_airline,
    flight_number_reporting_airline as flight_number,
    origin,
    origin_tz,  -- IANA tz from the airports seed: local schedule -> UTC joins
    dest,
    concat(origin, '-', dest) as route,
    crs_dep_time,
    extract(hour from crs_dep_time) as crs_dep_hour,
    crs_arr_time,
    extract(hour from crs_arr_time) as crs_arr_hour,
    distance,
    crs_elapsed_time,
    cancelled,
    cancellation_code,
    diverted,
    dep_time,
    dep_delay_minutes,
    dep_del15,
    taxi_out,
    taxi_in,
    air_time,
    actual_elapsed_time,
    arr_time,
    arr_delay_minutes,
    arr_del15,
    carrier_delay,
    weather_delay,
    nas_delay,
    security_delay,
    late_aircraft_delay
from {{ ref('silver_flights') }}
