{{
    config(
        materialized='table',
        partition_by={'field': 'flight_date', 'data_type': 'date', 'granularity': 'month'},
        cluster_by=['origin', 'dest', 'reporting_airline'],
    )
}}

-- The silver flights table: one row per flight leg, every cleaned column from
-- staging INCLUDING post-departure/arrival outcomes (analytics uses them; the
-- leakage boundary is the gold ML mart, CLAUDE.md §9), enriched with origin
-- and destination airport coordinates, elevation and IANA timezone.
-- Left joins on stg_airports.iata (unique-tested) cannot fan out or drop
-- rows: row count stays exactly stg_bts_flights'. Flights to airports missing
-- from the reference would carry NULL enrichment; the relationships tests
-- pin that set to zero.

with flights as (

    select * from {{ ref('stg_bts_flights') }}

),

airports as (

    select * from {{ ref('stg_airports') }}

)

select
    flights.*,
    origin_airport.name as origin_airport_name,
    origin_airport.latitude as origin_latitude,
    origin_airport.longitude as origin_longitude,
    origin_airport.elevation_ft as origin_elevation_ft,
    origin_airport.tz as origin_tz,
    dest_airport.name as dest_airport_name,
    dest_airport.latitude as dest_latitude,
    dest_airport.longitude as dest_longitude,
    dest_airport.elevation_ft as dest_elevation_ft,
    dest_airport.tz as dest_tz
from flights
left join airports as origin_airport
    on flights.origin = origin_airport.iata
left join airports as dest_airport
    on flights.dest = dest_airport.iata
