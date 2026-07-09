{{ config(materialized='table') }}

-- Maps each reference airport to its nearest NOAA GSOD station by geodesic
-- distance (BigQuery geography functions). Candidate stations must have
-- coordinates and substantial observation coverage in the 2022-2024 window
-- (>= min_station_obs_days of 1096 possible; default 900) so an airport is
-- never mapped to a station that barely reports. distance_km is kept so
-- mapping quality is auditable; ties break deterministically on station_id.

with airports as (

    select iata, name as airport_name, tz as airport_tz, latitude, longitude
    from {{ ref('stg_airports') }}
    where latitude is not null and longitude is not null

),

station_coverage as (

    select station_id, count(distinct obs_date) as n_obs_days
    from {{ ref('stg_weather') }}
    group by station_id

),

stations as (

    select
        s.station_id,
        s.station_name,
        s.latitude,
        s.longitude,
        c.n_obs_days
    from {{ ref('stg_weather_stations') }} as s
    inner join station_coverage as c using (station_id)
    where
        s.latitude is not null and s.longitude is not null
        and not (s.latitude = 0 and s.longitude = 0)
        and c.n_obs_days >= {{ var('min_station_obs_days', 900) }}

),

ranked as (

    select
        airports.iata,
        airports.airport_name,
        airports.airport_tz,
        stations.station_id,
        stations.station_name,
        stations.n_obs_days,
        st_distance(
            st_geogpoint(airports.longitude, airports.latitude),
            st_geogpoint(stations.longitude, stations.latitude)
        ) / 1000 as distance_km
    from airports
    cross join stations

),

nearest as (

    -- rank on the TRUE distance from `ranked` — rounding first would rank in
    -- 10 m buckets and let the station_id tiebreak pick the farther station
    select *
    from ranked
    qualify row_number() over (
        partition by iata
        order by distance_km, station_id
    ) = 1

)

select
    iata,
    airport_name,
    airport_tz,
    station_id,
    station_name,
    n_obs_days,
    round(distance_km, 2) as distance_km
from nearest
