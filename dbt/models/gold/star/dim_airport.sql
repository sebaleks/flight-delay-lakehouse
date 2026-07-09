{{ config(materialized='table') }}

-- Airport dimension: one row per IATA code, with geography, timezone, and the
-- mapped nearest GSOD weather station as attributes.

select
    airports.iata as airport_key,
    airports.name as airport_name,
    airports.city,
    airports.country,
    airports.latitude,
    airports.longitude,
    airports.elevation_ft,
    airports.tz,
    map.station_id as weather_station_id,
    map.distance_km as weather_station_distance_km
from {{ ref('stg_airports') }} as airports
left join {{ ref('airport_station_map') }} as map
    on airports.iata = map.iata
