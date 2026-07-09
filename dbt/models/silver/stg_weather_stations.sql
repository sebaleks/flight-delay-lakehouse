{{ config(materialized='table') }}

-- NOAA station registry, read in place from the public dataset. One row per
-- station_id (usaf-wban), deduplicated deterministically preferring the most
-- recently active record. Used later to match airports to their nearest
-- weather station.

select
    concat(usaf, '-', wban) as station_id,
    usaf as usaf_id,
    wban as wban_id,
    nullif(trim(name), '') as station_name,
    nullif(trim(country), '') as country,
    nullif(trim(state), '') as state,
    nullif(trim(`call`), '') as call_sign,
    lat as latitude,
    lon as longitude,
    safe_cast(elev as float64) as elevation_m,
    safe.parse_date('%Y%m%d', begin) as active_from,
    safe.parse_date('%Y%m%d', `end`) as active_to
from {{ source('noaa', 'stations') }} as stations
qualify row_number() over (
    partition by usaf, wban
    order by
        safe.parse_date('%Y%m%d', `end`) desc nulls last,
        farm_fingerprint(to_json_string(stations))
) = 1
