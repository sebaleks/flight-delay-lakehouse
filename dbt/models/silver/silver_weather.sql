{{
    config(
        materialized='table',
        partition_by={'field': 'obs_date', 'data_type': 'date', 'granularity': 'month'},
        cluster_by=['station_id'],
    )
}}

-- The silver weather table: GSOD daily observations 2022-2024 restricted to
-- the stations that serve as some airport's nearest station. This is a
-- relevance filter, not row loss — the full typed feed remains available in
-- stg_weather. Joining flights to weather goes through airport_station_map
-- (airport -> station_id) on obs_date.
--
-- COVERAGE CAVEAT: a mapped station is not guaranteed a row for every date
-- (stations gap up to ~18% of days, plausibly biased toward severe weather).
-- Gold consumers must LEFT JOIN and treat missing weather as a signal of its
-- own — an inner join would silently drop exactly the interesting days.

with mapped_stations as (

    select distinct station_id from {{ ref('airport_station_map') }}

)

select stg_weather.*
from {{ ref('stg_weather') }} as stg_weather
inner join mapped_stations using (station_id)
