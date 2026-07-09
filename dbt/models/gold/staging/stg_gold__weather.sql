{{
    config(
        materialized='table',
        partition_by={'field': 'obs_date', 'data_type': 'date', 'granularity': 'month'},
        cluster_by=['iata'],
    )
}}

-- Gold staging: daily GSOD weather keyed by AIRPORT (via each airport's
-- nearest mapped station) so consumers join on (iata, date) without knowing
-- about stations. Grain: one row per airport per observed date. A mapped
-- station can gap on individual dates — consumers must LEFT JOIN and treat
-- missing weather as a signal (see silver_weather).

select
    map.iata,
    weather.obs_date,
    map.station_id,
    map.distance_km as station_distance_km,
    weather.mean_temp_f,
    weather.max_temp_f,
    weather.min_temp_f,
    weather.mean_dewpoint_f,
    weather.mean_sea_level_pressure_mb,
    weather.mean_station_pressure_mb,
    weather.visibility_mi,
    weather.mean_wind_speed_kn,
    weather.max_sustained_wind_kn,
    weather.max_gust_kn,
    weather.precip_in,
    weather.snow_depth_in,
    weather.had_fog,
    weather.had_rain_drizzle,
    weather.had_snow_ice_pellets,
    weather.had_hail,
    weather.had_thunder,
    weather.had_tornado_funnel_cloud
from {{ ref('airport_station_map') }} as map
inner join {{ ref('silver_weather') }} as weather using (station_id)
