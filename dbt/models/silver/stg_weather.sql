{{
    config(
        materialized='table',
        partition_by={'field': 'obs_date', 'data_type': 'date', 'granularity': 'month'},
        cluster_by=['station_id'],
    )
}}

-- NOAA GSOD daily surface summaries 2022-2024, read in place from the public
-- dataset (never copied to bronze — CLAUDE.md §8), typed and with NOAA
-- sentinel values (9999.9 / 999.9 / 99.99 = "missing") nulled out.
-- Units preserved from GSOD: temperatures °F, wind knots, visibility miles,
-- precipitation/snow depth inches, pressure millibars.
-- obs_date is constructed from year/mo/da so the model does not depend on
-- newer-vintage convenience columns existing in every year table.

{% set year_tables = ['gsod2022', 'gsod2023', 'gsod2024'] %}

with unioned as (

    {% for t in year_tables %}
    select
        stn, wban, year, mo, da,
        temp, count_temp, dewp, count_dewp, slp, count_slp, stp, count_stp,
        visib, count_visib, wdsp, count_wdsp, mxpsd, gust,
        `max`, flag_max, `min`, flag_min, prcp, flag_prcp, sndp,
        fog, rain_drizzle, snow_ice_pellets, hail, thunder, tornado_funnel_cloud
    from {{ source('noaa', t) }}
    {% if not loop.last %}union all{% endif %}
    {% endfor %}

)

select
    concat(stn, '-', wban) as station_id,
    stn as usaf_id,
    wban as wban_id,
    date(
        safe_cast(year as int64), safe_cast(mo as int64), safe_cast(da as int64)
    ) as obs_date,
    nullif(temp, 9999.9) as mean_temp_f,
    count_temp as n_temp_obs,
    nullif(dewp, 9999.9) as mean_dewpoint_f,
    count_dewp as n_dewpoint_obs,
    nullif(slp, 9999.9) as mean_sea_level_pressure_mb,
    count_slp as n_sea_level_pressure_obs,
    -- GSOD stp quirks (verified empirically 2022-2024): 999.9 is the de facto
    -- missing marker here (the documented 9999.9 never occurs), and values
    -- >= 1000 mb are stored with the leading '1' dropped (13.5 = 1013.5 mb;
    -- nothing occurs in [100, 500), so the decode rule is unambiguous)
    case
        when stp in (999.9, 9999.9) then null
        when stp < 100 then stp + 1000
        else stp
    end as mean_station_pressure_mb,
    count_stp as n_station_pressure_obs,
    nullif(visib, 999.9) as visibility_mi,
    count_visib as n_visibility_obs,
    nullif(safe_cast(wdsp as float64), 999.9) as mean_wind_speed_kn,
    safe_cast(count_wdsp as int64) as n_wind_obs,
    nullif(safe_cast(mxpsd as float64), 999.9) as max_sustained_wind_kn,
    nullif(gust, 999.9) as max_gust_kn,
    nullif(`max`, 9999.9) as max_temp_f,
    nullif(trim(flag_max), '') as max_temp_flag,
    nullif(`min`, 9999.9) as min_temp_f,
    nullif(trim(flag_min), '') as min_temp_flag,
    nullif(prcp, 99.99) as precip_in,
    nullif(trim(flag_prcp), '') as precip_flag,
    nullif(sndp, 999.9) as snow_depth_in,
    fog = '1' as had_fog,
    rain_drizzle = '1' as had_rain_drizzle,
    snow_ice_pellets = '1' as had_snow_ice_pellets,
    hail = '1' as had_hail,
    thunder = '1' as had_thunder,
    tornado_funnel_cloud = '1' as had_tornado_funnel_cloud
from unioned
