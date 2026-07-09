-- Standing leakage guard: the ML mart's origin weather must derive from the
-- PRIOR day (obs_date = flight_date - 1), never the flight's own day. Fails
-- with one row per violation:
--   * has_flag_but_no_prior_day_obs   — has_origin_weather is true yet no
--     prior-day observation exists (the signature of a same-day revert);
--   * no_flag_but_prior_day_exists    — flag false despite prior-day data;
--   * values_not_prior_day            — weather values differ from the
--     prior-day observation (wrong source row).
-- Together these pin the lag at the value level, not just the SQL text.

with features as (

    select flight_date, origin, has_origin_weather,
        origin_mean_temp_f, origin_precip_in, origin_mean_wind_speed_kn
    from {{ ref('ml_flight_features') }}

),

prior_day as (

    select iata, obs_date, mean_temp_f, precip_in, mean_wind_speed_kn
    from {{ ref('stg_gold__weather') }}

),

checked as (

    select
        features.flight_date,
        features.origin,
        features.has_origin_weather,
        features.origin_mean_temp_f,
        features.origin_precip_in,
        features.origin_mean_wind_speed_kn,
        prior_day.iata is not null as prior_day_exists,
        prior_day.mean_temp_f,
        prior_day.precip_in,
        prior_day.mean_wind_speed_kn
    from features
    left join prior_day
        on features.origin = prior_day.iata
        and prior_day.obs_date = date_sub(features.flight_date, interval 1 day)

)

select flight_date, origin, 'has_flag_but_no_prior_day_obs' as violation
from checked
where has_origin_weather and not prior_day_exists

union all

select flight_date, origin, 'no_flag_but_prior_day_exists' as violation
from checked
where not has_origin_weather and prior_day_exists

union all

select flight_date, origin, 'values_not_prior_day' as violation
from checked
where
    has_origin_weather
    and (
        origin_mean_temp_f is distinct from mean_temp_f
        or origin_precip_in is distinct from precip_in
        or origin_mean_wind_speed_kn is distinct from mean_wind_speed_kn
    )
