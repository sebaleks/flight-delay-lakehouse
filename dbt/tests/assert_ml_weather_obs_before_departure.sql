-- Standing leakage guard (replaces assert_ml_weather_is_prior_day, retired
-- with the daily-GSOD join): every origin-weather observation feeding the ML
-- mart must sit AT OR BEFORE the flight's SCHEDULED departure and inside the
-- 3-hour staleness ceiling. Value-level over the FULL table — the mart
-- carries origin_weather_obs_ts_utc (bookkeeping, not a feature) for exactly
-- this check, and dep_ts_utc is RECOMPUTED here from the seed tz rather than
-- trusted from the mart's own arithmetic. Fails one row per violation:
--   * obs_after_scheduled_departure — the leak signature: weather from at or
--     after the moment the schedule says wheels-away prep begins;
--   * obs_outside_lookback_window   — staleness ceiling breached (a silent
--     window widening would surface here);
--   * flag_true_but_no_obs_ts / flag_false_but_obs_ts_present — the
--     has_origin_weather flag must exactly mirror observation presence;
--   * weather_path_dead_or_degraded — anti-vacuity: if the station map or
--     the hourly silver table stopped matching, every row would be
--     has_origin_weather = false and the four checks above would pass
--     emptily. Observed missingness is 0.065%; > 5% means broken wiring,
--     not weather.
-- crs_dep_time is the published schedule; actual departure times appear
-- nowhere in this test or the mart's weather path.

with features as (

    select
        flight_date,
        origin,
        crs_dep_time,
        has_origin_weather,
        origin_weather_obs_ts_utc
    from {{ ref('ml_flight_features') }}

),

airports as (

    select iata, tz from {{ ref('stg_airports') }}

),

checked as (

    select
        features.*,
        timestamp(
            datetime(features.flight_date, features.crs_dep_time), airports.tz
        ) as dep_ts_utc
    from features
    left join airports
        on features.origin = airports.iata

)

select flight_date, origin, crs_dep_time, 'obs_after_scheduled_departure' as violation
from checked
where origin_weather_obs_ts_utc > dep_ts_utc

union all

select flight_date, origin, crs_dep_time, 'obs_outside_lookback_window' as violation
from checked
where origin_weather_obs_ts_utc <= timestamp_sub(dep_ts_utc, interval 3 hour)

union all

select flight_date, origin, crs_dep_time, 'flag_true_but_no_obs_ts' as violation
from checked
where has_origin_weather and origin_weather_obs_ts_utc is null

union all

-- fail CLOSED, not open: an unverifiable row (recomputed dep_ts NULL — tz
-- missing from the seed for a flight that HAS a weather obs) is a failure,
-- not a silent pass; NULL comparisons above would otherwise skip such rows
select flight_date, origin, crs_dep_time, 'cannot_verify_null_dep_ts' as violation
from checked
where origin_weather_obs_ts_utc is not null and dep_ts_utc is null

union all

select flight_date, origin, crs_dep_time, 'flag_false_but_obs_ts_present' as violation
from checked
where not has_origin_weather and origin_weather_obs_ts_utc is not null

union all

select
    date '1900-01-01' as flight_date,
    'ALL' as origin,
    time '00:00:00' as crs_dep_time,
    'weather_path_dead_or_degraded' as violation
from (
    select countif(not has_origin_weather) / count(*) as missing_frac
    from features
)
where missing_frac > 0.05
