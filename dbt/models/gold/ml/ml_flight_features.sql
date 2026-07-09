{{
    config(
        materialized='table',
        partition_by={'field': 'flight_date', 'data_type': 'date', 'granularity': 'month'},
        cluster_by=['origin', 'dest', 'carrier'],
    )
}}

-- ============================================================================
-- ML FEATURE MART — the leakage boundary (CLAUDE.md §9).
-- One row per completed flight (not cancelled, not diverted, labels present).
-- Every feature column must be knowable BEFORE departure:
--   * identifiers/schedule: carrier, origin, dest, route, distance, scheduled
--     dep/arr times & hour buckets, day of week, month — published schedule;
--   * hist_* rates: from the SHARED int_historical_delay_rates model
--     (pre-cutoff flights only). For TRAINING rows the row's own outcome is
--     removed with a LEAVE-ONE-OUT adjustment computed algebraically from the
--     shared model's n and rate — so no training row's feature contains its
--     own label, and the rate definition still lives exactly once. LOO
--     applies only to route/carrier/origin, where the flight is inside its
--     own aggregate; hist_dest_* stays raw for all rows because the
--     destination's outbound aggregate never contains the arriving flight
--     (leave-one-out by construction). Test rows keep the raw rates
--     (they are not in any aggregate at all).
--     Remaining documented property: training-row rates aggregate the whole
--     pre-cutoff window (a 2022 row sees 2023 peers' outcomes). Accepted:
--     the shared-model requirement (one definition for marts + ML) rules out
--     per-flight as-of-date rates; anyone carving a validation slice out of
--     the training window must re-derive rates as-of that slice.
--     Entities new in the test window stay NULL — never zero-filled.
--     hist_dest_* are the destination airport's OUTBOUND (origin-grain)
--     rates — a congestion propensity proxy, not inbound arrival performance.
--   * origin_* weather: PRIOR-DAY daily GSOD at the origin, joined strictly
--     on obs_date = flight_date - 1, so every weather feature (including
--     has_origin_weather, the prior-day-observation-exists flag) is fully
--     observed before any departure on flight_date. Missing prior-day
--     weather stays NULL with has_origin_weather = false (missing-as-signal;
--     note all 2022-01-01 rows are false — 2021-12-31 predates the ingested
--     GSOD window). assert_ml_weather_is_prior_day pins the lag so the join
--     can never silently revert to same-day. A production system would use
--     forecast-issued-at-T data; prior-day observations are the leakage-safe
--     stand-in.
--   * holiday flags: generated calendar — known years ahead.
-- Labels are prefixed label_ and are the ONLY post-departure columns.
-- assert_ml_features_no_leakage pins this table's schema to the audited
-- column allowlist and fails on ANY unexpected column.
-- is_training_row derives from the same cutoff var as the shared rates —
-- split on it (or flight_date), never randomly.
-- ============================================================================

with flights as (

    select *
    from {{ ref('stg_gold__flights') }}
    where
        not cancelled
        and not diverted
        and arr_del15 is not null
        and arr_delay_minutes is not null

),

rates as (

    select * from {{ ref('int_historical_delay_rates') }}

),

joined as (

    select
        flights.*,
        flights.flight_date < date('{{ var("train_test_cutoff_date") }}') as is_training_row,
        {% for grain, key in [('route', 'flights.route'), ('carrier', 'flights.carrier'),
                              ('origin', 'flights.origin'), ('dest', 'flights.dest')] %}
        {{ grain }}_rates.arr_del15_rate as {{ grain }}_rate_raw,
        {{ grain }}_rates.avg_arr_delay_minutes as {{ grain }}_avg_raw,
        {{ grain }}_rates.n_flights as {{ grain }}_n_raw,
        {% endfor %}
        weather.mean_temp_f,
        weather.max_temp_f,
        weather.min_temp_f,
        weather.visibility_mi,
        weather.mean_wind_speed_kn,
        weather.max_gust_kn,
        weather.precip_in,
        weather.snow_depth_in,
        weather.had_fog,
        weather.had_rain_drizzle,
        weather.had_snow_ice_pellets,
        weather.had_thunder,
        weather.iata is not null as has_origin_weather,
        coalesce(holidays.is_holiday, false) as is_holiday,
        coalesce(holidays.is_day_before_holiday, false) as is_day_before_holiday,
        coalesce(holidays.is_day_after_holiday, false) as is_day_after_holiday
    from flights
    left join rates as route_rates
        on route_rates.entity_level = 'route' and route_rates.entity_key = flights.route
    left join rates as carrier_rates
        on carrier_rates.entity_level = 'carrier' and carrier_rates.entity_key = flights.carrier
    left join rates as origin_rates
        on origin_rates.entity_level = 'airport' and origin_rates.entity_key = flights.origin
    left join rates as dest_rates
        on dest_rates.entity_level = 'airport' and dest_rates.entity_key = flights.dest
    -- PRIOR-DAY join (leakage boundary): weather observed the day BEFORE the
    -- flight — never obs_date = flight_date, which would span post-departure
    -- hours. Pinned by assert_ml_weather_is_prior_day.
    left join {{ ref('stg_gold__weather') }} as weather
        on flights.origin = weather.iata
        and weather.obs_date = date_sub(flights.flight_date, interval 1 day)
    left join {{ ref('stg_holidays') }} as holidays
        on flights.flight_date = holidays.date_day

)

select
    -- keys (identify the row; carrier/origin/dest double as categorical features)
    flight_date,
    carrier,
    flight_number,
    origin,
    dest,
    route,
    crs_dep_time,

    -- schedule features (pre-departure by definition)
    distance,
    crs_dep_hour,
    crs_arr_hour,
    day_of_week,
    month,

    -- shared historical delay rates. LOO applies ONLY where the flight is
    -- genuinely inside its own aggregate: route, carrier, and origin (the
    -- airport level is origin-grain). Training rows there get leave-one-out
    -- (own outcome removed; NULL when the row was the entity's only
    -- pre-cutoff flight). Test rows: raw shared rates (never in the
    -- aggregate).
    {% for grain in ['route', 'carrier', 'origin'] %}
    case
        when not is_training_row then {{ grain }}_rate_raw
        when {{ grain }}_n_raw > 1
            then ({{ grain }}_n_raw * {{ grain }}_rate_raw - cast(arr_del15 as int64))
                / ({{ grain }}_n_raw - 1)
    end as hist_{{ grain }}_arr_del15_rate,
    case
        when not is_training_row then {{ grain }}_avg_raw
        when {{ grain }}_n_raw > 1
            then ({{ grain }}_n_raw * {{ grain }}_avg_raw - arr_delay_minutes)
                / ({{ grain }}_n_raw - 1)
    end as hist_{{ grain }}_avg_arr_delay_minutes,
    case
        when not is_training_row then {{ grain }}_n_raw
        else {{ grain }}_n_raw - 1
    end as hist_{{ grain }}_n_flights,
    {% endfor %}

    -- hist_dest_*: the destination airport's OUTBOUND aggregate. An arriving
    -- flight is never part of it, so the raw lookup is already leave-one-out
    -- by construction — applying the LOO algebra here would subtract an
    -- outcome that was never in the sum (and force valid single-flight
    -- airports to NULL). Raw for all rows, training included.
    dest_rate_raw as hist_dest_arr_del15_rate,
    dest_avg_raw as hist_dest_avg_arr_delay_minutes,
    dest_n_raw as hist_dest_n_flights,

    -- origin weather for the flight date (daily GSOD; see header note)
    mean_temp_f as origin_mean_temp_f,
    max_temp_f as origin_max_temp_f,
    min_temp_f as origin_min_temp_f,
    visibility_mi as origin_visibility_mi,
    mean_wind_speed_kn as origin_mean_wind_speed_kn,
    max_gust_kn as origin_max_gust_kn,
    precip_in as origin_precip_in,
    snow_depth_in as origin_snow_depth_in,
    had_fog as origin_had_fog,
    had_rain_drizzle as origin_had_rain_drizzle,
    had_snow_ice_pellets as origin_had_snow_ice_pellets,
    had_thunder as origin_had_thunder,
    has_origin_weather,

    -- holiday flags (generated calendar, knowable years ahead)
    is_holiday,
    is_day_before_holiday,
    is_day_after_holiday,

    -- time-based split marker (same var as the shared rates — never random)
    is_training_row,

    -- labels (the only post-departure columns, explicitly prefixed)
    arr_del15 as label_arr_del15,
    arr_delay_minutes as label_arr_delay_minutes

from joined
