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
--     (pre-cutoff flights only), SMOOTHED toward the training-window global:
--     (n * entity_rate + m * global_rate) / (n + m),
--     m = var('hist_smoothing_prior_strength'). The smoothed value is
--     CONSTANT within an entity — identical for every row, train and test —
--     so no per-row channel of any kind can exist, and the prior damps the
--     single-flight-entity case (own-outcome weight 1/(1+m)). Residual
--     self-inclusion, stated precisely: the own-label weight is 1/(n+m),
--     maximal at n=1 where the feature takes one of two values —
--     (50g)/51 ≈ 0.206 or (1+50g)/51 ≈ 0.226 — so for the ~102
--     single-flight-route TRAINING rows the value does encode that row's
--     label. ACCEPTED anyway (owner decision): 102 of 16,678,880 training
--     rows is immaterial to the fit, and there is no test-side effect —
--     test rows on those routes receive training-window information only.
--     v2's LOO removed this residual exactly but opened the worse per-row
--     artifact described below.
--     m = 50: below ~50 observations an entity rate is noise-dominated and
--     should shrink hard toward the prior; test metrics are insensitive to
--     the choice (m in {10, 50, 100}: XGB ROC 0.6806/0.6809/0.6805,
--     PR-AUC 0.3478/0.3482/0.3476 — deltas < 0.001).
--     The global derives from the carrier level of the shared model (which
--     partitions all pre-cutoff completed flights exactly once). It is one
--     scalar for the whole table, so it cannot carry per-row signal; an
--     entity's own flights re-entering its feature via the prior are bounded
--     by m*n/(N*(n+m)) <= 3.0e-6 of the value at every grain (N = 16.7M).
--     History of this block: v1 joined raw rates (self-inclusion: a
--     single-flight route's feature equaled its own label); v2 replaced that
--     with leave-one-out, which created the classic TARGET-ENCODING ARTIFACT
--     — per-row micro-perturbations anti-correlated with the training label
--     that HANDICAPPED the boosted trees. Note the artifact degraded model
--     quality only; it never inflated metrics (test features never contained
--     test labels — reported numbers were honest throughout). v3 (current)
--     uses smoothed raw rates.
--     Remaining documented property: rates aggregate the whole pre-cutoff
--     window (a 2022 row sees 2023 peers' outcomes). Accepted: the
--     shared-model requirement (one definition for marts + ML) rules out
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

-- Training-window GLOBAL rate/avg for the smoothing prior. The carrier level
-- partitions every completed pre-cutoff flight exactly once, so its weighted
-- average IS the global — still a single definition, still the shared model.
globals as (

    select
        sum(n_flights * arr_del15_rate) / sum(n_flights) as global_arr_del15_rate,
        sum(n_flights * avg_arr_delay_minutes) / sum(n_flights) as global_avg_arr_delay_minutes
    from rates
    where entity_level = 'carrier'

),

joined as (

    select
        flights.*,
        globals.global_arr_del15_rate,
        globals.global_avg_arr_delay_minutes,
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
    cross join globals
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

    -- shared historical delay rates, SMOOTHED toward the training-window
    -- global: (n*p + m*global)/(n + m). Identical formula for every grain and
    -- every row (train and test): the value is constant within an entity, so
    -- no per-row channel can exist, and the prior handles tiny-n entities.
    -- See the header for the v1 (raw) -> v2 (LOO, target-encoding artifact
    -- that handicapped the booster — never inflated metrics) -> v3 (this)
    -- history. NULL stays NULL for entities absent from the training window.
    {% set m = var('hist_smoothing_prior_strength') %}
    {% for grain in ['route', 'carrier', 'origin', 'dest'] %}
    ({{ grain }}_n_raw * {{ grain }}_rate_raw + {{ m }} * global_arr_del15_rate)
        / ({{ grain }}_n_raw + {{ m }}) as hist_{{ grain }}_arr_del15_rate,
    ({{ grain }}_n_raw * {{ grain }}_avg_raw + {{ m }} * global_avg_arr_delay_minutes)
        / ({{ grain }}_n_raw + {{ m }}) as hist_{{ grain }}_avg_arr_delay_minutes,
    {{ grain }}_n_raw as hist_{{ grain }}_n_flights,
    {% endfor %}

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
