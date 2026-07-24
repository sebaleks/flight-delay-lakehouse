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
--   * origin_* weather: the LAST hourly ISD observation at the origin AT OR
--     BEFORE the flight's SCHEDULED departure, inside a 3-hour lookback:
--       obs_ts_utc in (dep_ts_utc - 3h, dep_ts_utc], latest wins, where
--       dep_ts_utc = timestamp(datetime(flight_date, crs_dep_time), origin_tz)
--     crs_dep_time is the published schedule — the ACTUAL dep_time never
--     enters — so every observation is pre-departure BY CONSTRUCTION,
--     including same-day conditions. (v1..v3 of this mart used PRIOR-DAY
--     daily GSOD because daily grain cannot bound same-day data at departure
--     time; hourly grain can. The features now MEAN "conditions at the
--     scheduled departure hour" — a semantic shift from the daily-summary
--     era, deliberate and owner-approved.)
--     TIMEZONES: ISD timestamps are UTC, BTS schedule times are LOCAL; the
--     join converts via the airports-seed IANA tz (origin_tz, pinned by
--     assert_flown_airports_have_timezone). An unconverted join would reach
--     up to a UTC-offset past departure. DST gap times (a handful of 02:xx
--     schedules a year) resolve deterministically inside TIMESTAMP(); worst
--     case one hour, still bounded by the <= predicate.
--     The 3-HOUR LOOKBACK is a STALENESS CEILING, not a coverage knob:
--     mapped majors report ~2 obs/hour (METAR + specials), so the window is
--     nowhere near binding there; it binds only at SPARSE stations (e.g.
--     arctic BTI), where a >3h-old observation would otherwise be presented
--     as "conditions at departure". Past the ceiling we prefer honest
--     missingness: no observation in the window -> all weather NULL,
--     has_origin_weather = false (missing-as-signal). The lookback may cross
--     local midnight (a 00:30 red-eye can use a 23:5x prior-evening obs) —
--     still strictly pre-departure.
--     Value semantics: origin_visibility_mi is RIGHT-CENSORED at 10.0 (the
--     source caps VIS at 16093 m — 10.0 means "10 or better");
--     origin_gust_kn is 0.0 when the chosen observation reports no gust
--     group (calm-hours encoding, owner decision) with origin_gust_reported
--     carrying the distinction — never left NaN for the imputer to fill
--     with a typical gust; origin_precip_1h_in uses ONLY 1-hour accumulation
--     groups, never mixed windows (policy + QC nulling in silver_isd_hourly).
--     assert_ml_weather_obs_before_departure pins obs <= scheduled departure
--     AND the lookback window at the value level over the FULL table via the
--     origin_weather_obs_ts_utc bookkeeping column.
--   * holiday flags: generated calendar — known years ahead.
-- Labels are prefixed label_ and are the ONLY post-departure columns.
-- assert_ml_features_no_leakage pins this table's schema to the audited
-- column allowlist and fails on ANY unexpected column.
-- is_training_row derives from the same cutoff var as the shared rates —
-- split on it (or flight_date), never randomly.
-- ============================================================================

with flights as (

    -- dep_ts_utc: scheduled departure in UTC — LOCAL wall clock + seed tz.
    -- The only time column here is crs_dep_time (published schedule); actual
    -- dep_time is never referenced. Known limitation: BTS '2400' scheduled
    -- times are stored as 00:00 of flight_date (staging convention), so a
    -- true midnight-END-of-day departure gets weather up to ~24h STALE —
    -- never future, never a leak; population counted in the build report.
    select
        *,
        timestamp(datetime(flight_date, crs_dep_time), origin_tz) as dep_ts_utc
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
        weather.temp_f,
        weather.dewpoint_f,
        weather.wind_speed_kn,
        weather.gust_kn,
        weather.gust_reported,
        weather.visibility_mi,
        weather.precip_1h_in,
        weather.had_fog,
        weather.had_rain_drizzle,
        weather.had_snow_ice_pellets,
        weather.had_thunder,
        weather.station_id is not null as has_origin_weather,
        weather.obs_ts_utc as origin_weather_obs_ts_utc,
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
    -- AT-OR-BEFORE hourly join (leakage boundary): the LAST ISD observation
    -- at or before SCHEDULED departure, never after it, bounded by the
    -- 3-hour staleness ceiling. Pinned at the value level by
    -- assert_ml_weather_obs_before_departure. See header for tz handling.
    -- wx_cand_date enumerates the 1-2 UTC dates the lookback window can
    -- touch, giving the join a (station_id, obs_date) EQUI-key so BigQuery
    -- hash-joins ~2 days of observations per flight instead of enumerating a
    -- station's full 3-year history against every hub departure.
    -- LEFT join (not cross): a NULL dep_ts_utc (impossible today — tz and
    -- crs_dep_time are guarded non-null upstream — but structural) yields a
    -- NULL array, and a cross join would silently DROP the flight row;
    -- left join keeps it on the all-NULL-weather path instead
    left join unnest(generate_date_array(
        date(timestamp_sub(flights.dep_ts_utc, interval 3 hour)),
        date(flights.dep_ts_utc)
    )) as wx_cand_date on true
    left join {{ ref('airport_station_map') }} as station_map
        on flights.origin = station_map.iata
    left join {{ ref('silver_isd_hourly') }} as weather
        on weather.station_id = station_map.station_id
        and weather.obs_date = wx_cand_date
        and weather.obs_ts_utc <= flights.dep_ts_utc
        and weather.obs_ts_utc > timestamp_sub(flights.dep_ts_utc, interval 3 hour)
    left join {{ ref('stg_holidays') }} as holidays
        on flights.flight_date = holidays.date_day
    -- one row per flight survives: the latest in-window observation across
    -- both candidate dates, or a bare flight row (weather.* NULL) when no
    -- observation exists — DESC puts NULL obs_ts last, so an observation
    -- always beats the no-match copy. Partitioning by the natural key is
    -- safe: stg_bts_flights carries the authoritative uniqueness test on it
    -- (this QUALIFY makes the mart's own uniqueness test pass by
    -- construction, so the UPSTREAM test is the real duplicate guard).
    qualify row_number() over (
        partition by
            flights.flight_date, flights.carrier, flights.flight_number,
            flights.origin, flights.dest, flights.crs_dep_time
        order by weather.obs_ts_utc desc
    ) = 1

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

    -- origin weather AT THE SCHEDULED DEPARTURE HOUR (hourly ISD; see the
    -- header for the join predicate, tz handling, and value semantics)
    temp_f as origin_temp_f,
    dewpoint_f as origin_dewpoint_f,
    wind_speed_kn as origin_wind_speed_kn,
    -- observation reports a gust -> value; observation without a gust group
    -- -> 0.0 + indicator false (calm-hours encoding, owner decision); no
    -- observation at all -> NULL like every other weather feature
    case when has_origin_weather then coalesce(gust_kn, 0.0) end as origin_gust_kn,
    case when has_origin_weather then gust_reported end as origin_gust_reported,
    -- uniform right-censoring at 10.0: most stations cap VIS at 16093 m but
    -- 0.16% of observations (extended-reporting software) exceed it — whether
    -- a station caps is instrumentation, not weather; silver keeps raw values
    least(visibility_mi, 10.0) as origin_visibility_mi,
    precip_1h_in as origin_precip_1h_in,
    had_fog as origin_had_fog,
    had_rain_drizzle as origin_had_rain_drizzle,
    had_snow_ice_pellets as origin_had_snow_ice_pellets,
    had_thunder as origin_had_thunder,
    has_origin_weather,
    -- bookkeeping, NOT a feature (EXCLUDED in ml/features.py): the timestamp
    -- of the chosen observation, kept so the standing guard can prove
    -- obs <= scheduled departure over the whole table at any time
    origin_weather_obs_ts_utc,

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
