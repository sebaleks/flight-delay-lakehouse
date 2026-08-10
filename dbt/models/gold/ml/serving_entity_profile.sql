{{ config(materialized='table') }}

-- ============================================================================
-- SERVING LOOKUP — the constant-within-entity hist values, one row per entity.
--
-- WHY THIS EXISTS: ml/serving.py used to issue one BigQuery query PER GRAIN PER
-- REQUEST (route/carrier/origin/dest), plus one for route distance, plus two
-- more at startup for the rotation grains. None of them could partition-prune,
-- so every /predict call scanned ~2.7 GB of the 20.2M-row mart to fetch a few
-- thousand constants that only change when the mart is rebuilt. This model
-- materializes them once per dbt build; serving reads the whole thing at
-- startup into a dict and the request path issues ZERO queries.
--
-- THE PARITY RULE: each SELECT below must stay the SAME EXPRESSION serving used
-- to run — `any_value(hist_<grain>_<stat>) ... group by <key>`. The hist values
-- are read from the mart via any_value SPECIFICALLY so the smoothing formula
-- lives in exactly one place (int_historical_delay_rates) and serving reproduces
-- training values byte-for-byte. Rewriting the arithmetic here would recreate
-- the duplication that design removed, and would silently move the pinned
-- headline. If you change anything here, re-run the golden-vector parity check.
--
-- GRAIN: (entity_level, entity_key). Levels and approximate cardinality:
--   route (7.5k) · carrier (17) · origin (374) · dest (375)
--   turnaround_band (5) · rotation_position (6)          => ~8.3k rows total.
--
-- `distance` is populated on `route` rows only (it folds in what _route_distance
-- used to query) and is NULL at every other level.
--
-- The band / rotation_position derivations mirror int_aircraft_rotation.sql,
-- exactly as _load_rotation_hist mirrored them in Python. Both are restricted to
-- `rotation_position is not null` — swap-shaped links carry NULL rotation
-- columns under the tail-swap restriction (CLAUDE.md §9) and must not define a
-- band. Entities absent here simply do not appear: serving leaves them NaN,
-- which IS the training NULL path for an unseen entity.
-- ============================================================================

with route_level as (

    select
        'route' as entity_level,
        route as entity_key,
        any_value(hist_route_arr_del15_rate) as hist_arr_del15_rate,
        any_value(hist_route_avg_arr_delay_minutes) as hist_avg_arr_delay_minutes,
        any_value(hist_route_n_flights) as hist_n_flights,
        -- min(), NOT any_value(): 85 of 7,539 routes carry TWO distinct
        -- distances (a 1-mile rounding split in the BTS source), and the
        -- any_value() this replaces returned an arbitrary one of them per
        -- call — so the serving distance for those routes changed run to run.
        -- min() is deterministic; the 1-mile difference is immaterial to a
        -- feature spanning ~30-5,000 miles. The hist_* columns above keep
        -- any_value() because they are provably constant within an entity.
        min(distance) as distance
    from {{ ref('ml_flight_features') }}
    group by entity_key

),

carrier_level as (

    select
        'carrier' as entity_level,
        carrier as entity_key,
        any_value(hist_carrier_arr_del15_rate),
        any_value(hist_carrier_avg_arr_delay_minutes),
        any_value(hist_carrier_n_flights),
        cast(null as float64)
    from {{ ref('ml_flight_features') }}
    group by entity_key

),

origin_level as (

    select
        'origin' as entity_level,
        origin as entity_key,
        any_value(hist_origin_arr_del15_rate),
        any_value(hist_origin_avg_arr_delay_minutes),
        any_value(hist_origin_n_flights),
        cast(null as float64)
    from {{ ref('ml_flight_features') }}
    group by entity_key

),

dest_level as (

    select
        'dest' as entity_level,
        dest as entity_key,
        any_value(hist_dest_arr_del15_rate),
        any_value(hist_dest_avg_arr_delay_minutes),
        any_value(hist_dest_n_flights),
        cast(null as float64)
    from {{ ref('ml_flight_features') }}
    group by entity_key

),

turnaround_band_level as (

    -- mirrors int_aircraft_rotation.sql's band CASE (and the Python
    -- _turnaround_band the request path still uses to derive a request's key)
    select
        'turnaround_band' as entity_level,
        case
            when not has_inbound_leg then 'no_inbound'
            when sched_turnaround_min < 35 then 'lt_35'
            when sched_turnaround_min < 60 then '35_60'
            when sched_turnaround_min < 120 then '60_120'
            else 'ge_120'
        end as entity_key,
        any_value(hist_turnaround_band_arr_del15_rate),
        any_value(hist_turnaround_band_avg_arr_delay_minutes),
        any_value(hist_turnaround_band_n_flights),
        cast(null as float64)
    from {{ ref('ml_flight_features') }}
    where rotation_position is not null
    group by entity_key

),

rotation_position_level as (

    select
        'rotation_position' as entity_level,
        cast(least(rotation_position, 6) as string) as entity_key,
        any_value(hist_rotation_position_arr_del15_rate),
        any_value(hist_rotation_position_avg_arr_delay_minutes),
        any_value(hist_rotation_position_n_flights),
        cast(null as float64)
    from {{ ref('ml_flight_features') }}
    where rotation_position is not null
    group by entity_key

)

select * from route_level
union all select * from carrier_level
union all select * from origin_level
union all select * from dest_level
union all select * from turnaround_band_level
union all select * from rotation_position_level
