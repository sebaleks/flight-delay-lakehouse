{{ config(materialized='table') }}

-- ============================================================================
-- LEAKAGE-CRITICAL SHARED MODEL — the single definition of historical delay
-- rates by route, carrier, (origin) airport, and — since the cascade work —
-- scheduled-turnaround band and rotation position (the leak-free substitute
-- for actual inbound delay: the TRAINING-WINDOW tendency of a rotation
-- profile to run late, never the realized delay of this flight's inbound
-- leg). Both the analytics marts and the ML feature mart must ref() this
-- model; nothing recomputes these rates.
--
-- Every row is computed ONLY from flights strictly BEFORE the train/test
-- cutoff (var 'train_test_cutoff_date' in dbt_project.yml — the one and only
-- place the boundary is defined). Flights on/after the cutoff belong to the
-- ML test window: including them here would leak future outcomes backward
-- into training features (CLAUDE.md §9). The WHERE below is the ONLY date
-- filter in this model.
--
-- Rates are over completed flights (not cancelled, not diverted, labels
-- present). n_flights is kept so consumers can regularize sparse entities;
-- entities that first appear in the test window are simply absent — consumers
-- LEFT JOIN and keep the NULL rather than inventing a rate.
-- ============================================================================

with training_flights as (

    select
        flights.route,
        flights.carrier,
        flights.origin,
        flights.arr_del15,
        flights.arr_delay_minutes,
        flights.dep_delay_minutes,
        -- rotation attributes from the shared schedule-only chain (one
        -- definition, int_aircraft_rotation) — band keys, never NULL
        rotation.turnaround_band,
        rotation.rotation_position_key
    from {{ ref('stg_gold__flights') }} as flights
    left join {{ ref('int_aircraft_rotation') }} as rotation
        on rotation.flight_date = flights.flight_date
        and rotation.carrier = flights.carrier
        -- null-safe: flight_number is the one nullable natural-key column
        and rotation.flight_number is not distinct from flights.flight_number
        and rotation.origin = flights.origin
        and rotation.dest = flights.dest
        and rotation.crs_dep_time = flights.crs_dep_time
    where
        flights.flight_date < date('{{ var("train_test_cutoff_date") }}')
        and not flights.cancelled
        and not flights.diverted
        and flights.arr_del15 is not null
        and flights.arr_delay_minutes is not null

),

route_rates as (

    select
        'route' as entity_level,
        route as entity_key,
        count(*) as n_flights,
        countif(arr_del15) / count(*) as arr_del15_rate,
        avg(arr_delay_minutes) as avg_arr_delay_minutes,
        avg(dep_delay_minutes) as avg_dep_delay_minutes
    from training_flights
    group by route

),

carrier_rates as (

    select
        'carrier' as entity_level,
        carrier as entity_key,
        count(*) as n_flights,
        countif(arr_del15) / count(*) as arr_del15_rate,
        avg(arr_delay_minutes) as avg_arr_delay_minutes,
        avg(dep_delay_minutes) as avg_dep_delay_minutes
    from training_flights
    group by carrier

),

airport_rates as (

    -- airport level = flights DEPARTING the airport (origin grain)
    select
        'airport' as entity_level,
        origin as entity_key,
        count(*) as n_flights,
        countif(arr_del15) / count(*) as arr_del15_rate,
        avg(arr_delay_minutes) as avg_arr_delay_minutes,
        avg(dep_delay_minutes) as avg_dep_delay_minutes
    from training_flights
    group by origin

),

turnaround_band_rates as (

    -- the leak-free substitute for actual inbound delay: how often flights
    -- with THIS scheduled-turnaround profile ran late in the TRAINING window
    select
        'turnaround_band' as entity_level,
        turnaround_band as entity_key,
        count(*) as n_flights,
        countif(arr_del15) / count(*) as arr_del15_rate,
        avg(arr_delay_minutes) as avg_arr_delay_minutes,
        avg(dep_delay_minutes) as avg_dep_delay_minutes
    from training_flights
    where turnaround_band is not null
    group by turnaround_band

),

rotation_position_rates as (

    select
        'rotation_position' as entity_level,
        rotation_position_key as entity_key,
        count(*) as n_flights,
        countif(arr_del15) / count(*) as arr_del15_rate,
        avg(arr_delay_minutes) as avg_arr_delay_minutes,
        avg(dep_delay_minutes) as avg_dep_delay_minutes
    from training_flights
    where rotation_position_key is not null
    group by rotation_position_key

)

select * from route_rates
union all
select * from carrier_rates
union all
select * from airport_rates
union all
select * from turnaround_band_rates
union all
select * from rotation_position_rates
