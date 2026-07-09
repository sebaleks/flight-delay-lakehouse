{{ config(materialized='table') }}

-- ============================================================================
-- LEAKAGE-CRITICAL SHARED MODEL — the single definition of historical delay
-- rates by route, carrier, and (origin) airport. Both the analytics marts and
-- the ML feature mart must ref() this model; nothing recomputes these rates.
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
        route,
        carrier,
        origin,
        arr_del15,
        arr_delay_minutes,
        dep_delay_minutes
    from {{ ref('stg_gold__flights') }}
    where
        flight_date < date('{{ var("train_test_cutoff_date") }}')
        and not cancelled
        and not diverted
        and arr_del15 is not null
        and arr_delay_minutes is not null

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

)

select * from route_rates
union all
select * from carrier_rates
union all
select * from airport_rates
