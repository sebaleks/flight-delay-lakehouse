{{ config(materialized='view') }}

-- Ops-capacity page source: the historical shape of each departure bank at an
-- airport — "ORD normally schedules ~48 departures in the Friday 18:00 hour
-- and ~11 of them arrive late". THIN view over mart_delays_by_airport_hour
-- (a materialized TABLE, TRAINING WINDOW ONLY — the comparator must predate
-- the holdout it is judged against; see the mart's header): adds airport
-- display labels and the weekday name, no aggregation, never touches
-- fact_flights. Additive counts pass through 1:1; consumers compute rates as
-- SUM/SUM — never averaged rate columns.

select
    base.origin_airport_key as airport_key,
    dim_airport.airport_name,
    dim_airport.city,
    dim_airport.tz,
    base.day_of_week,  -- BTS convention: 1 = Monday .. 7 = Sunday
    case base.day_of_week
        when 1 then 'Mon' when 2 then 'Tue' when 3 then 'Wed'
        when 4 then 'Thu' when 5 then 'Fri' when 6 then 'Sat'
        when 7 then 'Sun'
    end as day_name,
    base.crs_dep_hour as dep_hour,
    base.n_flights,
    base.n_with_arr_outcome,
    base.n_with_dep_outcome,
    base.n_arr_del15,
    base.n_dep_del15,
    base.n_cancelled,
    base.n_diverted,
    base.sum_arr_delay_minutes,
    base.sum_dep_delay_minutes
from {{ ref('mart_delays_by_airport_hour') }} as base
left join {{ ref('dim_airport') }} as dim_airport
    on base.origin_airport_key = dim_airport.airport_key
