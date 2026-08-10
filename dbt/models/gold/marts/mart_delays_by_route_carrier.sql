{{ config(materialized='table') }}

-- Route x carrier grain: who flies this route, and how reliably.
--
-- The route drill-down could previously only say "ORD-LAX runs 22% late"; it
-- could not say WHICH airline is dragging that number, which is the first
-- question anyone asks of a route. 16,463 rows — small enough for the
-- dashboard's read-whole-view-and-cache pattern.
--
-- ADDITIVE COUNTS AND SUMS ONLY, deliberately NOT the delay_measures() macro.
-- The page rolls these up across whatever routes the filters selected, so it
-- needs denominators: a rate per route-carrier cannot be averaged across
-- routes without weighting, and pre-divided rates are exactly how that goes
-- wrong. dashboard/metrics.py computes every rate as SUM(num)/SUM(den).

select
    route,
    origin_airport_key,
    dest_airport_key,
    carrier_key,
    count(*) as n_flights,
    countif(arr_del15 is not null) as n_with_arr_outcome,
    countif(dep_delay_minutes is not null) as n_with_dep_outcome,
    countif(arr_del15) as n_arr_del15,
    countif(cancelled) as n_cancelled,
    countif(diverted) as n_diverted,
    sum(arr_delay_minutes) as sum_arr_delay_minutes,
    sum(dep_delay_minutes) as sum_dep_delay_minutes
from {{ ref('fact_flights') }}
group by route, origin_airport_key, dest_airport_key, carrier_key
