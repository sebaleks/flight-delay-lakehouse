{{ config(materialized='table') }}

-- Schedule-grain mart (year, month, day_of_week 1=Mon..7=Sun, scheduled
-- departure hour): ADDITIVE counts and sums only — no pre-divided rates, so
-- any rollup (hour only, season only, ...) computes rates as SUM/SUM and can
-- never average averages. This TABLE is the single aggregation over
-- fact_flights for the time cuts; the dashboard view dash_delays_by_time is
-- a thin label-adding skin over it.

select
    extract(year from date_key) as year,
    extract(month from date_key) as month,
    day_of_week,
    crs_dep_hour,
    count(*) as n_flights,
    countif(arr_del15 is not null) as n_with_arr_outcome,
    countif(arr_del15) as n_arr_del15,
    countif(cancelled) as n_cancelled,
    countif(diverted) as n_diverted,
    sum(arr_delay_minutes) as sum_arr_delay_minutes,
    sum(dep_delay_minutes) as sum_dep_delay_minutes
from {{ ref('fact_flights') }}
group by year, month, day_of_week, crs_dep_hour
