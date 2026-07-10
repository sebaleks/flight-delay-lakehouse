{{ config(materialized='table') }}

-- Monthly mart (one row per calendar month, true DATE month_start for time
-- series and YoY): ADDITIVE counts and sums only — rates are computed
-- downstream as SUM/SUM. Single aggregation over fact_flights for the trend;
-- dash_monthly_trend is a thin skin over this TABLE.

select
    date_trunc(date_key, month) as month_start,
    extract(year from date_key) as year,
    extract(month from date_key) as month,
    count(*) as n_flights,
    countif(arr_del15 is not null) as n_with_arr_outcome,
    countif(arr_del15) as n_arr_del15,
    countif(cancelled) as n_cancelled,
    countif(diverted) as n_diverted,
    sum(arr_delay_minutes) as sum_arr_delay_minutes,
    sum(dep_delay_minutes) as sum_dep_delay_minutes
from {{ ref('fact_flights') }}
group by month_start, year, month
