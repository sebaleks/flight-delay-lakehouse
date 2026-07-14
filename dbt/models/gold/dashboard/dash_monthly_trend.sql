{{ config(materialized='view') }}

-- Looker Studio source: monthly trend + year-over-year. THIN view over
-- mart_delays_monthly (a materialized TABLE, 36 rows): month_start is a true
-- DATE for time series and Looker's YoY comparison; year + month support an
-- explicit YoY overlay. Additive counts pass through 1:1; rates are Looker
-- calculated fields (SUM/SUM) — no pre-divided rate columns.

select
    month_start,
    year,
    month,
    format_date('%b', month_start) as month_name,
    n_flights,
    n_with_arr_outcome,
    n_with_dep_outcome,
    n_arr_del15,
    n_cancelled,
    n_diverted,
    sum_arr_delay_minutes,
    sum_dep_delay_minutes
from {{ ref('mart_delays_monthly') }}
