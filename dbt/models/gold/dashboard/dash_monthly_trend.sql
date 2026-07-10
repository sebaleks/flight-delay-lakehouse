{{ config(materialized='view') }}

-- Looker Studio source: monthly trend + year-over-year. One row per calendar
-- month (36 rows). month_start is a true DATE for time-series charts and
-- Looker's date-comparison (YoY) feature; year + month support an explicit
-- YoY overlay (month on the x-axis, year as series). Additive counts/sums —
-- rates via calculated fields when rolling up.

select
    date_trunc(date_key, month) as month_start,
    extract(year from date_key) as year,
    extract(month from date_key) as month,
    format_date('%b', date_key) as month_name,
    count(*) as n_flights,
    countif(arr_del15 is not null) as n_with_arr_outcome,
    countif(arr_del15) as n_arr_del15,
    countif(cancelled) as n_cancelled,
    countif(diverted) as n_diverted,
    sum(arr_delay_minutes) as sum_arr_delay_minutes,
    round(countif(arr_del15) / nullif(countif(arr_del15 is not null), 0), 4) as arr_del15_rate,
    round(countif(cancelled) / count(*), 4) as cancellation_rate,
    round(avg(arr_delay_minutes), 2) as avg_arr_delay_minutes
from {{ ref('fact_flights') }}
group by month_start, year, month, month_name
