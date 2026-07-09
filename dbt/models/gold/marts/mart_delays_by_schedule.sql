{{ config(materialized='table') }}

-- Dashboard mart: delay/cancellation aggregates on the schedule grain
-- (year, month, day_of_week 1=Mon..7=Sun, scheduled departure hour 0-23).
-- One mart serves the hour-of-day, day-of-week, and month cuts: n_* counts
-- are included so any single-dimension rollup can reweight correctly.

select
    extract(year from date_key) as year,
    extract(month from date_key) as month,
    day_of_week,
    crs_dep_hour,
    {{ delay_measures() }}
from {{ ref('fact_flights') }}
group by year, month, day_of_week, crs_dep_hour
