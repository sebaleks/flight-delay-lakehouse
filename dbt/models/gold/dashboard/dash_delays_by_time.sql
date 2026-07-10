{{ config(materialized='view') }}

-- Looker Studio source: when delays happen — hour of day, day of week, month,
-- season. Grain: (year, month, day_of_week, dep_hour), ~6k rows. Carries
-- ADDITIVE counts and sums so any rollup (hour only, season only, ...) stays
-- correct via calculated fields (SUM(n_arr_del15)/SUM(n_with_arr_outcome));
-- never average the rate columns of pre-aggregated rows.

select
    extract(year from date_key) as year,
    extract(month from date_key) as month,
    format_date('%b', date_key) as month_name,
    case
        when extract(month from date_key) in (12, 1, 2) then 'Winter'
        when extract(month from date_key) between 3 and 5 then 'Spring'
        when extract(month from date_key) between 6 and 8 then 'Summer'
        else 'Fall'
    end as season,
    case
        when extract(month from date_key) in (12, 1, 2) then 1
        when extract(month from date_key) between 3 and 5 then 2
        when extract(month from date_key) between 6 and 8 then 3
        else 4
    end as season_order,
    day_of_week,  -- BTS convention: 1 = Monday .. 7 = Sunday
    case day_of_week
        when 1 then 'Mon' when 2 then 'Tue' when 3 then 'Wed'
        when 4 then 'Thu' when 5 then 'Fri' when 6 then 'Sat'
        when 7 then 'Sun'
    end as day_name,
    crs_dep_hour as dep_hour,
    count(*) as n_flights,
    countif(arr_del15 is not null) as n_with_arr_outcome,
    countif(arr_del15) as n_arr_del15,
    countif(cancelled) as n_cancelled,
    countif(diverted) as n_diverted,
    sum(arr_delay_minutes) as sum_arr_delay_minutes,
    sum(dep_delay_minutes) as sum_dep_delay_minutes
from {{ ref('fact_flights') }}
group by year, month, month_name, season, season_order, day_of_week, day_name, dep_hour
