{{ config(materialized='view') }}

-- Looker Studio source: when delays happen — hour of day, day of week, month,
-- season. THIN view over mart_delays_by_schedule (a materialized TABLE): adds
-- display labels only, no aggregation, and never touches fact_flights — every
-- Looker interaction scans ~6k pre-aggregated rows. Additive counts pass
-- through 1:1; rates are calculated fields in Looker
-- (SUM(n_arr_del15)/SUM(n_with_arr_outcome)) — never averaged rate columns.

select
    year,
    month,
    format_date('%b', date(year, month, 1)) as month_name,
    case
        when month in (12, 1, 2) then 'Winter'
        when month between 3 and 5 then 'Spring'
        when month between 6 and 8 then 'Summer'
        else 'Fall'
    end as season,
    case
        when month in (12, 1, 2) then 1
        when month between 3 and 5 then 2
        when month between 6 and 8 then 3
        else 4
    end as season_order,
    day_of_week,  -- BTS convention: 1 = Monday .. 7 = Sunday
    case day_of_week
        when 1 then 'Mon' when 2 then 'Tue' when 3 then 'Wed'
        when 4 then 'Thu' when 5 then 'Fri' when 6 then 'Sat'
        when 7 then 'Sun'
    end as day_name,
    crs_dep_hour as dep_hour,
    n_flights,
    n_with_arr_outcome,
    n_arr_del15,
    n_cancelled,
    n_diverted,
    sum_arr_delay_minutes,
    sum_dep_delay_minutes
from {{ ref('mart_delays_by_schedule') }}
