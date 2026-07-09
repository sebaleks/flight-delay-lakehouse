{{ config(materialized='table') }}

-- Date dimension: one row per calendar date spanning the flight data, with
-- holiday flags from the seed-backed silver calendar.
-- day_of_week uses the BTS convention (1 = Monday .. 7 = Sunday) so it lines
-- up with fact/staging columns.

with bounds as (

    select min(flight_date) as min_date, max(flight_date) as max_date
    from {{ ref('stg_gold__flights') }}

),

spine as (

    select date_day
    from bounds, unnest(generate_date_array(min_date, max_date)) as date_day

)

select
    spine.date_day as date_key,
    extract(year from spine.date_day) as year,
    extract(quarter from spine.date_day) as quarter,
    extract(month from spine.date_day) as month,
    format_date('%B', spine.date_day) as month_name,
    extract(day from spine.date_day) as day_of_month,
    cast(format_date('%u', spine.date_day) as int64) as day_of_week,
    format_date('%A', spine.date_day) as day_name,
    cast(format_date('%u', spine.date_day) as int64) >= 6 as is_weekend,
    coalesce(holidays.is_holiday, false) as is_holiday,
    holidays.holiday_name,
    coalesce(holidays.is_day_before_holiday, false) as is_day_before_holiday,
    coalesce(holidays.is_day_after_holiday, false) as is_day_after_holiday
from spine
left join {{ ref('stg_holidays') }} as holidays
    on spine.date_day = holidays.date_day
