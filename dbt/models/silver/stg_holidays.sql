{{ config(materialized='table') }}

-- US federal holiday calendar 2022-2024, one row per calendar date, from the
-- holidays dbt SEED via ref() — never a bronze source (dbt/seeds/README.md).
-- Adjacency flags were generated against a holiday set padded one year each
-- side, so they are correct at the range edges.

select
    cast(date_day as date) as date_day,
    cast(is_holiday as bool) as is_holiday,
    nullif(trim(holiday_name), '') as holiday_name,
    cast(is_day_before_holiday as bool) as is_day_before_holiday,
    cast(is_day_after_holiday as bool) as is_day_after_holiday
from {{ ref('holidays') }}
