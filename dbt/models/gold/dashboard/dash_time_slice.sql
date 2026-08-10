{{ config(materialized='view') }}

-- Time-of-travel counts sliceable by origin airport and airline.
--
-- Unlike every other dash_* view this is NOT read whole: it is ~3.6M rows and
-- the dashboard queries it WITH a WHERE on airport and/or carrier. The base
-- mart is clustered on exactly those two columns, and a view is inlined, so
-- the predicate still prunes to a few MB.
--
-- ADDITIVE ONLY — labels are added here, rates are computed downstream as
-- SUM/SUM in dashboard/metrics.py.

select
    ts.origin_airport_key,
    origin_airport.airport_name as origin_airport_name,
    ts.carrier_key,
    ts.year,
    ts.month,
    format_date('%b', date(ts.year, ts.month, 1)) as month_name,
    ts.day_of_week,
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][
        ordinal(ts.day_of_week)
    ] as day_name,
    ts.crs_dep_hour as dep_hour,
    ts.n_flights,
    ts.n_with_arr_outcome,
    ts.n_with_dep_outcome,
    ts.n_arr_del15,
    ts.n_cancelled,
    ts.n_diverted,
    ts.sum_arr_delay_minutes,
    ts.sum_dep_delay_minutes
from {{ ref('mart_delays_by_time_slice') }} as ts
left join {{ ref('dim_airport') }} as origin_airport
    on ts.origin_airport_key = origin_airport.airport_key
