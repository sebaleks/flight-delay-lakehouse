{{
    config(
        materialized='table',
        cluster_by=['origin_airport_key', 'carrier_key'],
    )
}}

-- Time-of-travel counts, sliceable by origin airport AND carrier.
--
-- WHY A SEPARATE MART, AND WHY IT IS QUERIED DIFFERENTLY. mart_delays_by_schedule
-- answers "when do delays happen" for the whole network in ~6k rows, which the
-- dashboard reads whole and caches. Adding origin and carrier to that grain
-- takes it to ~3.6M rows — far too big to SELECT * into pandas, and the
-- dashboard's <1 MB-per-interaction rule would be broken on every page load.
--
-- So this one is CLUSTERED and read with a WHERE instead: pick an airport
-- and/or an airline and BigQuery prunes to those blocks, scanning a few MB
-- rather than the table. The unfiltered page keeps using the small cached view
-- and never touches this at all, so the default page load costs nothing extra.
--
-- ADDITIVE COUNTS AND SUMS ONLY, per the binding rule: no pre-divided rates, so
-- any rollup (hour only, month only, airport across all carriers) computes
-- rates as SUM/SUM in dashboard/metrics.py and can never average an average.
--
-- year is kept so the year-over-year chart works under a filter too; without it
-- a filtered view would silently lose a chart the unfiltered view has.

select
    origin_airport_key,
    carrier_key,
    extract(year from date_key) as year,
    extract(month from date_key) as month,
    day_of_week,
    crs_dep_hour,
    count(*) as n_flights,
    countif(arr_del15 is not null) as n_with_arr_outcome,
    countif(dep_delay_minutes is not null) as n_with_dep_outcome,
    countif(arr_del15) as n_arr_del15,
    countif(cancelled) as n_cancelled,
    countif(diverted) as n_diverted,
    sum(arr_delay_minutes) as sum_arr_delay_minutes,
    sum(dep_delay_minutes) as sum_dep_delay_minutes
from {{ ref('fact_flights') }}
group by origin_airport_key, carrier_key, year, month, day_of_week, crs_dep_hour
