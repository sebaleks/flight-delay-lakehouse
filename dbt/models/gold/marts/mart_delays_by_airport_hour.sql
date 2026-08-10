{{ config(materialized='table') }}

-- Ops-capacity baseline mart: what a departure bank at an airport NORMALLY
-- looks like. Grain (origin_airport_key, day_of_week 1=Mon..7=Sun, scheduled
-- departure hour) — at most airports x 7 x 24 rows (~62k; hours with no
-- scheduled departures simply have no row). ADDITIVE counts and sums only —
-- no pre-divided rates, so any rollup (all Fridays, a whole day, a season)
-- computes rates as SUM/SUM and can never average averages. Carries BOTH
-- outcomes: the ops page frames a bank by its departures (dep_del15) while
-- the model's per-flight probability is P(arr_del15) — the baseline must
-- support either comparison without a second aggregation. This TABLE is the
-- single aggregation over fact_flights at this grain; the dashboard view
-- dash_airport_hour_baseline is a thin label-adding skin over it.

select
    origin_airport_key,
    day_of_week,
    crs_dep_hour,
    count(*) as n_flights,
    countif(arr_del15 is not null) as n_with_arr_outcome,
    countif(dep_del15 is not null) as n_with_dep_outcome,
    countif(arr_del15) as n_arr_del15,
    countif(dep_del15) as n_dep_del15,
    countif(cancelled) as n_cancelled,
    countif(diverted) as n_diverted,
    sum(arr_delay_minutes) as sum_arr_delay_minutes,
    sum(dep_delay_minutes) as sum_dep_delay_minutes
from {{ ref('fact_flights') }}
group by origin_airport_key, day_of_week, crs_dep_hour
