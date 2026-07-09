{{ config(materialized='table') }}

-- Dashboard mart: full-period delay/cancellation aggregates per directed
-- route (origin -> dest), with the SHARED training-window historical rates
-- joined (never recomputed).

select
    fact.route,
    fact.origin_airport_key,
    fact.dest_airport_key,
    {{ delay_measures() }},
    hist.arr_del15_rate as hist_arr_del15_rate,
    hist.avg_arr_delay_minutes as hist_avg_arr_delay_minutes,
    hist.n_flights as hist_n_flights
from {{ ref('fact_flights') }} as fact
left join {{ ref('int_historical_delay_rates') }} as hist
    on hist.entity_level = 'route' and hist.entity_key = fact.route
group by route, origin_airport_key, dest_airport_key,
    hist.arr_del15_rate, hist.avg_arr_delay_minutes, hist.n_flights
