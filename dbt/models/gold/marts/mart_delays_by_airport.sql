{{ config(materialized='table') }}

-- Dashboard mart: full-period (2022-2024) delay/cancellation aggregates per
-- origin airport. The hist_* columns come from the SHARED training-window
-- rate model (never recomputed here) and are labeled as such: they cover only
-- flights before the train/test cutoff, unlike the full-period measures.

select
    fact.origin_airport_key as airport_key,
    dim_airport.airport_name,
    dim_airport.city,
    dim_airport.tz,
    {{ delay_measures() }},
    hist.arr_del15_rate as hist_arr_del15_rate,
    hist.avg_arr_delay_minutes as hist_avg_arr_delay_minutes,
    hist.n_flights as hist_n_flights
from {{ ref('fact_flights') }} as fact
left join {{ ref('dim_airport') }} as dim_airport
    on fact.origin_airport_key = dim_airport.airport_key
left join {{ ref('int_historical_delay_rates') }} as hist
    on hist.entity_level = 'airport' and hist.entity_key = fact.origin_airport_key
group by airport_key, airport_name, city, tz,
    hist.arr_del15_rate, hist.avg_arr_delay_minutes, hist.n_flights
