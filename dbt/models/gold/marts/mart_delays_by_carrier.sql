{{ config(materialized='table') }}

-- Dashboard mart: full-period delay/cancellation aggregates per carrier, with
-- the SHARED training-window historical rates joined (never recomputed).

select
    fact.carrier_key,
    dim_carrier.dot_id,
    {{ delay_measures() }},
    hist.arr_del15_rate as hist_arr_del15_rate,
    hist.avg_arr_delay_minutes as hist_avg_arr_delay_minutes,
    hist.n_flights as hist_n_flights
from {{ ref('fact_flights') }} as fact
left join {{ ref('dim_carrier') }} as dim_carrier
    on fact.carrier_key = dim_carrier.carrier_key
left join {{ ref('int_historical_delay_rates') }} as hist
    on hist.entity_level = 'carrier' and hist.entity_key = fact.carrier_key
group by carrier_key, dot_id,
    hist.arr_del15_rate, hist.avg_arr_delay_minutes, hist.n_flights
