{{ config(materialized='view') }}

-- Carrier reliability ranking. One row per reporting carrier, full period.
--
-- carrier_name / is_regional come from dim_carrier (seeded, see stg_carriers):
-- a dropdown reading "MQ" tells a traveller nothing. is_regional is surfaced
-- because a passenger books "American" and flies Envoy — the per-carrier
-- ranking splits one booking experience across several rows, and the page
-- should be able to say so.

select
    m.carrier_key,
    c.carrier_name,
    c.is_regional,
    m.dot_id,
    m.n_flight_legs,
    m.n_arr_del15,
    m.arr_del15_rate,
    1 - m.arr_del15_rate as on_time_rate,
    m.avg_arr_delay_minutes,
    m.p90_arr_delay_minutes,
    m.n_cancelled,
    m.cancellation_rate,
    m.n_diverted,
    m.diversion_rate,
    m.hist_arr_del15_rate,
    m.hist_n_flights
from {{ ref('mart_delays_by_carrier') }} as m
left join {{ ref('dim_carrier') }} as c
    on m.carrier_key = c.carrier_key
