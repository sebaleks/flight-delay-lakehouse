{{ config(materialized='view') }}

-- Looker Studio source: airport reliability ranking (lead page). One row per
-- origin airport, full period. Rates here are safe to display as-is at this
-- grain; the spec filters to n_flight_legs >= 10,000 so tiny airports don't
-- dominate "least reliable" rankings.

select
    airport_key,
    airport_name,
    city,
    tz,
    n_flight_legs,
    n_arr_del15,
    arr_del15_rate,
    1 - arr_del15_rate as on_time_rate,
    avg_arr_delay_minutes,
    p90_arr_delay_minutes,
    n_cancelled,
    cancellation_rate,
    n_diverted,
    diversion_rate,
    hist_arr_del15_rate,
    hist_n_flights
from {{ ref('mart_delays_by_airport') }}
