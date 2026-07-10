{{ config(materialized='view') }}

-- Looker Studio source: carrier reliability ranking (lead page). One row per
-- reporting carrier, full period.

select
    carrier_key,
    dot_id,
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
from {{ ref('mart_delays_by_carrier') }}
