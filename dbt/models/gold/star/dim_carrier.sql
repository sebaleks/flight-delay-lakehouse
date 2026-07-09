{{ config(materialized='table') }}

-- Carrier dimension: one row per reporting carrier code observed in the data,
-- with its DOT id and IATA code. BTS ships no carrier display names in the
-- on-time files; add the L_UNIQUE_CARRIERS lookup as a seed later if the
-- dashboard needs names.

select
    carrier as carrier_key,
    max(dot_id_reporting_airline) as dot_id,
    max(iata_code_reporting_airline) as iata_code,
    min(flight_date) as first_flight_date,
    max(flight_date) as last_flight_date,
    count(*) as n_flight_legs
from {{ ref('stg_gold__flights') }}
group by carrier
