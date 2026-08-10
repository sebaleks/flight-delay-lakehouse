{{ config(materialized='table') }}

-- Carrier dimension: one row per reporting carrier code observed in the data,
-- with its DOT id, IATA code and DISPLAY NAME. BTS ships no carrier names in
-- the on-time files, so the name comes from the carriers seed (stg_carriers).
--
-- LEFT join, deliberately: a carrier appearing in the data but missing from the
-- seed must still get a dimension row (and show as its bare code) rather than
-- vanish from the star. The relationships tests on fact_flights would catch the
-- disappearance, but silently dropping a carrier is the worse failure.

select
    flights.carrier as carrier_key,
    carriers.carrier_name,
    coalesce(carriers.is_regional, false) as is_regional,
    max(flights.dot_id_reporting_airline) as dot_id,
    max(flights.iata_code_reporting_airline) as iata_code,
    min(flights.flight_date) as first_flight_date,
    max(flights.flight_date) as last_flight_date,
    count(*) as n_flight_legs
from {{ ref('stg_gold__flights') }} as flights
left join {{ ref('stg_carriers') }} as carriers
    on flights.carrier = carriers.carrier_key
group by carrier_key, carriers.carrier_name, carriers.is_regional
