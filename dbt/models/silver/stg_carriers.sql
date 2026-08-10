{{ config(materialized='table') }}

-- Carrier display names, from the carriers dbt SEED via ref() — never a bronze
-- source (dbt/seeds/README.md). dim_carrier's own header called for exactly
-- this: "BTS ships no carrier display names in the on-time files; add the
-- lookup as a seed later if the dashboard needs names." It does — a dropdown
-- reading "MQ" tells a traveller nothing.
--
-- is_regional matters for reading the numbers honestly: MQ, OH, OO, QX, YV, YX
-- and 9E are REGIONAL carriers that operate under mainline brands (American
-- Eagle, Delta Connection, United Express). A passenger books "American" and
-- flies Envoy, so a per-carrier delay ranking splits one booking experience
-- across several rows.

select
    upper(trim(carrier_key)) as carrier_key,
    trim(carrier_name) as carrier_name,
    is_regional
from {{ ref('carriers') }}
