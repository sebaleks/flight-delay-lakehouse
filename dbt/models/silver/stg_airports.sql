{{ config(materialized='table') }}

-- US airport reference (incl. territories), one row per IATA code, from the
-- airports dbt SEED via ref() — never a bronze source (dbt/seeds/README.md).
-- Seed types are already inferred by dbt (floats/ints); this model conforms
-- keys and empty strings.

select
    upper(trim(iata)) as iata,
    nullif(trim(icao), '') as icao,
    name,
    city,
    country,
    latitude,
    longitude,
    elevation_ft,
    nullif(trim(tz), '') as tz
from {{ ref('airports') }}
