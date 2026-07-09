-- Every airport that BTS flights reference must carry an IANA timezone in the
-- airports seed — local-time features depend on it, and a missing tz must
-- fail HERE, loudly, instead of surfacing as NULL origin_tz/dest_tz rows
-- downstream (BIH did exactly that before its backfill).
--
-- Deliberately scoped to FLOWN airports: 118 non-BTS general-aviation rows in
-- the seed ship without a tz from OpenFlights, are referenced by nothing, and
-- the seed is kept byte-identical to upstream + curated backfills. A new
-- airport, territory, or source regression that loses a flown airport's tz
-- returns rows below (one per offending IATA code) and fails the build.
-- (An airport missing from the seed entirely is caught by the existing
-- relationships tests on stg_bts_flights.origin/dest.)

with flown as (

    select distinct origin as iata from {{ ref('stg_bts_flights') }}
    union distinct
    select distinct dest from {{ ref('stg_bts_flights') }}

)

select airports.iata
from {{ ref('airports') }} as airports
inner join flown
    on upper(trim(airports.iata)) = flown.iata
where airports.tz is null or trim(airports.tz) = ''
