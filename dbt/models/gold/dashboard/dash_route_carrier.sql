{{ config(materialized='view') }}

-- Route x airline, additive. Thin skin over mart_delays_by_route_carrier so
-- the route drill-down can break a route down by airline without a second
-- aggregation. Rates are computed downstream as SUM/SUM.

select
    rc.route,
    rc.origin_airport_key,
    rc.dest_airport_key,
    rc.carrier_key,
    carrier.dot_id,
    rc.n_flights,
    rc.n_with_arr_outcome,
    rc.n_with_dep_outcome,
    rc.n_arr_del15,
    rc.n_cancelled,
    rc.n_diverted,
    rc.sum_arr_delay_minutes,
    rc.sum_dep_delay_minutes
from {{ ref('mart_delays_by_route_carrier') }} as rc
left join {{ ref('dim_carrier') }} as carrier
    on rc.carrier_key = carrier.carrier_key
