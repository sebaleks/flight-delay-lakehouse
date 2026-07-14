{{ config(materialized='view') }}

-- Looker Studio source: route drill-down (secondary page). One row per
-- directed route with origin/dest names for readable tables and filter
-- controls. Full period; rates safe at this grain.

select
    routes.route,
    routes.origin_airport_key,
    origin_airport.airport_name as origin_airport_name,
    origin_airport.city as origin_city,
    routes.dest_airport_key,
    dest_airport.airport_name as dest_airport_name,
    dest_airport.city as dest_city,
    routes.n_flight_legs,
    routes.n_arr_del15,
    routes.arr_del15_rate,
    1 - routes.arr_del15_rate as on_time_rate,
    routes.avg_arr_delay_minutes,
    routes.p90_arr_delay_minutes,
    routes.n_cancelled,
    routes.cancellation_rate,
    routes.n_diverted,
    routes.diversion_rate,
    routes.hist_arr_del15_rate
from {{ ref('mart_delays_by_route') }} as routes
left join {{ ref('dim_airport') }} as origin_airport
    on routes.origin_airport_key = origin_airport.airport_key
left join {{ ref('dim_airport') }} as dest_airport
    on routes.dest_airport_key = dest_airport.airport_key
