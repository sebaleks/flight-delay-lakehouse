{{ config(materialized='table') }}

-- ============================================================================
-- SERVING LOOKUP — the origin_dep_density_hour ESTIMATE, precomputed.
--
-- Serve-time estimate of origin_dep_density_hour for a caller who does not
-- supply one: the TRAINING-window median over DISTINCT SCHEDULE-HOURS for
-- (origin, hour, weekday). This is _density_estimates' inner query, materialized
-- once instead of run per request against a mart it could not prune.
--
-- TWO DETAILS THAT ARE LOAD-BEARING, both carried over verbatim:
--   1. `select distinct origin, flight_date, crs_dep_hour, origin_dep_density_hour`
--      BEFORE the median — the median is over schedule-HOURS, not flight ROWS.
--      A flight-row median would overweight busy banks (a 40-departure hour
--      contributes 40 times), biasing the estimate upward.
--   2. `where is_training_row` — rule 12 of docs/leakage_discipline.md: every
--      serve-time ESTIMATE is aggregated over the training window only, so the
--      test window can never be used to build a serving feature.
--
-- EXACT MEDIANS, DELIBERATELY: percentile_disc, not the approx_quantiles this
-- replaces. Same reasoning as serving_typical_rotation — approx_quantiles was
-- measured returning different answers for identical data, which made a
-- context-less prediction depend on which process served it. See that model's
-- header for the measurements.
--
-- GRAIN: (origin, crs_dep_hour, day_of_week), <= 374 x 24 x 7 = 62,832 rows.
-- Misses (an origin/hour/weekday absent here, or an unknown airport) fall back
-- to the global training median in serving_typical_rotation — always an
-- in-distribution value, never NaN.
-- ============================================================================

with schedule_hours as (

    select distinct
        origin,
        flight_date,
        cast(crs_dep_hour as int64) as crs_dep_hour,
        cast(day_of_week as int64) as day_of_week,
        origin_dep_density_hour as density
    from {{ ref('ml_flight_features') }}
    where is_training_row

)

select distinct
    origin,
    crs_dep_hour,
    day_of_week,
    percentile_disc(density, 0.5) over (
        partition by origin, crs_dep_hour, day_of_week
    ) as density_median
from schedule_hours
