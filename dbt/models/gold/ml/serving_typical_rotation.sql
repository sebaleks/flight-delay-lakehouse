{{ config(materialized='table') }}

-- ============================================================================
-- SERVING LOOKUP — the "typical rotation profile", exactly one row.
--
-- The training MEDIANS of the rotation schedule attributes, used when a caller
-- supplies no rotation context, plus the last-resort global density median.
-- This is _load_rotation_hist's median query, materialized.
--
-- WHY MEDIANS AND NOT NULL (the reasoning this model inherits, unchanged):
-- completed flights carry tails, so the mart has essentially no tail-unknown
-- rows and NaN in these columns sits OUTSIDE the training distribution —
-- empirically it produces garbage scores. Worse, under the tail-swap
-- restriction an all-NULL rotation is in-distribution but MEANS "the operated
-- linkage was swap-restructured"; nulling a merely-unknown FUTURE plan would
-- misclassify it as swap-shaped. Unknown-but-knowable schedule facts are
-- therefore estimated with training medians and the response flags the
-- estimate as `rotation_context: "typical_estimate"`.
--
-- EXACT MEDIANS, DELIBERATELY — this is a behaviour FIX, not a port.
-- The Python this replaces used approx_quantiles(x, 2)[offset(1)], which is
-- APPROXIMATE and whose result depends on how BigQuery shards the scan. The
-- same query was measured returning FOUR different answers on identical data
-- (inbound_distance 666 / 674 / 663 / 651 against an exact 667;
-- sched_turnaround 63 against an exact 64), and
-- because it ran at PROCESS STARTUP, every restart could serve a different
-- typical profile — i.e. the same context-less request could score differently
-- across deploys. percentile_disc is exact and deterministic, so the profile is
-- now pinned by the data alone. Costs ~35s once per dbt build instead of an
-- approximation per startup. Do not "optimize" this back to approx_quantiles.
--
-- `where has_inbound_leg and is_training_row` on the attribute medians, and
-- `where is_training_row` on the density median: TRAINING-window only, so the
-- fallback sits inside the distribution the models were fit on and never uses
-- the test window (docs/leakage_discipline.md rule 12).
--
-- Serving raises if any value here is NULL — an empty or rotation-less mart
-- must fail startup loudly, never score every request on the NaN path.
-- ============================================================================

with schedule_hours as (

    -- distinct schedule-hours, not flight rows — see serving_density_profile
    select distinct origin, flight_date, crs_dep_hour, origin_dep_density_hour as d
    from {{ ref('ml_flight_features') }}
    where is_training_row

),

attributes as (

    select distinct
        percentile_disc(rotation_position, 0.5) over () as typical_rotation_position,
        percentile_disc(legs_today, 0.5) over () as typical_legs_today,
        percentile_disc(sched_turnaround_min, 0.5) over () as typical_sched_turnaround_min,
        percentile_disc(inbound_distance, 0.5) over () as typical_inbound_distance,
        percentile_disc(inbound_crs_elapsed_min, 0.5) over () as typical_inbound_crs_elapsed_min
    from {{ ref('ml_flight_features') }}
    where has_inbound_leg and is_training_row

),

density as (

    select distinct percentile_disc(d, 0.5) over () as typical_density
    from schedule_hours

)

select * from attributes cross join density
