{{
    config(
        materialized='table',
        partition_by={'field': 'flight_date', 'data_type': 'date', 'granularity': 'month'},
        cluster_by=['origin', 'dest', 'carrier'],
    )
}}

-- ============================================================================
-- LEAKAGE-CRITICAL SHARED MODEL — per-leg aircraft-rotation attributes, from
-- SCHEDULE COLUMNS ONLY. The single definition of the rotation chain: the ML
-- mart and the historical rates model both ref() this; nothing recomputes it.
--
-- THE RULE (CLAUDE.md §9): the prior leg's SCHEDULED data is knowable at
-- booking and leak-free. The prior leg's ACTUAL data — arr_delay, dep_delay,
-- actual times, cancellation — is post-departure information for the flight
-- being predicted and must NEVER be selected here. This model reads
-- crs_dep_time / crs_arr_time / crs_elapsed_time / distance and NOTHING
-- realized. assert_ml_rotation_schedule_only recomputes the turnaround from
-- silver schedule columns independently and pins it value-level.
--
-- Timestamps: dep_ts_utc = local schedule + seed tz (the same construction
-- the mart's weather join uses). Scheduled arrival = dep_ts_utc +
-- crs_elapsed_time minutes — timezone-proof, and immune to the local-clock
-- midnight wrap (a red-eye's scheduled arrival lands on the next UTC day
-- automatically; no clock arithmetic).
--
-- RED-EYES / DAY BOUNDARIES, explicitly: the rotation CHAIN (prior leg,
-- turnaround) links by UTC timestamps ACROSS calendar days — a 23:50 red-eye
-- is the inbound of the next morning's first departure if the gap is within
-- the duty window. The day POSITION (rotation_position, legs_today) counts
-- within the BTS service date (flight_date), the schedule-publication
-- convention. Both choices documented here, tested downstream.
--
-- has_inbound_leg is TRUE only when the same tail has a prior leg whose
-- scheduled gap is in [0, 14h] AND whose destination equals this leg's
-- origin (station continuity — an unrecorded ferry/positioning move is not
-- a usable inbound). Beyond 14h is an overnight/duty break (the 'first leg
-- of the day' case), negative gaps are schedule-data quirks; all take the
-- no-inbound path. Tail-unknown legs (0.34%) take their own 'no_tail' band.
-- Bands are KEYS with their own training-window history, never silent NULLs.
--
-- Known accepted limitation (shared with the weather join, counted there as
-- 1 mart row): BTS '2400' scheduled times are stored as 00:00 of
-- flight_date, so a true end-of-day midnight departure sorts ~24h early in
-- the chain, perturbing its own and its neighbor's position/turnaround —
-- a handful of legs, never a leak (all inputs remain schedule columns).
--
-- EPISTEMOLOGY CAVEAT, stated honestly: BTS records the OPERATED tail,
-- post-hoc. The feature VALUES are all schedule columns (leak-free), but
-- the LINKAGE reflects the realized aircraft assignment — day-of tail swaps
-- (which correlate with disruption) shape which legs chain together. BTS
-- carries no planned-assignment field, so this is the best available
-- approximation; a production system would use the airline's planned
-- rotation feed. Second-order relative to the feature values themselves;
-- disclosed in the mart header and the model-card/PR.
-- ============================================================================

with legs as (

    select
        flight_date,
        carrier,
        flight_number,
        origin,
        dest,
        crs_dep_time,
        tail_number,
        distance,
        crs_elapsed_time,
        crs_dep_hour,
        timestamp(datetime(flight_date, crs_dep_time), origin_tz) as dep_ts_utc,
        -- computed HERE, before any tail filtering: congestion counts every
        -- scheduled leg (published schedule; cancellations unknowable at
        -- prediction time), tail-known or not
        count(*) over (partition by origin, flight_date, crs_dep_hour)
            as origin_dep_density_hour
    from {{ ref('stg_gold__flights') }}

),

chained as (

    select
        *,
        timestamp_add(dep_ts_utc, interval cast(crs_elapsed_time as int64) minute)
            as arr_ts_utc,
        -- prior leg of the SAME TAIL by scheduled departure — schedule
        -- columns only; ACTUAL outcome columns are never selected
        lag(timestamp_add(dep_ts_utc, interval cast(crs_elapsed_time as int64) minute))
            over w as prior_arr_ts_utc,
        lag(dest) over w as prior_dest,
        lag(distance) over w as inbound_distance,
        lag(crs_elapsed_time) over w as inbound_crs_elapsed_min,
        row_number() over (
            partition by tail_number, flight_date
            order by dep_ts_utc, carrier, flight_number, origin, dest
        ) as rotation_position,
        count(*) over (partition by tail_number, flight_date) as legs_today
    from legs
    where tail_number is not null
    window w as (
        partition by tail_number
        order by dep_ts_utc, carrier, flight_number, origin, dest
    )

),

with_turnaround as (

    select
        *,
        case
            when prior_arr_ts_utc is null then null
            else timestamp_diff(dep_ts_utc, prior_arr_ts_utc, minute)
        end as raw_gap_min
    from chained

),

classified as (

    select
        * except (raw_gap_min),
        -- inbound counts only when (a) the gap is inside the duty window
        -- (negative gaps are schedule-data quirks, >14h is an overnight
        -- break) AND (b) the prior leg actually ARRIVES AT THIS ORIGIN —
        -- a station discontinuity (unrecorded ferry/positioning move, tail
        -- data anomaly) is not a usable inbound; both take the no-inbound
        -- path, fraction reported at build time
        raw_gap_min is not null
            and raw_gap_min between 0 and 840
            and prior_dest = origin as has_inbound_leg,
        case
            when raw_gap_min is not null
                and raw_gap_min between 0 and 840
                and prior_dest = origin
                then raw_gap_min
        end as sched_turnaround_min
    from with_turnaround

)

select
    flight_date,
    carrier,
    flight_number,
    origin,
    dest,
    crs_dep_time,
    rotation_position,
    legs_today,
    origin_dep_density_hour,
    has_inbound_leg,
    sched_turnaround_min,
    -- slack vs a typical 35-minute narrow-body minimum turnaround
    sched_turnaround_min - 35 as sched_turnaround_slack_min,
    coalesce(sched_turnaround_min < 35, false) as is_tight_turnaround,
    case when has_inbound_leg then inbound_distance end as inbound_distance,
    case when has_inbound_leg then inbound_crs_elapsed_min end as inbound_crs_elapsed_min,
    -- BAND KEYS for the shared historical rates (never NULL — first-leg and
    -- unknown-tail flights get their own training-window history)
    case
        when not has_inbound_leg then 'no_inbound'
        when sched_turnaround_min < 35 then 'lt_35'
        when sched_turnaround_min < 60 then '35_60'
        when sched_turnaround_min < 120 then '60_120'
        else 'ge_120'
    end as turnaround_band,
    cast(least(rotation_position, 6) as string) as rotation_position_key
from classified

union all

-- tail-unknown legs (0.34%): no chain, own band, position NULL
select
    flight_date,
    carrier,
    flight_number,
    origin,
    dest,
    crs_dep_time,
    cast(null as int64) as rotation_position,
    cast(null as int64) as legs_today,
    origin_dep_density_hour,
    false as has_inbound_leg,
    null as sched_turnaround_min,
    null as sched_turnaround_slack_min,
    false as is_tight_turnaround,
    null as inbound_distance,
    null as inbound_crs_elapsed_min,
    'no_tail' as turnaround_band,
    'none' as rotation_position_key
from legs
where tail_number is null
