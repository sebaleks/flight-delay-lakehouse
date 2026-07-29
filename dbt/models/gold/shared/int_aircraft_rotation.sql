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
-- THE TAIL-SWAP RESTRICTION (adopted 2026-07 after a gating experiment;
-- this RESOLVES the operated-tail epistemology caveat): BTS records the
-- OPERATED tail post-hoc, so the linkage itself can be a day-of operational
-- outcome — same-day swaps restructure which legs chain together, and
-- swap-shaped links carry disruption information no one had pre-departure.
-- The experiment quantified it: with rotation features present ONLY for
-- schedule-consistent links (91.95% consistent inbound + 3.93% clean first
-- leg; 4.12% swap-shaped nulled), 89% of the cascade PR-AUC uplift
-- survived (0.4652 vs 0.4748 contaminated, over the 0.3893 baseline), and
-- the no_inbound band's training rate fell 0.388 -> 0.224 once swap rows
-- left it — the elevated rate was substantially a swap fingerprint.
-- Production therefore ships the RESTRICTED definition:
--   * has_inbound_leg TRUE  — schedule-consistent inbound: gap in [0, 14h],
--     station continuity (prior dest = this origin), prior leg not
--     schedule-overlapped;
--   * has_inbound_leg FALSE — clean first leg: no prior, or an overnight
--     break (>14h) parked at this origin; keeps position/legs and its own
--     (now clean) no_inbound band history;
--   * has_inbound_leg NULL  — SWAP-SHAPED linkage (negative gap, continuity
--     violation, overlapped prior, unknown tail): not schedule-explained,
--     not knowable pre-departure -> EVERY rotation feature NULL, band NULL
--     (excluded from the rates and from consumption). Density is kept — a
--     schedule aggregate, not tail-based.
--
-- Known accepted limitation (shared with the weather join, counted there as
-- 1 mart row): BTS '2400' scheduled times are stored as 00:00 of
-- flight_date, so a true end-of-day midnight departure sorts ~24h early in
-- the chain, perturbing its own and its neighbor's position/turnaround —
-- a handful of legs, never a leak (all inputs remain schedule columns).
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

overlap_flagged as (

    -- a leg whose own gap is negative OVERLAPS its predecessor's scheduled
    -- interval — the tail record there is unreliable (2/3 of overlaps
    -- exceed 30 min, i.e. phantom/duplicate entries, not rounding); its
    -- SUCCESSOR must not treat it as a trustworthy inbound: fail closed
    select
        *,
        coalesce(
            lag(raw_gap_min < 0) over (
                partition by tail_number
                order by dep_ts_utc, carrier, flight_number, origin, dest
            ),
            false
        ) as prior_leg_overlapped
    from with_turnaround

),

classified as (

    -- SCHEDULE-CONSISTENCY CLASSES (the tail-swap restriction, adopted after
    -- the 2026-07 leakage experiment — see the header):
    --   a: consistent inbound  — gap in the duty window, station continuity,
    --      prior not schedule-overlapped
    --   b: clean first leg     — no prior at all, or an overnight break
    --      (>14h) that ENDS AT THIS ORIGIN (aircraft parked here)
    --   c: SWAP-SHAPED         — negative gap, continuity violation,
    --      overlapped prior: the operated linkage was restructured by day-of
    --      operations and is NOT knowable pre-departure -> every rotation
    --      feature NULL (band included). Density is kept — it is a schedule
    --      aggregate, not tail-based.
    select
        * except (raw_gap_min, prior_leg_overlapped),
        case
            -- no prior AT ALL (prior_dest NULL <=> no prior row; dest is
            -- never NULL) -> clean first leg. A prior whose arrival is
            -- UNKNOWN (elapsed-null leg, ~6 in 20.7M) falls through to 'c':
            -- an unexplained linkage is not a clean first leg
            when prior_dest is null then 'b'
            when raw_gap_min between 0 and 840
                and prior_dest = origin
                and not prior_leg_overlapped then 'a'
            when raw_gap_min > 840
                and prior_dest = origin
                and not prior_leg_overlapped then 'b'
            else 'c'
        end as link_class,
        case
            when raw_gap_min between 0 and 840
                and prior_dest = origin
                and not prior_leg_overlapped
                then raw_gap_min
        end as sched_turnaround_min
    from overlap_flagged

)

select
    flight_date,
    carrier,
    flight_number,
    origin,
    dest,
    crs_dep_time,
    if(link_class = 'c', null, rotation_position) as rotation_position,
    if(link_class = 'c', null, legs_today) as legs_today,
    origin_dep_density_hour,
    case when link_class = 'c' then null else link_class = 'a' end as has_inbound_leg,
    sched_turnaround_min,
    -- slack vs a typical 35-minute narrow-body minimum turnaround
    sched_turnaround_min - 35 as sched_turnaround_slack_min,
    case
        when link_class = 'c' then null
        else coalesce(sched_turnaround_min < 35, false)
    end as is_tight_turnaround,
    case when link_class = 'a' then inbound_distance end as inbound_distance,
    case when link_class = 'a' then inbound_crs_elapsed_min end as inbound_crs_elapsed_min,
    -- BAND KEYS for the shared historical rates. NULL means SWAP-SHAPED
    -- (linkage not schedule-explained — excluded from the rates and from
    -- consumption); clean first legs keep their own no_inbound history.
    case
        when link_class = 'c' then null
        when link_class = 'b' then 'no_inbound'
        when sched_turnaround_min < 35 then 'lt_35'
        when sched_turnaround_min < 60 then '35_60'
        when sched_turnaround_min < 120 then '60_120'
        else 'ge_120'
    end as turnaround_band,
    if(link_class = 'c', null, cast(least(rotation_position, 6) as string))
        as rotation_position_key
from classified

union all

-- tail-unknown legs (0.34%): no chain — class c under the restriction
-- (linkage not knowable), every rotation feature NULL, density kept
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
    cast(null as bool) as has_inbound_leg,
    null as sched_turnaround_min,
    null as sched_turnaround_slack_min,
    cast(null as bool) as is_tight_turnaround,
    null as inbound_distance,
    null as inbound_crs_elapsed_min,
    cast(null as string) as turnaround_band,
    cast(null as string) as rotation_position_key
from legs
where tail_number is null
