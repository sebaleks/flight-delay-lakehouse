-- Standing leakage guard for the cascade features: every rotation feature in
-- the ML mart must derive from SCHEDULE columns only. The tempting leak —
-- wiring the prior leg's ACTUAL arrival anywhere into the rotation chain —
-- is caught VALUE-LEVEL: this test rebuilds the ENTIRE rotation feature set
-- independently from silver_flights using nothing but crs_dep_time /
-- crs_elapsed_time / tz / tail_number / dest (no actual_* or *_delay column
-- is referenced anywhere in this file — that absence is the point) and
-- compares against the mart over the FULL table. A chain accidentally built
-- on actual times diverges from this recompute wherever a prior leg ran
-- late, i.e. massively. Coverage (each with its own violation label):
--   * sched_turnaround_min, inbound_distance, inbound_crs_elapsed_min,
--     sched_turnaround_slack_min, is_tight_turnaround — exact value match;
--   * rotation_position, legs_today, origin_dep_density_hour — exact match;
--   * hist_turnaround_band_* / hist_rotation_position_* — must be CONSTANT
--     within the RECOMPUTED band/position (pins the rates join to the clean
--     schedule-derived key: a band derived from actuals in the int model
--     would split a recomputed band across multiple hist values);
--   * flag mirror + no-inbound attribute nulls + position-without-tail.
-- The chain is recomputed over ALL scheduled legs (including later-cancelled
-- ones — unknowable at prediction time), exactly the int-model convention,
-- with the same duty window (0-14h) and station-continuity rule
-- (prior dest = this origin).

with legs as (

    select
        flight_date,
        reporting_airline as carrier,
        flight_number_reporting_airline as flight_number,
        origin,
        dest,
        crs_dep_time,
        tail_number,
        distance,
        crs_elapsed_time,
        extract(hour from crs_dep_time) as dep_hour,
        timestamp(datetime(flight_date, crs_dep_time), origin_tz) as dep_ts,
        timestamp_add(
            timestamp(datetime(flight_date, crs_dep_time), origin_tz),
            interval cast(crs_elapsed_time as int64) minute
        ) as arr_ts,
        count(*) over (
            partition by origin, flight_date, extract(hour from crs_dep_time)
        ) as exp_density
    from {{ ref('silver_flights') }}

),

chained as (

    select
        *,
        lag(arr_ts) over w as prior_arr_ts,
        lag(dest) over w as prior_dest,
        lag(distance) over w as prior_distance,
        lag(crs_elapsed_time) over w as prior_elapsed,
        row_number() over (
            partition by tail_number, flight_date
            order by dep_ts, carrier, flight_number, origin, dest
        ) as exp_position,
        count(*) over (partition by tail_number, flight_date) as exp_legs_today
    from legs
    where tail_number is not null
    window w as (
        partition by tail_number
        order by dep_ts, carrier, flight_number, origin, dest
    )

),

expected as (

    select
        flight_date,
        carrier,
        flight_number,
        origin,
        dest,
        crs_dep_time,
        exp_position,
        exp_legs_today,
        exp_density,
        prior_arr_ts is not null
            and timestamp_diff(dep_ts, prior_arr_ts, minute) between 0 and 840
            and prior_dest = origin as exp_has_inbound,
        case
            when prior_arr_ts is not null
                and timestamp_diff(dep_ts, prior_arr_ts, minute) between 0 and 840
                and prior_dest = origin
                then timestamp_diff(dep_ts, prior_arr_ts, minute)
        end as exp_turnaround_min,
        prior_distance,
        prior_elapsed
    from chained

),

mart as (

    select
        flight_date, carrier, flight_number, origin, dest, crs_dep_time,
        sched_turnaround_min, sched_turnaround_slack_min, has_inbound_leg,
        is_tight_turnaround, inbound_distance, inbound_crs_elapsed_min,
        rotation_position, legs_today, origin_dep_density_hour,
        hist_turnaround_band_arr_del15_rate, hist_turnaround_band_n_flights,
        hist_rotation_position_arr_del15_rate, hist_rotation_position_n_flights
    from {{ ref('ml_flight_features') }}

),

joined as (

    -- null-safe on flight_number (the one nullable natural-key column) so
    -- the NULL-flight-number leg is value-pinned like every other row
    select mart.*, expected.* except (flight_date, carrier, flight_number,
                                      origin, dest, crs_dep_time)
    from mart
    inner join expected
        on expected.flight_date = mart.flight_date
        and expected.carrier = mart.carrier
        and expected.flight_number is not distinct from mart.flight_number
        and expected.origin = mart.origin
        and expected.dest = mart.dest
        and expected.crs_dep_time = mart.crs_dep_time

)

select flight_date, origin, carrier, 'turnaround_mismatch' as violation
from joined
where sched_turnaround_min is distinct from exp_turnaround_min

union all

select flight_date, origin, carrier, 'inbound_attr_mismatch' as violation
from joined
where
    has_inbound_leg
    and (
        inbound_distance is distinct from prior_distance
        or inbound_crs_elapsed_min is distinct from prior_elapsed
        or sched_turnaround_slack_min is distinct from exp_turnaround_min - 35
        or is_tight_turnaround is distinct from (exp_turnaround_min < 35)
    )

union all

select flight_date, origin, carrier, 'position_or_day_mismatch' as violation
from joined
where
    rotation_position is distinct from exp_position
    or legs_today is distinct from exp_legs_today
    or origin_dep_density_hour is distinct from exp_density

union all

select flight_date, origin, carrier, 'flag_mismatch' as violation
from joined
-- IS DISTINCT FROM, not != : a NULL flag (rotation-join miss) must FAIL
-- here, never null out of the comparison silently
where has_inbound_leg is distinct from exp_has_inbound

union all

-- the int model emits a row for EVERY scheduled leg (tail-known or not), so
-- a NULL flag in the mart means the rotation join itself missed — fail loud
select flight_date, origin, carrier, 'rotation_join_miss' as violation
from mart
where has_inbound_leg is null

union all

select flight_date, origin, carrier, 'inbound_attrs_without_inbound' as violation
from mart
where
    not has_inbound_leg
    and (
        sched_turnaround_min is not null
        or sched_turnaround_slack_min is not null
        or inbound_distance is not null
        or inbound_crs_elapsed_min is not null
    )

union all

select mart.flight_date, mart.origin, mart.carrier, 'position_without_tail' as violation
from mart
left join expected
    on expected.flight_date = mart.flight_date
    and expected.carrier = mart.carrier
    and expected.flight_number is not distinct from mart.flight_number
    and expected.origin = mart.origin
    and expected.dest = mart.dest
    and expected.crs_dep_time = mart.crs_dep_time
where expected.flight_date is null and mart.rotation_position is not null

union all

-- hist rates must be constant within the RECOMPUTED band: a band key derived
-- from actuals in the int model splits a clean band across hist values
select date '1900-01-01', band, 'ALL', 'hist_not_constant_within_band' as violation
from (
    select
        case
            when not exp_has_inbound then 'no_inbound'
            when exp_turnaround_min < 35 then 'lt_35'
            when exp_turnaround_min < 60 then '35_60'
            when exp_turnaround_min < 120 then '60_120'
            else 'ge_120'
        end as band,
        count(distinct hist_turnaround_band_arr_del15_rate) as n_rates,
        count(distinct hist_turnaround_band_n_flights) as n_ns
    from joined
    group by band
)
where n_rates > 1 or n_ns > 1

union all

select date '1900-01-01', pos, 'ALL', 'hist_not_constant_within_position' as violation
from (
    select
        cast(least(exp_position, 6) as string) as pos,
        count(distinct hist_rotation_position_arr_del15_rate) as n_rates,
        count(distinct hist_rotation_position_n_flights) as n_ns
    from joined
    group by pos
)
where n_rates > 1 or n_ns > 1

union all

-- IDENTITY pin (constancy alone would miss a consistent RELABELING): each
-- recomputed band's joined n_flights must equal the rates model's n_flights
-- for that same key — the join must be the identity on band labels
select date '1900-01-01', band, 'ALL', 'band_hist_relabeled' as violation
from (
    select
        case
            when not exp_has_inbound then 'no_inbound'
            when exp_turnaround_min < 35 then 'lt_35'
            when exp_turnaround_min < 60 then '35_60'
            when exp_turnaround_min < 120 then '60_120'
            else 'ge_120'
        end as band,
        any_value(hist_turnaround_band_n_flights) as joined_n
    from joined
    group by band
) as per_band
inner join {{ ref('int_historical_delay_rates') }} as rates
    on rates.entity_level = 'turnaround_band' and rates.entity_key = per_band.band
where per_band.joined_n is distinct from rates.n_flights

union all

select date '1900-01-01', pos, 'ALL', 'position_hist_relabeled' as violation
from (
    select
        cast(least(exp_position, 6) as string) as pos,
        any_value(hist_rotation_position_n_flights) as joined_n
    from joined
    group by pos
) as per_pos
inner join {{ ref('int_historical_delay_rates') }} as rates
    on rates.entity_level = 'rotation_position' and rates.entity_key = per_pos.pos
where per_pos.joined_n is distinct from rates.n_flights
