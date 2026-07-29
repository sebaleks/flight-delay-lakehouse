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
--   * class-aware flag semantics (a TRUE / b FALSE / c NULL) + full
--     nulling of swap-shaped rows + clean-first-leg attribute rules;
--   * the CONVERSE direction: clean (class-a/b) rows must CARRY all six
--     hist values — rotation-hist NULL must mean class c, nothing else.
-- The chain is recomputed over ALL scheduled legs (including later-cancelled
-- ones — unknowable at prediction time), exactly the int-model convention,
-- with the same duty window (0-14h), station-continuity rule (prior dest =
-- this origin), and the TAIL-SWAP RESTRICTION classes: a = consistent
-- inbound, b = clean first leg, c = swap-shaped (every rotation feature
-- and hist value must be NULL — the leak the restriction removes).

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
        -- schedule-consistency class (the tail-swap restriction):
        -- a = consistent inbound, b = clean first leg, c = swap-shaped
        case
            -- mirror of the int model: prior_dest NULL <=> no prior at all;
            -- a prior with unknown arrival falls through to 'c'
            when prior_dest is null then 'b'
            when timestamp_diff(dep_ts, prior_arr_ts, minute) between 0 and 840
                and prior_dest = origin and not prior_overlapped then 'a'
            when timestamp_diff(dep_ts, prior_arr_ts, minute) > 840
                and prior_dest = origin and not prior_overlapped then 'b'
            else 'c'
        end as exp_class,
        case
            when prior_arr_ts is not null
                and timestamp_diff(dep_ts, prior_arr_ts, minute) between 0 and 840
                and prior_dest = origin
                and not prior_overlapped
                then timestamp_diff(dep_ts, prior_arr_ts, minute)
        end as exp_turnaround_min,
        prior_distance,
        prior_elapsed
    from (
        select
            *,
            -- mirror the int model's overlap fail-closed rule: a prior leg
            -- whose own gap was negative is not a trustworthy inbound
            coalesce(
                lag(timestamp_diff(dep_ts, prior_arr_ts, minute) < 0) over (
                    partition by tail_number
                    order by dep_ts, carrier, flight_number, origin, dest
                ),
                false
            ) as prior_overlapped
        from chained
    )

),

mart as (

    select
        flight_date, carrier, flight_number, origin, dest, crs_dep_time,
        sched_turnaround_min, sched_turnaround_slack_min, has_inbound_leg,
        is_tight_turnaround, inbound_distance, inbound_crs_elapsed_min,
        rotation_position, legs_today, origin_dep_density_hour,
        hist_turnaround_band_arr_del15_rate, hist_turnaround_band_n_flights,
        hist_turnaround_band_avg_arr_delay_minutes,
        hist_rotation_position_arr_del15_rate, hist_rotation_position_n_flights,
        hist_rotation_position_avg_arr_delay_minutes
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
    origin_dep_density_hour is distinct from exp_density
    or (
        exp_class != 'c'
        and (
            rotation_position is distinct from exp_position
            or legs_today is distinct from exp_legs_today
        )
    )

union all

select flight_date, origin, carrier, 'flag_mismatch' as violation
from joined
-- class-aware flag semantics under the restriction:
-- a -> TRUE, b -> FALSE, c -> NULL (swap-shaped: unknowable). IS DISTINCT
-- FROM keeps NULLs comparable — a wrong NULL/false swap fails here.
where has_inbound_leg is distinct from
    case exp_class when 'a' then true when 'b' then false else cast(null as bool) end

union all

-- c rows must be FULLY nulled — a band or position surviving on a
-- swap-shaped link is exactly the leak the restriction removes
select flight_date, origin, carrier, 'swap_shaped_not_nulled' as violation
from joined
where exp_class = 'c' and (
    rotation_position is not null or legs_today is not null
    or sched_turnaround_min is not null or sched_turnaround_slack_min is not null
    or is_tight_turnaround is not null
    or inbound_distance is not null or inbound_crs_elapsed_min is not null
    -- ALL six hist columns, not just the rates: a partial null-out that
    -- leaves avg/n populated would still feed the model
    or hist_turnaround_band_arr_del15_rate is not null
    or hist_turnaround_band_avg_arr_delay_minutes is not null
    or hist_turnaround_band_n_flights is not null
    or hist_rotation_position_arr_del15_rate is not null
    or hist_rotation_position_avg_arr_delay_minutes is not null
    or hist_rotation_position_n_flights is not null
)

union all

-- the CONVERSE of swap_shaped_not_nulled (added after the pre-merge review:
-- two independent reviewers converged on the hole): clean links must
-- actually CARRY their hist values. Bands and positions are CLOSED sets,
-- every key present in the training window, so NULL hist on a class-a/b row
-- means the band key was lost upstream — and no other arm can see that:
-- the constancy arms count DISTINCT non-null values (an all-NULL band
-- passes as 0 distinct), and the identity arms' inner join silently drops
-- an entity row that vanished from the rates model along with its key
select flight_date, origin, carrier, 'clean_link_hist_missing' as violation
from joined
where exp_class != 'c' and (
    hist_turnaround_band_arr_del15_rate is null
    or hist_turnaround_band_avg_arr_delay_minutes is null
    or hist_turnaround_band_n_flights is null
    or hist_rotation_position_arr_del15_rate is null
    or hist_rotation_position_avg_arr_delay_minutes is null
    or hist_rotation_position_n_flights is null
)

union all

-- the int model emits a row for EVERY scheduled leg (tail-known or not), so
-- a NULL density in the mart means the rotation join itself missed — fail
-- loud (density is the one column the int model never nulls)
select flight_date, origin, carrier, 'rotation_join_miss' as violation
from mart
where origin_dep_density_hour is null

union all

select flight_date, origin, carrier, 'inbound_attrs_without_inbound' as violation
from joined
where
    exp_class = 'b'
    and (
        sched_turnaround_min is not null
        or sched_turnaround_slack_min is not null
        or inbound_distance is not null
        or inbound_crs_elapsed_min is not null
        -- clean first legs must be exactly FALSE (the int model's
        -- coalesce), never NULL — a silent FALSE->NULL distribution change
        -- in a model feature must fail here
        or is_tight_turnaround is distinct from false
        or rotation_position is null
        or legs_today is null
    )

union all

-- hist rates must be constant within the RECOMPUTED band: a band key derived
-- from actuals in the int model splits a clean band across hist values
select date '1900-01-01', band, 'ALL', 'hist_not_constant_within_band' as violation
from (
    select
        case
            when exp_class = 'b' then 'no_inbound'
            when exp_turnaround_min < 35 then 'lt_35'
            when exp_turnaround_min < 60 then '35_60'
            when exp_turnaround_min < 120 then '60_120'
            else 'ge_120'
        end as band,
        count(distinct hist_turnaround_band_arr_del15_rate) as n_rates,
        count(distinct hist_turnaround_band_n_flights) as n_ns,
        count(distinct hist_turnaround_band_avg_arr_delay_minutes) as n_avgs
    from joined
    where exp_class != 'c'
    group by band
)
where n_rates > 1 or n_ns > 1 or n_avgs > 1

union all

select date '1900-01-01', pos, 'ALL', 'hist_not_constant_within_position' as violation
from (
    select
        cast(least(exp_position, 6) as string) as pos,
        count(distinct hist_rotation_position_arr_del15_rate) as n_rates,
        count(distinct hist_rotation_position_n_flights) as n_ns,
        count(distinct hist_rotation_position_avg_arr_delay_minutes) as n_avgs
    from joined
    where exp_class != 'c'
    group by pos
)
where n_rates > 1 or n_ns > 1 or n_avgs > 1

union all

-- IDENTITY pin (constancy alone would miss a consistent RELABELING): each
-- recomputed band's joined n_flights must equal the rates model's n_flights
-- for that same key — the join must be the identity on band labels
select date '1900-01-01', band, 'ALL', 'band_hist_relabeled' as violation
from (
    select
        case
            when exp_class = 'b' then 'no_inbound'
            when exp_turnaround_min < 35 then 'lt_35'
            when exp_turnaround_min < 60 then '35_60'
            when exp_turnaround_min < 120 then '60_120'
            else 'ge_120'
        end as band,
        any_value(hist_turnaround_band_n_flights) as joined_n
    from joined
    where exp_class != 'c'
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
    where exp_class != 'c'
    group by pos
) as per_pos
inner join {{ ref('int_historical_delay_rates') }} as rates
    on rates.entity_level = 'rotation_position' and rates.entity_key = per_pos.pos
where per_pos.joined_n is distinct from rates.n_flights

union all

-- tail-null mart rows (absent from `expected`) are class c under the
-- restriction: every rotation feature NULL, hist NULL
select mart.flight_date, mart.origin, mart.carrier, 'tail_null_not_nulled' as violation
from mart
left join expected
    on expected.flight_date = mart.flight_date
    and expected.carrier = mart.carrier
    and expected.flight_number is not distinct from mart.flight_number
    and expected.origin = mart.origin
    and expected.dest = mart.dest
    and expected.crs_dep_time = mart.crs_dep_time
where
    expected.flight_date is null
    and (
        mart.rotation_position is not null
        or mart.legs_today is not null
        or mart.has_inbound_leg is not null
        or mart.sched_turnaround_min is not null
        or mart.sched_turnaround_slack_min is not null
        or mart.inbound_distance is not null
        or mart.inbound_crs_elapsed_min is not null
        or mart.is_tight_turnaround is not null
        or mart.hist_turnaround_band_arr_del15_rate is not null
        or mart.hist_turnaround_band_avg_arr_delay_minutes is not null
        or mart.hist_turnaround_band_n_flights is not null
        or mart.hist_rotation_position_arr_del15_rate is not null
        or mart.hist_rotation_position_avg_arr_delay_minutes is not null
        or mart.hist_rotation_position_n_flights is not null
        -- density is the one column class c KEEPS (schedule aggregate)
        or mart.origin_dep_density_hour is null
    )
