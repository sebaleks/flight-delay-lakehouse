{{
    config(
        materialized='table',
        partition_by={'field': 'obs_date', 'data_type': 'date', 'granularity': 'month'},
        cluster_by=['station_id'],
    )
}}

-- Hourly surface observations decoded from the ISD bronze access layer.
-- One row per (station_id, obs_ts_utc) after dedup. Timestamps are UTC —
-- the ISD convention; consumers joining to LOCAL schedule times must convert
-- (the ML mart does, via the airports seed tz column).
--
-- Decoding rules (ISD packed fields; sentinels +9999/9999/999999 = missing):
--   * QC POLICY (owner-approved): element quality codes 2, 3, 6, 7
--     (suspect/erroneous, checked or unchecked source) null the ELEMENT, not
--     the row. Affected fractions are reported at build time.
--   * TMP/DEW: "-0194,5" = value in tenths degC + QC -> Fahrenheit.
--   * WND: "240,5,N,0057,5" -> speed in tenths m/s -> knots.
--   * VIS: metres -> statute miles. Most stations cap at 16093 m (= 10 mi);
--     ~0.16% of observations (extended-reporting software) exceed it, max
--     observed 99.4 mi. Kept RAW here; the ML mart right-censors uniformly
--     at 10.0 ("10 or better") to remove instrumentation heterogeneity.
--   * OC1 gust: tenths m/s -> knots; NULL when not reported (gusts are only
--     encoded when observed). gust_reported carries the distinction so
--     downstream can encode absent-as-calm (0.0 + indicator) without losing
--     the reported/absent signal. NULL here is honest observation semantics.
--   * PRECIP (owner-approved policy): AA1-AA4 are period-coded accumulation
--     groups ("01,0000,9,5" = period hrs, depth in tenths mm, condition, QC).
--     ONLY the 1-hour (period 01) group feeds precip_1h_in — accumulation
--     windows are never mixed. No AA groups at all -> 0.0 (the METAR
--     convention: nothing to report means no precipitation). Groups present
--     but no usable 1-hour group (longer-period synoptic totals only, or the
--     1-hr group failed QC) -> NULL, honest missingness.
--   * Present weather: MW1-MW7 (manual, WMO ww table 4677) and AW1-AW6
--     (automated, WMO wawa table 4680) codes -> coarse booleans matching the
--     GSOD-era flag names. Range mapping (documented judgment calls; MIXED
--     rain-and-snow codes set BOTH flags; freezing LIQUID precip counts as
--     rain, the FRSHTT convention):
--       fog:   MW 10-12 (mist/shallow fog), 40-49 | AW 10, 30-35
--       rain:  MW 50-69 (drizzle+rain incl freezing 56-57/66-67, mixed
--              68-69), 80-84 (showers) | AW 40-44 (unclassified/liquid),
--              47-48 (freezing), 50-58, 60-68, 80-84
--       snow:  MW 68-69 (mixed), 70-79, 83-88 (mixed + snow showers)
--              | AW 45-46 (solid), 67-68 (mixed), 70-78, 85-87
--       thunder: MW 17, 95-99 (at observation; 91-94 "recent" excluded)
--              | AW 90-96
--   * SOD/SOM rows (embedded daily/monthly summaries) are excluded; all true
--     observation report types (FM-15 METAR, FM-16 SPECI, FM-12 SYNOP, ...)
--     are kept, including rows with a NULL report type. Duplicate
--     (station, timestamp) rows dedup with a deterministic TOTAL order:
--     METAR > SPECI > SYNOP > other, then a fingerprint over every raw field
--     — ties beyond that are byte-identical rows, where any pick is the same
--     row. Never nondeterministic.

{% set bad_q = "('2', '3', '6', '7')" %}

with raw as (

    -- safe.parse + not-null filter: a malformed DATE drops that row rather
    -- than aborting the build (machine-generated ISO timestamps; malformed is
    -- vanishingly rare and a not_null test guards obs_ts_utc downstream).
    -- The report-type filter is null-safe: only literal SOD/SOM rows drop.
    select
        _station_id as station_id,
        safe.parse_timestamp('%Y-%m-%dT%H:%M:%S', `DATE`) as obs_ts_utc,
        trim(REPORT_TYPE) as report_type,
        WND, VIS, TMP, DEW, OC1,
        AA1, AA2, AA3, AA4,
        MW1, MW2, MW3, MW4, MW5, MW6, MW7,
        AW1, AW2, AW3, AW4, AW5, AW6
    from {{ source('bronze', 'isd_hourly') }}
    where
        (REPORT_TYPE is null or trim(REPORT_TYPE) not in ('SOD', 'SOM'))
        and safe.parse_timestamp('%Y-%m-%dT%H:%M:%S', `DATE`) is not null

),

decoded as (

    select
        station_id,
        obs_ts_utc,
        date(obs_ts_utc) as obs_date,
        report_type,

        -- TMP / DEW: tenths degC -> degF, sentinel +9999, element-level QC
        {% for src, col in [('TMP', 'temp_f'), ('DEW', 'dewpoint_f')] %}
        case
            when {{ src }} is null then null
            when safe_cast(split({{ src }}, ',')[safe_offset(0)] as int64) = 9999 then null
            when split({{ src }}, ',')[safe_offset(1)] in {{ bad_q }} then null
            else safe_cast(split({{ src }}, ',')[safe_offset(0)] as int64) / 10 * 9 / 5 + 32
        end as {{ col }},
        {% endfor %}

        -- WND speed: tenths m/s -> knots, sentinel 9999
        case
            when WND is null then null
            when safe_cast(split(WND, ',')[safe_offset(3)] as int64) = 9999 then null
            when split(WND, ',')[safe_offset(4)] in {{ bad_q }} then null
            else safe_cast(split(WND, ',')[safe_offset(3)] as int64) / 10 * 1.9438445
        end as wind_speed_kn,

        -- VIS: metres -> miles, sentinel 999999 (raw values; mart censors)
        case
            when VIS is null then null
            when safe_cast(split(VIS, ',')[safe_offset(0)] as int64) = 999999 then null
            when split(VIS, ',')[safe_offset(1)] in {{ bad_q }} then null
            else safe_cast(split(VIS, ',')[safe_offset(0)] as int64) / 1609.344
        end as visibility_mi,

        -- OC1 gust: tenths m/s -> knots; NULL = not reported (see header)
        case
            when OC1 is null then null
            when safe_cast(split(OC1, ',')[safe_offset(0)] as int64) = 9999 then null
            when split(OC1, ',')[safe_offset(1)] in {{ bad_q }} then null
            else safe_cast(split(OC1, ',')[safe_offset(0)] as int64) / 10 * 1.9438445
        end as gust_kn,

        -- precip accumulation groups, QC-filtered, period codes preserved
        array(
            select as struct
                split(g, ',')[safe_offset(0)] as period,
                safe_cast(split(g, ',')[safe_offset(1)] as int64) as depth_tenths_mm
            from unnest([AA1, AA2, AA3, AA4]) as g
            where g is not null
        ) as aa_all,
        array(
            select as struct
                split(g, ',')[safe_offset(0)] as period,
                safe_cast(split(g, ',')[safe_offset(1)] as int64) as depth_tenths_mm
            from unnest([AA1, AA2, AA3, AA4]) as g
            where
                g is not null
                and split(g, ',')[safe_offset(3)] not in {{ bad_q }}
                and safe_cast(split(g, ',')[safe_offset(1)] as int64) != 9999
        ) as aa_usable,

        -- present-weather codes with bad-QC codes dropped
        array(
            select safe_cast(split(c, ',')[safe_offset(0)] as int64)
            from unnest([MW1, MW2, MW3, MW4, MW5, MW6, MW7]) as c
            where c is not null and split(c, ',')[safe_offset(1)] not in {{ bad_q }}
        ) as mw_codes,
        array(
            select safe_cast(split(c, ',')[safe_offset(0)] as int64)
            from unnest([AW1, AW2, AW3, AW4, AW5, AW6]) as c
            where c is not null and split(c, ',')[safe_offset(1)] not in {{ bad_q }}
        ) as aw_codes,

        -- fingerprint over every raw field: a TOTAL dedup order (ties beyond
        -- it are byte-identical rows, where any pick yields the same output)
        farm_fingerprint(to_json_string(struct(
            report_type, WND, VIS, TMP, DEW, OC1,
            AA1, AA2, AA3, AA4,
            MW1, MW2, MW3, MW4, MW5, MW6, MW7,
            AW1, AW2, AW3, AW4, AW5, AW6
        ))) as raw_fp

    from raw

)

select
    station_id,
    obs_ts_utc,
    obs_date,
    report_type,
    temp_f,
    dewpoint_f,
    wind_speed_kn,
    visibility_mi,
    gust_kn,
    gust_kn is not null as gust_reported,
    case
        when array_length(aa_all) = 0 then 0.0
        when (select min(depth_tenths_mm) from unnest(aa_usable) where period = '01')
            is not null
            then (select min(depth_tenths_mm) from unnest(aa_usable) where period = '01')
                / 10 / 25.4
        else null
    end as precip_1h_in,
    exists(
        select 1 from unnest(mw_codes) as c
        where c between 40 and 49 or c between 10 and 12
    ) or exists(
        select 1 from unnest(aw_codes) as c
        where c = 10 or c between 30 and 35
    ) as had_fog,
    exists(
        select 1 from unnest(mw_codes) as c
        where c between 50 and 69 or c between 80 and 84
    ) or exists(
        select 1 from unnest(aw_codes) as c
        where c between 40 and 44 or c between 47 and 48  -- liquid + freezing
            or c between 50 and 58 or c between 60 and 68 or c between 80 and 84
    ) as had_rain_drizzle,
    exists(
        select 1 from unnest(mw_codes) as c
        where c between 68 and 79 or c between 83 and 88  -- mixed codes: both flags
    ) or exists(
        select 1 from unnest(aw_codes) as c
        where c between 45 and 46  -- solid unclassified
            or c between 67 and 68 or c between 70 and 78 or c between 85 and 87
    ) as had_snow_ice_pellets,
    exists(
        select 1 from unnest(mw_codes) as c
        where c = 17 or c between 95 and 99
    ) or exists(
        select 1 from unnest(aw_codes) as c
        where c between 90 and 96
    ) as had_thunder
from decoded
qualify row_number() over (
    partition by station_id, obs_ts_utc
    order by
        case report_type
            when 'FM-15' then 0
            when 'FM-16' then 1
            when 'FM-12' then 2
            else 3
        end,
        report_type,
        raw_fp
) = 1
