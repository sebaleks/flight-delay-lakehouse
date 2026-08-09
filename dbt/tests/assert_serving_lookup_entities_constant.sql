-- Standing guard for the serving lookup layer: every hist_* value that
-- serving_entity_profile collapses with any_value() must be CONSTANT within
-- its entity in the mart. Returns one row per (level, key) that is not.
--
-- WHY THIS EXISTS. serving_entity_profile reads the hist_* triples with
-- any_value(...) group by <key>. That is only correct because the shared
-- rates model makes them constant within an entity — which is the property
-- that lets serving reproduce TRAINING values byte-for-byte with zero
-- formula duplication. If it ever stops holding, any_value() picks
-- arbitrarily and the picked value changes between dbt rebuilds, silently
-- moving predictions. That is exactly the defect the same PR fixed for route
-- distance (85 of 7,539 routes carried two distances, so the model uses
-- min(distance) and this test deliberately does NOT cover that column).
--
-- The turnaround_band / rotation_position levels are the sharper case: the
-- lookup model RE-DERIVES those keys with its own CASE expression, mirroring
-- int_aircraft_rotation.sql. If a threshold moves on one side only, a single
-- key would span rows carrying two different hist values and this test fails
-- rather than letting the drift through.

{% set entity_grains = ['route', 'carrier', 'origin', 'dest'] %}

with

{% for grain in entity_grains %}
{{ grain }}_level as (
    select
        '{{ grain }}' as entity_level,
        {{ grain }} as entity_key,
        count(distinct hist_{{ grain }}_arr_del15_rate) as n_rate,
        count(distinct hist_{{ grain }}_avg_arr_delay_minutes) as n_avg,
        count(distinct hist_{{ grain }}_n_flights) as n_n
    from {{ ref('ml_flight_features') }}
    group by entity_key
),
{% endfor %}

turnaround_band_level as (
    select
        'turnaround_band' as entity_level,
        case
            when not has_inbound_leg then 'no_inbound'
            when sched_turnaround_min < 35 then 'lt_35'
            when sched_turnaround_min < 60 then '35_60'
            when sched_turnaround_min < 120 then '60_120'
            else 'ge_120'
        end as entity_key,
        count(distinct hist_turnaround_band_arr_del15_rate),
        count(distinct hist_turnaround_band_avg_arr_delay_minutes),
        count(distinct hist_turnaround_band_n_flights)
    from {{ ref('ml_flight_features') }}
    where rotation_position is not null
    group by entity_key
),

rotation_position_level as (
    select
        'rotation_position' as entity_level,
        cast(least(rotation_position, 6) as string) as entity_key,
        count(distinct hist_rotation_position_arr_del15_rate),
        count(distinct hist_rotation_position_avg_arr_delay_minutes),
        count(distinct hist_rotation_position_n_flights)
    from {{ ref('ml_flight_features') }}
    where rotation_position is not null
    group by entity_key
),

all_levels as (
    {% for grain in entity_grains %}
    select * from {{ grain }}_level union all
    {% endfor %}
    select * from turnaround_band_level
    union all select * from rotation_position_level
)

select entity_level, entity_key, n_rate, n_avg, n_n
from all_levels
-- a NULL hist value yields count(distinct) = 0, which is fine (the whole
-- entity is absent from the training window); only a genuine SPLIT fails
where n_rate > 1 or n_avg > 1 or n_n > 1
