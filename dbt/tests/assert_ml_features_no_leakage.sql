-- Standing leakage guard (CLAUDE.md §9), ALLOWLIST form: fails — one row per
-- offending column — if the built ML feature mart contains ANY column not in
-- the audited set below. Stronger than a forbidden-names blacklist: a renamed
-- outcome column, a typo, or a new un-audited feature all fail by default.
-- Also fails with 'TABLE_MISSING_OR_RENAMED' if the table is absent, so a
-- rename/alias can never make this test pass vacuously.
--
-- Known limit (by design): this pins NAMES, not lineage — a leaky expression
-- laundered through an approved-sounding alias still needs human review when
-- this list is edited. Any change here is a leakage-boundary change.

-- depends_on: {{ ref('ml_flight_features') }}

{% set allowed_columns = [
    'flight_date', 'carrier', 'flight_number', 'origin', 'dest', 'route',
    'crs_dep_time',
    'distance', 'crs_dep_hour', 'crs_arr_hour', 'day_of_week', 'month',
    'hist_route_arr_del15_rate', 'hist_route_avg_arr_delay_minutes', 'hist_route_n_flights',
    'hist_carrier_arr_del15_rate', 'hist_carrier_avg_arr_delay_minutes', 'hist_carrier_n_flights',
    'hist_origin_arr_del15_rate', 'hist_origin_avg_arr_delay_minutes', 'hist_origin_n_flights',
    'hist_dest_arr_del15_rate', 'hist_dest_avg_arr_delay_minutes', 'hist_dest_n_flights',
    'origin_mean_temp_f', 'origin_max_temp_f', 'origin_min_temp_f',
    'origin_visibility_mi', 'origin_mean_wind_speed_kn', 'origin_max_gust_kn',
    'origin_precip_in', 'origin_snow_depth_in',
    'origin_had_fog', 'origin_had_rain_drizzle', 'origin_had_snow_ice_pellets',
    'origin_had_thunder', 'has_origin_weather',
    'is_holiday', 'is_day_before_holiday', 'is_day_after_holiday',
    'is_training_row',
    'label_arr_del15', 'label_arr_delay_minutes',
] %}

with built_columns as (

    select lower(column_name) as column_name
    from `{{ env_var('GCP_PROJECT_ID') }}.{{ env_var('BQ_GOLD_DATASET', 'flight_delays_gold') }}`.INFORMATION_SCHEMA.COLUMNS
    where table_name = 'ml_flight_features'

),

unexpected as (

    select column_name
    from built_columns
    where column_name not in (
        {%- for col in allowed_columns %}
        '{{ col }}'{{ ',' if not loop.last }}
        {%- endfor %}
    )

),

missing_table as (

    select 'TABLE_MISSING_OR_RENAMED' as column_name
    from (select count(*) as n_built from built_columns)
    where n_built = 0

)

select column_name from unexpected
union all
select column_name from missing_table
