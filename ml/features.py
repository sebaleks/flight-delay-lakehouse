"""Canonical feature registry — the single source of truth for what the models
may see. Everything here mirrors the gold ML feature mart
(``ml_flight_features``), which this pipeline consumes AS-IS from BigQuery:
the same gold layer the dashboard reads, no duplicated data or logic. The
mart already enforces the leakage boundary (CLAUDE.md §9) — historical rates
are training-window-only, smoothed toward the global and constant within an
entity; weather is prior-day — and the audit in ``ml.audit`` re-asserts it
at train time.
"""

from __future__ import annotations

# --- model inputs (order is the canonical feature order) -------------------

CATEGORICAL_FEATURES = ["carrier", "origin", "dest", "route"]

NUMERIC_FEATURES = [
    # published-schedule features
    "distance",
    "crs_dep_hour",
    "crs_arr_hour",
    "day_of_week",  # BTS convention: 1 = Monday .. 7 = Sunday
    "month",
    # shared historical delay rates: training-window only, smoothed toward
    # the global — constant within an entity, identical for train and test
    "hist_route_arr_del15_rate",
    "hist_route_avg_arr_delay_minutes",
    "hist_route_n_flights",
    "hist_carrier_arr_del15_rate",
    "hist_carrier_avg_arr_delay_minutes",
    "hist_carrier_n_flights",
    "hist_origin_arr_del15_rate",
    "hist_origin_avg_arr_delay_minutes",
    "hist_origin_n_flights",
    "hist_dest_arr_del15_rate",
    "hist_dest_avg_arr_delay_minutes",
    "hist_dest_n_flights",
    # PRIOR-DAY origin weather (obs_date = flight_date - 1)
    "origin_mean_temp_f",
    "origin_max_temp_f",
    "origin_min_temp_f",
    "origin_visibility_mi",
    "origin_mean_wind_speed_kn",
    "origin_max_gust_kn",
    "origin_precip_in",
    "origin_snow_depth_in",
    "origin_had_fog",
    "origin_had_rain_drizzle",
    "origin_had_snow_ice_pellets",
    "origin_had_thunder",
    "has_origin_weather",
    # holiday calendar (knowable years ahead)
    "is_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# --- non-features ----------------------------------------------------------

LABELS = ["label_arr_del15", "label_arr_delay_minutes"]
SPLIT_COL = "is_training_row"
# identifiers / bookkeeping deliberately NOT fed to any model
EXCLUDED = ["flight_date", "flight_number", "crs_dep_time", SPLIT_COL, *LABELS]

# The full audited mart schema (must equal BigQuery INFORMATION_SCHEMA —
# the dbt guard assert_ml_features_no_leakage pins the same set).
MART_COLUMNS = FEATURES + EXCLUDED

# Post-departure / arrival outcome columns that must NEVER appear as features
# (mirrors dbt/tests/assert_ml_features_no_leakage.sql).
FORBIDDEN_FEATURES = frozenset(
    {
        "dep_time",
        "dep_delay",
        "dep_delay_minutes",
        "dep_del15",
        "departure_delay_groups",
        "dep_time_blk",
        "taxi_out",
        "wheels_off",
        "wheels_on",
        "taxi_in",
        "arr_time",
        "arr_delay",
        "arr_delay_minutes",
        "arr_del15",
        "arrival_delay_groups",
        "arr_time_blk",
        "actual_elapsed_time",
        "air_time",
        "cancelled",
        "cancellation_code",
        "diverted",
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "security_delay",
        "late_aircraft_delay",
        "first_dep_time",
        "total_add_g_time",
        "longest_add_g_time",
        "div_airport_landings",
        "div_reached_dest",
        "div_actual_elapsed_time",
        "div_arr_delay",
        "div_distance",
    }
)
