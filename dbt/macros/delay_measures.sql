{% macro delay_measures() %}
        count(*) as n_flight_legs,
        countif(cancelled) as n_cancelled,
        countif(cancelled) / count(*) as cancellation_rate,
        countif(diverted) as n_diverted,
        countif(diverted) / count(*) as diversion_rate,
        countif(arr_del15) as n_arr_del15,
        countif(arr_del15) / nullif(countif(arr_del15 is not null), 0) as arr_del15_rate,
        avg(arr_delay_minutes) as avg_arr_delay_minutes,
        avg(dep_delay_minutes) as avg_dep_delay_minutes,
        approx_quantiles(arr_delay_minutes, 100)[offset(90)] as p90_arr_delay_minutes
{%- endmacro %}
