{#
  Map a model's +schema straight to the BigQuery dataset name, with NO
  "<target_schema>_<custom>" prefixing that dbt does by default. This lets us
  land models in the exact datasets bronze / silver / gold (from env vars).
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- set default_schema = target.schema -%}
    {%- if custom_schema_name is none -%}
        {{ default_schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
