-- fact_flights pruning benchmark: the representative dashboard drill-down.
-- Run against `fact_flights` and against the unpartitioned baseline below;
-- measure bytes from EXECUTED job statistics (dry-run bytes are an
-- UPPER_BOUND estimate for clustered tables, not exact).
--
-- Baseline recreate DDL (plain CTAS: no PARTITION BY, no CLUSTER BY):
--   CREATE OR REPLACE TABLE `de-flight-project.flight_delays_gold.fact_flights_benchmark_baseline`
--   AS SELECT * FROM `de-flight-project.flight_delays_gold.fact_flights`;
-- Run each variant with cache disabled, e.g.:
--   bq query --nouse_legacy_sql --nouse_cache --job_id=<id> < benchmark_query.sql
--   bq show --format=prettyjson -j <id>   # statistics.query.totalBytesProcessed
SELECT
  carrier_key,
  COUNT(*) AS n_flights,
  COUNTIF(arr_del15) / NULLIF(COUNTIF(arr_del15 IS NOT NULL), 0) AS arr_del15_rate,
  AVG(arr_delay_minutes) AS avg_arr_delay_minutes,
  COUNTIF(cancelled) AS n_cancelled
FROM `de-flight-project.flight_delays_gold.fact_flights`
WHERE date_key BETWEEN '2024-06-01' AND '2024-06-30'
  AND origin_airport_key = 'ORD'
GROUP BY carrier_key
ORDER BY n_flights DESC
