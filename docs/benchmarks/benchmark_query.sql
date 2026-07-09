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
