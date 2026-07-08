# gold/star/ (BigQuery tables) — analytics star schema

Dimensional model for BI/analytics:

- `fact_flights` — one row per flight leg; foreign keys to the dims, plus
  measures (delays, distance, etc.).
- `dim_airport` — airport attributes (name, city, coords, timezone).
- `dim_carrier` — carrier attributes.
- `dim_date` — calendar dimension (incl. holiday flags from the holiday seed).

Built from silver. Keep this normalized (star) — the ML mart is separate.
