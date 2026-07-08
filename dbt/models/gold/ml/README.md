# gold/ml/ (BigQuery table) — wide flat ML feature mart

A single denormalized, one-row-per-flight table for model training/inference.
**Not** the star schema — ML consumers must not join dims at train time.

**Leakage rule (see CLAUDE.md §9):** every column here must be knowable
**before departure**. Include labels explicitly (`ArrDel15`, `ArrDelayMinutes`)
but keep them clearly separated from features; never derive a feature from any
at/after-departure outcome. Weather features must be forecast/historical, not
the flight's realized conditions.

Suggested output: `feature_flights` (features + both labels + a split column
driven by the time-based split).
