# gold/ml/ — the ML feature mart and the serving lookup layer

Two kinds of artifact live here, for two different consumers.

## 1. `ml_flight_features` — the wide flat feature mart (training)

A single denormalized, one-row-per-flight table for model training.
**Not** the star schema — ML consumers must not join dims at train time
(CLAUDE.md §4).

**Leakage rule (see CLAUDE.md §9):** every column here must be knowable
**before departure**. Labels are included explicitly (`label_arr_del15`,
`label_arr_delay_minutes`) but carry the `label_` prefix and are the only
post-departure columns; never derive a feature from any at/after-departure
outcome. Weather features are the last hourly ISD observation at or before
scheduled departure, not the flight's realized conditions. The schema is
pinned to an audited allowlist by `assert_ml_features_no_leakage` — any change
to that list is a leakage-boundary change.

## 2. `serving_*` — the serving lookup layer (inference)

Three small tables read once at process startup by `ml/serving.py`, so the
request path issues **zero** BigQuery queries. They are not feature marts and
are not one-row-per-flight; they materialize aggregates the inference path
previously computed per request against the 20.2M-row mart (2.31 GB and ~5 s
per call, flat in batch size).

| Model | Grain | Rows | Replaces |
|---|---|---|---|
| `serving_entity_profile` | (entity_level, entity_key) | 8,316 | 4 hist lookups + route distance + 2 rotation-grain queries + the category-vocab union |
| `serving_density_profile` | (origin, crs_dep_hour, day_of_week) | 34,979 | the per-request density estimate |
| `serving_typical_rotation` | one row | 1 | the typical-rotation medians |

Two properties they must keep, both guarded:

- **Constant within entity.** `serving_entity_profile` collapses `hist_*` with
  `any_value()`, which is only correct because the shared rates model makes
  those values constant within an entity — the property that lets serving
  reproduce training values byte-for-byte. Pinned by
  `assert_serving_lookup_entities_constant`.
- **Deterministic.** The medians use exact `percentile_disc`, not
  `approx_quantiles`, which was measured returning four different values for
  the same median on identical data. `serving_typical_rotation` must stay
  exactly one row (`assert_serving_typical_rotation_singleton`) because serving
  pins every context-less prediction to it.

Serve-time ESTIMATES here carry `where is_training_row` — rule 12 of
`docs/leakage_discipline.md` now lives in these models rather than in Python.

Rationale and measurements: `docs/benchmarks/serving_preload_benchmark.md`.
