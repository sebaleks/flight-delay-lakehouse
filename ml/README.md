# ml/

Trains two models from the **gold wide-flat feature mart**
(`flight_delays_gold.ml_flight_features`) — **the same gold layer the
dashboard consumes; nothing is duplicated or recomputed here.** The mart owns
the leakage boundary (CLAUDE.md §9): historical rates are training-window-only
(smoothed toward the global, constant within an entity), weather is the last
hourly ISD observation **at or before scheduled departure** (3-hour staleness
ceiling, UTC observations joined to local schedule via the seed timezone),
and three standing dbt tests guard the boundary (schema allowlist, weather
obs-before-departure, rotation schedule-only). This package re-asserts the
contract at train time and fails hard if the mart schema drifts.

Lesson recorded from the deep dive: an earlier leave-one-out variant of the
historical rates created a target-encoding artifact — per-row perturbations
anti-correlated with the training label — that **handicapped the boosted
trees (it did not inflate metrics: test features never contained test
labels, so all reported numbers were honest throughout)**. The smoothed-rate
design removes the channel by construction.

- **Classification:** `label_arr_del15` (delayed ≥15 min) — logistic-regression
  baseline (class_weight='balanced') + XGBoost (scale_pos_weight, native
  categoricals). Headline metric: **PR-AUC** (~1-in-5 base rate makes accuracy
  nearly meaningless — the majority-class baseline is reported alongside).
- **Regression:** `label_arr_delay_minutes` — XGBoost vs a predict-train-mean
  baseline. RMSE + MAE.
- **Split:** STRICTLY the mart's `is_training_row` column (train = true,
  evaluate = false). Never re-derived from dates, never shuffled across the
  boundary; the trainer asserts the partition is exact and disjoint.

| Module        | Responsibility                                              |
|---------------|-------------------------------------------------------------|
| `features.py` | Canonical feature registry + forbidden-column mirror         |
| `audit.py`    | Pre-training leakage self-audit (hard gate; also standalone) |
| `data.py`     | Load the mart from BigQuery (ADC), typed, canonically sorted |
| `train.py`    | Split, fit both models, evaluate on held-out rows, artifacts |

## Headline (cascade/rotation + hourly weather, held-out Jul–Dec 2024)

| Metric | Hourly weather (previous) | + Cascade/rotation | Δ |
|---|---|---|---|
| XGB ROC-AUC | 0.6979 | **0.7397** | +0.0418 |
| XGB PR-AUC (headline) | 0.3893 | **0.4748** | +0.0855 (+22.0% rel.) |
| Regression RMSE | 51.70 | **49.56** | −2.14 |
| Regression MAE | 20.22 | **19.00** | −1.22 |
| Logreg ROC / PR-AUC | 0.6550 / 0.3310 | 0.6920 / 0.3998 | +0.0370 / +0.0688 |

Every generation is a controlled comparison: identical row set (20,240,662),
identical `is_training_row` split (16,678,880 / 3,561,782), identical
hyperparameters — each delta is attributable to its feature change alone.
The linear baseline moving too confirms the signal is in the features, not
a tree-specific artifact. PR-AUC base rate is 0.1969 (lift 1.98→2.41).

**Morning vs evening (lift over prevalence, XGB), across generations:**

| Generation | Morning 5–11 | Evening 17–23 |
|---|---|---|
| Daily prior-day weather | 1.61× | 1.53× |
| Hourly at-departure weather | 1.94× | 1.71× |
| + Cascade/rotation | **2.66×** | **1.98×** |

The hourly-weather generation showed mornings gaining most from
time-resolved weather (not staleness — the arithmetic runs the other way;
morning conditions are simply closer to the whole story before disruption
accumulates). The cascade generation then REFUTED its own pre-registered
hypothesis: we predicted cascade features would lift evenings most (evening
outcomes being cascade-dominated), and the table says mornings gained
nearly three times more. The post-hoc reading — explicitly labeled as
interpretation after the fact — is that features pay off where the exposure
they measure VARIES, not where its effects dominate: by evening nearly
every tail is deep in rotation (uniform exposure, weak discrimination),
while mornings span the full first-leg-to-red-eye-turnaround gradient
(position-1 risk 14.2% vs position-6+ 29.4% in the training window).

**Determinism, stated precisely:** the headline is **reproducible across
mart rebuilds** — verified on the cascade mart: a full dbt rebuild of the
rotation chain, the shared rates, and `ml_flight_features`, followed by a
complete retrain, reproduced `metrics.json` byte-identically
(ROC 0.7396687332 / PR-AUC 0.4748073992; the same protocol verified the
hourly-era and daily-era headlines before it, and repeated fits on a fixed
frame were already 5/5 bit-identical — the loader's canonical sort removes
read-order sensitivity). Precision of the claim: the rebuild stability is
an empirical result — the observed rebuilds reproduced the mart values to
the last bit — not a BigQuery contract about distributed aggregation order;
a future rebuild shifting last bits would move metrics within the
historical ±0.002 band, visible immediately against the pinned headline.

Model artifacts go to `ml/artifacts/` (git-ignored).

Run:
```
uv sync --extra ml
uv run --extra ml python -m ml.audit    # leakage audit alone
uv run --extra ml python -m ml.train    # audit + train + evaluate
```

## Inference endpoint (FastAPI): scoring FUTURE flights with real forecasts

```
uv run --extra ml --extra serve --extra ingestion uvicorn ml.api:app --port 8000
# POST /predict            one flight    POST /predict/batch    many
# GET  /demo/ord-departures?target_date=YYYY-MM-DD   proxy batch demo
```

**Why this is leak-free:** the pre-departure boundary (CLAUDE.md §9) requires
features knowable before departure. For a FUTURE flight, a weather forecast
issued now predates departure by construction. Training uses the last
OBSERVED hourly reading at or before the scheduled departure hour; serving
fetches the NDFD forecast valid AT that same hour (api.weather.gov:
official, keyless, covers the whole BTS territory).

**Train/serve mismatch, stated honestly — TWO gaps, both a consequence of
scoring flights that have not happened yet.** Training and serving reference
the SAME instant, the scheduled departure hour, so the daily-weather era's
prior-day-vs-flight-day time misalignment is gone. What remains:

1. **Weather is forecast-vs-observed.** Training features are OBSERVATIONS;
   serving features are NDFD FORECASTS of those same quantities (incl. a
   documented QPF-per-hour apportionment when NDFD issues multi-hour precip
   intervals). Missing forecast coverage (beyond the ~7-day horizon, off-grid
   points) reproduces the training NULL path exactly: weather NaN +
   `has_origin_weather=false`.
2. **Rotation context depends on what the caller knows.** The 15
   cascade/rotation features are SCHEDULE-derived (knowable at booking). A
   caller with the aircraft's planned rotation passes it per request; the demo
   passes its proxy schedule's historical rotation. A caller WITHOUT it takes
   the **typical rotation profile** (training medians), and origin departure
   density falls back to the training `(origin, hour, weekday)` median. Why
   estimate rather than NULL: under the tail-swap restriction an all-NULL
   rotation is in-distribution but MEANS "operated linkage was
   swap-restructured", so nulling a merely-unknown future plan would
   misclassify it as swap-shaped. The rotation block is **complete-or-absent**:
   once `rotation_position` is given, `legs_today` is required and (for
   `position >= 2`) the inbound leg is too — a partial context is rejected
   (422) rather than assembled into a shape training never produced. Each
   response reports its estimation honestly: `rotation_context` is `"provided"`
   (the caller supplied the whole rotation linkage) or `"typical_estimate"`
   (the median profile), and `origin_density_source` is `"provided"` or
   `"estimated"` for the separately-optional departure-density feature.

Neither gap changes the held-out test metrics — those were computed entirely
on observed data and stand as reported (ROC 0.7389 / PR-AUC 0.4652); they are
the price of usability on flights that have not departed.

Serving-time feature parity: hist_* rates are read from `ml_flight_features`
itself (constant within an entity — byte-exact training values; new entities
stay NaN), the turnaround-band and rotation-position hist values come straight
from the mart the same way, holiday flags use the same `holidays` library, and
the assembled frame is asserted against the models' stored schemas before any
prediction. Note on outputs: with `scale_pos_weight` ≈ 3.75 the classifier's
probabilities are recall-weighted (systematically higher than raw delay
frequencies) — treat them as a ranking score, not a calibrated frequency.
