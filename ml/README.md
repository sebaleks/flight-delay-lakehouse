# ml/

Trains two models from the **gold wide-flat feature mart**
(`flight_delays_gold.ml_flight_features`) — **the same gold layer the
dashboard consumes; nothing is duplicated or recomputed here.**

> **Why gold and not silver/bronze**, plus the lineage diagram proving the
> analytical and ML consumers share one layer without duplication:
> [`docs/lakehouse_lineage.md`](../docs/lakehouse_lineage.md). Short form: gold
> is where the leakage boundary becomes a *build artifact* — three dbt tests can
> diff a built table, but nothing can test a promise made in Python — and it is
> what lets one definition of `hist_*` serve the dashboard and the model at once.

The mart owns
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
| `features.py`    | Canonical feature registry + forbidden-column mirror         |
| `audit.py`       | Pre-training leakage self-audit (hard gate; also standalone) |
| `data.py`        | Load the mart from BigQuery (ADC), typed, canonically sorted |
| `tuning.py`      | Stage 3 hyperparameter search (reproducible; regressor tuned) |
| `train.py`       | Split, fit both models, evaluate on held-out rows, artifacts |
| `calibration.py` | Stage 4 probability calibration of the classifier (Platt map) |
| `tracking.py`    | MLflow experiment tracking (GCS-backed artifacts; graceful) |
| `experiments.py` | Model-comparison harness ('try different models'; e.g. LightGBM) |
| `replay.py`      | Held-out replay: score never-seen flights, show prediction vs actual |

## Headline (cascade/rotation RESTRICTED + hourly weather, held-out Jul–Dec 2024)

| Metric | Hourly weather | Cascade (contaminated) | **Cascade RESTRICTED (shipping)** |
|---|---|---|---|
| XGB ROC-AUC | 0.6979 | 0.7397 | **0.7389** |
| XGB PR-AUC (headline) | 0.3893 | 0.4748 | **0.4652** |
| Regression RMSE / MAE | 51.70 / 20.22 | 49.56 / 19.00 | 49.71 / 19.10 → **49.26 / 18.99** (tuned) |
| Logreg ROC / PR-AUC | 0.6550 / 0.3310 | 0.6920 / 0.3998 | 0.6654 / 0.3382 |

**THE TAIL-SWAP RESTRICTION (resolved 2026-07; the shipping definition).**
BTS records the post-hoc OPERATED tail, so the rotation LINKAGE can itself
be a day-of outcome — same-day swaps restructure which legs chain together.
The gating experiment: rotation features present ONLY for
schedule-consistent links (91.95% consistent inbound, 3.93% clean first
leg; 4.12% swap-shaped → NULL). **89% of the cascade PR-AUC uplift
survived** (+0.0759 of +0.0855 over the hourly baseline); the ~11% that did
not was swap-linkage, with the mechanism verified at the feature level: the
no_inbound band's training delay rate fell **0.388 → 0.224** once
swap-shaped rows left it — the elevated rate was substantially a swap
fingerprint. The retrained model reorganized as a real-signal hypothesis
predicts: the CLEANED turnaround-band feature rose to #2 in importance
while operated-chain rotation-position (which absorbs swap restructuring)
fell #4 → #8. Production ships the restricted definition; **0.7389 /
0.4652 is the honest headline**.

**Honest logreg note:** the linear baseline retains little of the cascade
uplift under the restriction (0.3382 vs 0.3310 hourly) — it median-imputes
the 4.12% all-NULL rows and loses the band separation the cleaning
compressed. The surviving signal is largely tree-accessible (XGBoost
consumes the NULLs natively).

Every generation is a controlled comparison: identical row set (20,240,662),
identical `is_training_row` split (16,678,880 / 3,561,782), identical
hyperparameters — each delta is attributable to its feature change alone. The
one exception is the arrow on the regressor row: the value after it is the
Stage 3 hyperparameter tuning below, not a feature change.
PR-AUC base rate is 0.1969 (lift 1.98→2.36 restricted).

**Stage 3 — hyperparameter tuning.** The columns above hold hyperparameters at
the untuned defaults so each delta isolates a feature change. On the restricted
feature set, a time-based validation search — the last 8 weeks of the training
window (2024-05-06..2024-06-30), never the test set; reproducible in
`ml/tuning.py` — then tuned the two models independently over a curated grid
with early stopping. The **regressor adopts the tuned config** (`max_depth 12,
lr 0.04, min_child_weight 20, subsample 0.7, colsample_bytree 0.7`, 201 trees):
RMSE
**49.71 → 49.26**, MAE **19.10 → 18.99** on the same held-out test — the shipped
regressor headline. The **classifier keeps its defaults**: the same candidate
won on validation (+0.0025 PR-AUC) but REGRESSED on the held-out test
(ROC 0.7389 → 0.7373, PR-AUC 0.4652 → 0.4646) — validation-optimism from the
summer val-slice distribution (0.260 delay rate vs the test's 0.197) and the
documented full-window `hist_*` residual, not signal. **Both halves of this
split** — keep the classifier default, adopt the tuned regressor — were decided
on the held-out **test** comparison (test-informed; a documented, mild cross-run
deviation, held-out numbers intact — `docs/leakage_discipline.md` rule 7; the
rigorous form calls it on validation with test as confirmation), so **the
classifier headline stays 0.7389 / 0.4652**. The
hist_* residual is accepted deliberately, but NOT on the discredited
"common-mode" grounds: a fit-window recompute **measured** the leak is *not*
common-mode (it shifts val PR-AUC unequally — XGB +0.0008 vs LightGBM +0.0002),
so it can distort a ranking; it did not flip the current winner, but re-derive
fit-window rates for any closer/wider selection
(`docs/leakage_discipline.md` rule 10). The reported tuned-vs-untuned deltas are
on the leak-free test set. Both models retrain bit-identically (the regressor
pins `random_state`; the classifier's `subsample=1` default is unchanged).

**Stage 4 — probability calibration (classifier).** `scale_pos_weight` ≈ 3.75
buys recall on the ~1-in-5 positive rate but inflates the raw scores: they
*rank* well yet are not frequencies (raw test **ECE 0.227** — a flight scored
0.25 is delayed ~9% of the time). `ml/calibration.py` remaps them with a
monotonic calibrator fit on the **same 8-week validation slice** as Stage 3
(`2024-05-06..2024-06-30`, never the test set), leaving the model untouched.
Two maps are fit and both persisted; **serving ships Platt**, isotonic is kept
for offline analysis only:

| Test set | Brier | ECE | ROC-AUC Δ | PR-AUC Δ |
|---|---|---|---|---|
| raw | 0.19147 | 0.22689 | — | — |
| **Platt (shipped)** | **0.13513** | **0.01658** | **0 (exact)** | **0 (exact)** |
| isotonic (offline) | 0.13519 | 0.01967 | −3.3e-5 | −3.1e-3 |

**The invariant (hard-gated).** Calibration is a monotonic remap, so ROC/PR-AUC
must be unchanged. Platt is a strictly-monotonic sigmoid → it preserves the
complete ordering, so on the held-out test ROC and PR-AUC are **bit-unchanged**
(`Δ = 0`); `build_calibration` **fails the training run** if the served map
moves ROC or PR-AUC beyond `1e-6`. Isotonic is a step function whose ties
coarsen the ranking — harmless to ROC (−3.3e-5) but it moves `average_precision`
by −0.0031, which is exactly why it is **not** the serving map. Platt also wins
on Brier and ECE here (it transfers better across
the val→test base-rate gap, 0.260 → 0.197), so shipping the AUC-safe map costs
nothing. In-sample optimism from fitting on the (in-sample) validation slice
was measured against an out-of-sample fit and is negligible (test Brier 6e-5 /
ECE 5e-3). The calibrator fit is deterministic, so `metrics.json` and
`calibrator.joblib` stay bit-identical across rebuilds like the rest of the run.

**TreeSHAP / margin attribution note:** SHAP explains the **raw XGBoost margin**
(log-odds), which is upstream of the calibration map — SHAP values do not sum
to the calibrated `delay_probability`. Attribute the margin; read the
calibrated probability as the reported output.

**Morning vs evening (lift over prevalence, XGB), across generations:**

| Generation | Morning 5–11 | Evening 17–23 |
|---|---|---|
| Daily prior-day weather | 1.61× | 1.53× |
| Hourly at-departure weather | 1.94× | 1.71× |
| + Cascade/rotation (restricted, shipping) | **2.56×** | **1.95×** |

The hourly-weather generation showed mornings gaining most from
time-resolved weather (not staleness — the arithmetic runs the other way;
morning conditions are simply closer to the whole story before disruption
accumulates). The cascade generation then REFUTED its own pre-registered
hypothesis: we predicted cascade features would lift evenings most (evening
outcomes being cascade-dominated), and the table says mornings gained
nearly three times more. **The refutation stands on the clean model**
(restricted: morning Δlift +0.62 vs evening +0.24 over the hourly
baseline). The post-hoc reading — explicitly labeled as
interpretation after the fact — is that features pay off where the exposure
they measure VARIES, not where its effects dominate: by evening nearly
every tail is deep in rotation (uniform exposure, weak discrimination),
while mornings span the full first-leg-to-red-eye-turnaround gradient
(position-1 risk 14.2% vs position-6+ 29.4% in the training window).

**Determinism, stated precisely:** the headline is **reproducible across
mart rebuilds** — verified on the restricted mart TWO ways: the production
dbt rebuild (rotation chain + shared rates + `ml_flight_features`) followed
by a complete retrain reproduced `metrics.json` byte-identically against
the experiment run built through an entirely DIFFERENT construction path
(a passthrough-patch variant table), and again across a second full
rebuild+retrain. A final review-driven edge refinement (a prior leg whose
scheduled arrival is unknown classes as swap-shaped, not as a clean first
leg — ~6 rows in 20.7M) then received its own rebuild-pair verification:
two more independent full rebuild+retrains, byte-identical at
**ROC 0.7388902208 / PR-AUC 0.4651540494** — the pinned headline. (The
same protocol verified the cascade-, hourly- and daily-era headlines
before it, and repeated fits on a fixed frame were already 5/5
bit-identical — the loader's canonical sort removes read-order
sensitivity.) Precision of the claim: the rebuild stability is an
empirical result — the observed rebuilds reproduced the mart values to
the last bit — not a BigQuery contract about distributed aggregation
order; a future rebuild shifting last bits would move metrics within the
historical ±0.002 band, visible immediately against the pinned headline.

Model artifacts go to `ml/artifacts/` (git-ignored).

Run:
```
uv sync --extra ml
uv run --extra ml python -m ml.audit    # leakage audit alone
uv run --extra ml python -m ml.train    # audit + train + evaluate
```

## Held-out replay: prediction vs what actually happened

`ml/replay.py` is the demo counterpart to the API. The endpoint scores FUTURE
flights (real forecast, outcome unknowable yet); the replay scores flights from
the **held-out window**, where the outcome is known — so the prediction can be
put next to the truth. It scores through the same `coerce_feature_frame` and the
same Platt map as request serving, so a replayed probability is what the
endpoint would return for an identical vector.

```
uv run --extra ml python -m ml.replay --sample 200000 --limit 8
```

```
HELD-OUT REPLAY — 200,000 flights the model has never seen
  base rate     0.1981          ROC-AUC 0.7426 / PR-AUC 0.4697 (sample)

  CALIBRATION — does 'p' behave like a frequency?
    (0.0, 0.1]   62,061    mean p 0.066    actual 0.073
    (0.1, 0.2]   63,698           0.144           0.144
    (0.3, 0.5]   27,094           0.381           0.335
    (0.7, 1.0]    6,033           0.794           0.750

  TOP DECILE    0.574 actually delayed vs 0.198 base — 2.90x lift

  HOW EXTREME IS THE TOP? actual delay rate by cut
    top 10       ( 0.01%)   0.900        top 10,000   ( 5.00%)   0.696
    top 100      ( 0.05%)   0.940        top 20,000   (10.00%)   0.574
    top 1,000    ( 0.50%)   0.871        all                     0.198
```

**Two example blocks, deliberately.** "Top of the ranking" shows the highest-
scored flights — nearly all of which were in fact delayed, because that is the
extreme tail of the ranking and says more about the cut than about the model.
The cut curve above is printed precisely so that block cannot be read as an
accuracy claim: at the top 100 the actual rate is 0.94, at the top decile 0.574,
overall 0.198. "Across the range" is the representative view — a deterministic
sample from each predicted band, so misses appear next to hits (a flight scored
0.68 that left on time, one scored 0.43 that arrived 204 minutes late).

Sample metrics run slightly above the pinned full-test headline (0.7389 /
0.4652) because they are a 200k deterministic sample, not the 3,561,782-row
test set — the output labels them as such and the pinned numbers remain the
ones to quote.

**The caveat to state whenever these numbers are shown.** The replay uses the
**observed** weather in the mart, not a forecast, because `api.weather.gov`
serves only the current forecast — the forecast issued before a 2024 flight is
not retrievable, and routing a past date through `/predict` returns
`has_origin_weather=false` with all twelve weather features dropped (verified).
So this is the **test-set regime**: live serving substitutes forecasts for
observations (gap #1 above) and will be somewhat worse. Quantifying that gap
would need archived NDFD grids for the test window — a real ingestion job, and
the natural next experiment.

Nothing here fits, tunes, or selects: it scores the one shipped model and
reports, which is the diagnostic-report case rule 7 of
[`../docs/leakage_discipline.md`](../docs/leakage_discipline.md) permits.

## Experiment tracking (MLflow) + trying different models

Every `ml.train` run is logged to **MLflow** (`ml/tracking.py`): hyperparameters,
held-out metrics (classifier ROC/PR-AUC, regressor RMSE/MAE, calibration
Brier/ECE, baselines), and the whole artifacts directory. **GCS-backed
artifacts, local metadata backend** — run metadata lives in a git-ignored SQLite
DB (`MLFLOW_TRACKING_URI`, default `sqlite:///mlflow.db`; MLflow can't keep
metadata in GCS without a standing server), while artifacts (models, `metrics.json`) go to
`gs://$GCS_BUCKET/mlflow`, reusing the bronze bucket. Tracking is a **pure side
effect** — it reads what the pipeline already computed and never touches the
fits or `metrics.json`, so determinism is unaffected — and it **degrades to a
warning** if MLflow/GCS is unreachable (a tracking outage never fails a run).
Disable with `MLFLOW_TRACKING=off`. Browse runs with `mlflow ui`.

```
uv run --extra ml python -m ml.experiments   # XGBoost vs LightGBM, logged
uv run --extra ml mlflow ui --backend-store-uri sqlite:///mlflow.db   # compare runs
```

`ml.experiments` fits alternative **classifiers** on the *identical* split and
`FEATURES` (only the learner changes — same pre-departure boundary, CLAUDE.md
§9) and logs each for an apples-to-apples comparison; the first alternative is
**LightGBM**. The shipped classifier stays `ml.train`'s XGBoost until an
alternative **wins the validation selection** against it (the held-out test is a
one-time confirmation, NOT the adoption gate — adopting on a test comparison
re-selects on test; see Stage 3 and `docs/leakage_discipline.md` rule 7).
**Expectations, stated honestly:** the classifier is on a flat plateau (Stage 3:
six configs spanned val PR-AUC 0.514–0.518, the tuned candidate *regressed* on
test), so `0.7389 / 0.4652` is a **signal ceiling** of leak-free pre-departure
features, not a model-capacity limit — a model-family swap reshuffles the last
~0.002. The levers that move the number are new leak-free features or a
different (real-time) regime, not a bigger learner (`blog_material.md` ch. 5/25).

## Inference endpoint (FastAPI): scoring FUTURE flights with real forecasts

```
uv run --extra ml --extra serve --extra ingestion uvicorn ml.api:app --port 8000
# POST /predict            one flight    POST /predict/batch    many
# GET  /demo/ord-departures?target_date=YYYY-MM-DD   proxy batch demo
```

**The serving lookup layer (added with the preload).** `build_context()` reads
three tiny gold tables once at startup — `serving_entity_profile` (8,316 rows:
the constant-within-entity `hist_*` values at all six grains, plus route
distance), `serving_density_profile` (34,979 rows), and
`serving_typical_rotation` (1 row) — into plain dicts. The request path then
issues **zero** BigQuery queries. It previously ran 5-6 un-prunable aggregates
over the 20.2M-row mart per call: 2.31 GB and ~5.1 s for a single flight,
regardless of batch size. Details, method and the parity evidence:
[`../docs/benchmarks/serving_preload_benchmark.md`](../docs/benchmarks/serving_preload_benchmark.md).
That work also fixed a real nondeterminism — the typical rotation profile was
built with `approx_quantiles`, observed returning four different values for the
same median, so two processes could score the same context-less request
differently (largest measured divergence 0.4300 vs 0.4968). The lookups use
exact `percentile_disc` and rebuild byte-identically.

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
prediction. **Output note:** `delay_probability` is **Platt-calibrated**
(Stage 4) — a delay frequency, not the raw recall-inflated score; each response
echoes `probability_calibration: "platt"`. Calibration is monotonic, so the
ranking is unchanged (ROC 0.7389 / PR-AUC 0.4652) while the served probability
is trustworthy (held-out ECE 0.017). `logreg_baseline_probability` stays
uncalibrated — a comparison anchor, not a shipped output.

## Deploying the predictor (Cloud Run)

The API is a **separate, private** Cloud Run service — deliberately not folded
into the dashboard image. That image carries only `--extra dashboard`; adding
xgboost plus ~695 MB of artifacts would put a model load on every cold start for
every visitor who only wanted the delay map, force its memory up for all
traffic, and let a bad artifact take the live BI app down. **`dashboard/` must
never `import ml`** — it talks to the predictor over HTTP.

```bash
# 1. build the outcome-mix table for the run (once per training run)
uv run --extra ml --extra ingestion python -m ml.exceedance

# 2. publish the pinned artifact run to GCS (immutable; refuses to overwrite)
uv run --extra ml --extra ingestion python -m ml.publish        # prints the _RUN id

# 3. build + deploy that exact run
gcloud builds submit --config cloudbuild.predictor.yaml \
  --substitutions=_RUN=<run-id>,_BUCKET=$GCS_BUCKET,_NWS_CONTACT=you@example.com
```

**Deliberate choices, and their costs:**

- **Manual deploy, not push-to-main.** Which model is served is a decision, not
  a consequence of merging. The dashboard keeps its automatic trigger; this does
  not.
- **Pinned run, no `latest` alias.** An image is tied to one artifact set, so a
  redeploy is reproducible and `/health`'s `artifacts` field actually identifies
  the model. Artifacts are immutable once published.
- **Private (`--no-allow-unauthenticated`).** The dashboard is public because it
  serves pre-aggregated read-only views; an open `/predict/batch` is a free
  compute amplifier. The dashboard's service account gets `roles/run.invoker`
  and mints an ID token via ADC — no key file.
- **`min-instances=0`.** Keeps the zero-idle-cost posture the rest of the
  project holds (`docs/compute_choice.md`) at the price of a **~20-40s cold
  start**, dominated by deserializing the boosters. The UI must show that
  explicitly rather than a spinner that looks broken.
- **Order matters:** the `serving_*` lookup tables must be built before a new
  image is deployed against a dataset. Startup fails loudly if they are absent.

### Endpoints

| Route | What it is |
|---|---|
| `GET /health` | liveness + the artifact run being served |
| `POST /predict`, `/predict/batch` | score flights; every response carries `prediction_basis` |
| `GET /calibration` | held-out reliability table (predicted vs actual, 10 bands, 3,561,782 rows) |
| `GET /outcome-mix` | held-out P(arrival delay ≥ t) per band, t ∈ {15,30,60,90,120} |
| `GET /demo/airport-departures?origin=ORD` | proxy-schedule batch demo for any airport |

**`prediction_basis`** is on every prediction and is the honest part of the
response: `flight_in_past` (the API still scores a past date — debugging is a
legitimate use — but a UI must refuse to present it as a forecast) and
`weather_horizon` ∈ `forecast` / `beyond_horizon` / `past` / `unavailable`,
which distinguishes the three ways the twelve weather features can be missing.
Previously a past-date request returned a confident-looking number with only a
server-side log.
