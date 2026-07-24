# ml/

Trains two models from the **gold wide-flat feature mart**
(`flight_delays_gold.ml_flight_features`) — **the same gold layer the
dashboard consumes; nothing is duplicated or recomputed here.** The mart owns
the leakage boundary (CLAUDE.md §9): historical rates are training-window-only
(smoothed toward the global, constant within an entity), weather is the last
hourly ISD observation **at or before scheduled departure** (3-hour staleness
ceiling, UTC observations joined to local schedule via the seed timezone),
and two standing dbt tests guard the boundary. This package re-asserts the
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

## Headline (hourly weather at scheduled departure, held-out Jul–Dec 2024)

| Metric | Daily prior-day (previous) | Hourly at-departure | Δ |
|---|---|---|---|
| XGB ROC-AUC | 0.6806 | **0.6979** | +0.0173 |
| XGB PR-AUC (headline) | 0.3479 | **0.3893** | +0.0414 (+11.9% rel.) |
| Regression RMSE | 52.10 | **51.70** | −0.40 |
| Regression MAE | 20.67 | **20.22** | −0.45 |
| Logreg ROC / PR-AUC | 0.6464 / 0.3017 | 0.6550 / 0.3310 | +0.0086 / +0.0293 |

A controlled comparison: identical row set (20,240,662), identical
`is_training_row` split (16,678,880 / 3,561,782), identical hyperparameters —
the only change is the weather source, so the delta is attributable to it.
The linear baseline improving too confirms the signal is in the features,
not a tree-specific artifact. PR-AUC base rate is 0.1969 (lift 1.77→1.98).

**Morning vs evening (lift over prevalence, XGB):** evenings 1.53×→1.71×,
mornings 1.61×→**1.94×** — mornings gained about twice as much. This is NOT
a staleness story, and the staleness arithmetic runs the other way: a 07:00
departure sits ~7 h after the prior-day summary window closed, a 20:00
departure ~20 h, so evenings had the STALER prior-day weather and staleness
alone predicts the opposite of what we observe. The mechanism that fits:
**morning delays are weather-determined** — little disruption has accumulated
overnight, so conditions at the airport are close to the whole story, and
sharpening a day-old summary into a ~24-minute-old observation buys a lot;
**evening delays are cascade-determined** — the day's accumulated disruption
dominates, leaving departure-hour weather less to explain. This also explains
the old model's roughly flat lift across the day: without time-resolved
weather it could not exploit the weather-determined morning regime.

**Determinism, stated precisely:** the headline is **reproducible across
mart rebuilds** — verified on the hourly mart: a full dbt rebuild of
`ml_flight_features` followed by a complete retrain reproduced
`metrics.json` byte-identically (ROC 0.6979428331 / PR-AUC 0.3892655104;
the same protocol previously verified the daily-era headline, and repeated
fits on a fixed frame were already 5/5 bit-identical — the loader's
canonical sort removes read-order sensitivity). Precision of the claim: the
rebuild stability is an empirical result — the observed rebuilds reproduced
the mart values to the last bit — not a BigQuery contract about distributed
aggregation order; a future rebuild shifting last bits would move metrics
within the historical ±0.002 band, visible immediately against the pinned
headline.

Model artifacts go to `ml/artifacts/` (git-ignored).

Run:
```
uv sync --extra ml
uv run --extra ml python -m ml.audit    # leakage audit alone
uv run --extra ml python -m ml.train    # audit + train + evaluate
```

