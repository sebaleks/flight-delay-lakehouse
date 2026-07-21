# ml/

Trains two models from the **gold wide-flat feature mart**
(`flight_delays_gold.ml_flight_features`) — **the same gold layer the
dashboard consumes; nothing is duplicated or recomputed here.** The mart owns
the leakage boundary (CLAUDE.md §9): historical rates are training-window-only
(smoothed toward the global, constant within an entity), weather is prior-day,
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

**Determinism, stated precisely:** results are **deterministic given a fixed
mart build**. The loader sorts rows on the mart's unique flight grain, so the
frame is canonical regardless of BigQuery read order, and repeated fits on a
fixed frame are bit-identical (measured, 5/5). Across mart *rebuilds*,
BigQuery's distributed float aggregation order can change last bits of the
hist_* values; histogram binning very likely absorbs that, but last-bit
stability across rebuilds has not been verified — the guarantee is
per-mart-build, not across rebuilds.

Model artifacts go to `ml/artifacts/` (git-ignored).

Run:
```
uv sync --extra ml
uv run --extra ml python -m ml.audit    # leakage audit alone
uv run --extra ml python -m ml.train    # audit + train + evaluate
```
