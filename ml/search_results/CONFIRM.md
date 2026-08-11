# Held-out confirmation — CatBoost (validation-selected)

One-time test evaluation of the config chosen on validation. Not an adoption
gate; adopting on a test comparison re-selects on test (rule 7).

- classifier config (fixed in advance): `{'depth': 8, 'learning_rate': 0.05, 'l2_leaf_reg': 6, 'iterations': 600}`
- fit rows **16,678,880** (full fit window) · held-out test **3,561,782**
- test base rate **0.1969**
- Platt fit on the val slice **2024-05-06 .. 2024-06-30**, never on test

| metric | shipped XGBoost | CatBoost | delta |
|---|---|---|---|
| ROC-AUC | 0.7389 | 0.7362 | -0.0027 |
| PR-AUC | 0.4652 | 0.4623 | -0.0029 |
| Lift over the 0.1969 base rate | 2.36x | 2.35x | -0.01x |
| ECE (Platt) | 0.0170 | 0.0147 |  |
| Brier (Platt) | 0.135 | 0.1353 |  |

_Regressor deliberately NOT run — classifier-only was requested, so the run was
stopped once this block was written. The deck keeps the shipped regressor's
49.26 RMSE / 18.99 MAE, and no CatBoost regression claim is made._

Raw (uncalibrated) CatBoost ECE **0.2350**, Brier **0.1948** —
both models need the Platt step; it is a monotonic remap, so ROC/PR-AUC are
unchanged by it.

**Verdict on the classifier: CatBoost does NOT beat the shipped model on held-out PR-AUC.**

On validation THIS config scored 0.51285 against the shipped XGBoost config's
0.49805 — **+0.0148**. On the held-out test it is **-0.0029**. A swing of ~0.018,
which is validation optimism and nothing else: the two models saw identical
rows, identical features and identical splits at every stage.

That makes it the SECOND time in this project that a validation win failed to
transfer — the first was a challenger that led by +0.0025 and regressed to
ROC 0.7373 / PR-AUC 0.4646 (ml/README.md). Two for two is the argument for
rule 7: had either been adopted on its validation margin, the shipped model
would be measurably worse.

Note also that CatBoost calibrates BETTER here (ECE 0.0147 vs 0.0170) while
ranking slightly worse. Ranking quality and calibration quality are separable,
which is precisely why calibration is a separate monotonic step rather than
something trusted from raw model output.

The regressor is **untuned** — the search tuned the classifier only, so its
RMSE/MAE is a first look at the family, not a like-for-like against the tuned
shipped regressor.
