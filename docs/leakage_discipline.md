# Leakage & evaluation discipline

**This document is the authority** for how this repo prevents train/test leakage
and keeps held-out metrics honest — not any single source file. Every code path
that **reads the mart and fits or scores a model** must satisfy the checklist
below. When you add such a path, audit it here first; when a rule and a file
disagree, the checklist (and `ml/audit.py`) win — fix the file, not the rule.

Derived from the reference paths `ml/train.py`, `ml/tuning.py`, `ml/audit.py`,
and `ml/features.py`, and from the leakage rule in **CLAUDE.md §9**. The
warehouse side is pinned by three standing dbt tests
(`assert_ml_features_no_leakage`, `assert_ml_weather_obs_before_departure`,
`assert_ml_rotation_schedule_only`); this checklist governs the **Python** side.

_Last full audit: 2026-07-31 — all five mart-touching paths pass the applicable
rules WITHIN a run, with two documented cross-cutting caveats surfaced by a
re-review: (1) the final config selection was test-informed across runs, and
(2) the `hist_*` values on a training-window validation slice are leaky and not
common-mode (measured — did not flip the current selection). See the table and
rules 7 & 10._

---

## The checklist

### A. Audit gate & feature registry

1. **The leakage self-audit is a hard gate before any model fits or scores.**
   `run_audit(bq, dataset)` (`ml/audit.py`) raises `LeakageAuditError` on any
   failure and must be called on every path that trains/selects a model.
   *Ordering, stated precisely:* `run_audit` needs the BigQuery client returned
   by `load_mart()`, so the pipeline order is **`load_mart()` → `run_audit()` →
   fit**. This is safe — and NOT a violation of "audit before you read data" —
   because `load_mart()` (`ml/data.py`) `SELECT`s **exactly** the audited
   columns (`f.FEATURES` + `flight_date` + `SPLIT_COL` + `f.LABELS` + sort keys),
   never `SELECT *`, so an un-audited or forbidden mart column can never enter
   the frame, and the audit still gates before any *fit*. A new path must keep
   both properties: audit before fitting, and load only audited columns.

2. **The feature registry excludes labels, post-departure outcomes, and
   bookkeeping.** `run_audit` fails if `f.FEATURES` contains any `label_*`
   column, any `f.LABELS`, any name in `f.FORBIDDEN_FEATURES` (post-departure /
   arrival outcomes — `dep_delay`, `arr_delay`, `taxi_out`, prior-leg *actuals*,
   …), or anything in `f.EXCLUDED`.

3. **The live mart schema must equal the audited allowlist.** `run_audit`
   queries `INFORMATION_SCHEMA` and fails if the live mart columns differ from
   `f.MART_COLUMNS` or if any forbidden outcome column is present — so a mart
   change cannot silently feed the models something un-audited.

### B. Split & partition

4. **The split comes from `SPLIT_COL` (`is_training_row`), never re-derived from
   dates.** `train = df[f.SPLIT_COL]`, `test = ~train`. Never recompute the
   train/test boundary from `flight_date` in Python.

5. **The partition is proven clean and time-ordered.** `split_report`
   (`ml/train.py`) asserts `overlap == 0`, that the masks sum to the total, and
   `train_date_max < test_date_min`. `carve` (`ml/tuning.py`) additionally
   asserts `fit_max < val_min` **and** `train_max < test_min` (checking
   `train.max`, not just `val.min`, closes the drift hole where a mislabeled
   test-window row would land in the validation slice).

### C. Selection & the test set

6. **Model / hyperparameter / model-family SELECTION happens on a validation
   slice carved from INSIDE the training window — never on the test set.**
   `carve(df)` returns `(train, fit, val, test)` where `fit` = training rows
   before `VAL_START` and `val` = the last 8 weeks of training. Candidates are
   fit on `fit` and scored on `val`; the winner is chosen by the validation
   metric (classifier → val PR-AUC, regressor → val RMSE).

7. **The test set is scored exactly once, on the final selected model only.**
   After selection, retrain the winner on the FULL training window and score it
   on test a single time. Never re-select against the test set. (Scoring the
   *one* final model on test for multiple diagnostic **slices/reports** is fine;
   scoring *candidates* on test to pick one is not.)

   **Cross-invocation caveat (honest — do not claim unqualified PASS).** Rule 7
   holds *within a single run* of **`experiments.py`** (validation selects, only
   the winner is scored on test). `tuning.py`'s validation *search* is likewise
   clean, but `run_tuning` then scores the untuned baseline AND the tuned winner
   on test to make the keep/adopt call, so its within-run verdict is **WARN**
   (table). More broadly, the project's FINAL config choices were
   **test-informed across runs**: Stage 3 kept the
   classifier default vs adopted the tuned regressor by comparing each config's
   **test** score (the tuned classifier regressed on test → kept default; the
   tuned regressor improved on test → adopted), and `experiments.py`'s printed
   guidance **previously framed** a challenger's **test** score vs the incumbent
   `0.7389/0.4652` as the adoption criterion (**corrected in this PR** — see
   below). This is a **documented, mild** deviation — a
   two-way keep/adopt *decision*, not a search over many configs — and every
   reported number is still **held-out**: each config was scored on test exactly
   once and nothing was re-fit to improve a test number. The rigorous form is to
   make the keep/adopt call on the **validation** slice and treat the test score
   as a one-time confirmation report only; `experiments.py`'s printed guidance is
   corrected to do exactly that (recommend the validation winner; test is
   confirmation, not an adoption gate).

8. **`scale_pos_weight` is computed from the fitting rows in use** — the
   fit-set (`spw_fit`) for validation-stage fits, the full training window
   (`spw_full`) for the final/test fit — never from test rows.

### D. Feature provenance (mart-enforced, audit-asserted)

9. **All features are pre-departure-knowable by construction.** Enforced in the
   mart SQL + the three dbt guards, and re-asserted/logged by `run_audit`:
   `hist_*` = training-window rates smoothed toward the global (constant within
   an entity, identical train and test); origin weather = the last hourly ISD
   observation **at or before scheduled departure**; rotation features = SCHEDULE
   columns only, RESTRICTED to schedule-consistent linkages (swap-shaped
   operated-tail links → NULL). The prior leg's ACTUAL arrival delay is
   post-departure and is never an input.

10. **The `hist_*` values on a training-window validation slice are LEAKY — and
    the leak is NOT common-mode (measured, not assumed).** Because `hist_*`
    aggregate the whole pre-cutoff window, a val row's `hist_*` is computed
    partly from validation-period labels — `ml_flight_features.sql`'s own header
    says so: *"anyone carving a validation slice out of the training window must
    re-derive rates as-of that slice."* `tuning.py`/`experiments.py` carve such a
    slice and do **not** re-derive, so their validation-stage scores are
    optimistic. The prior "common-mode, doesn't distort the ranking"
    justification is **wrong**: different learners exploit the leak by different
    amounts.

    **Measured (2026-07-31).** A fit-window-only recompute of all 18 `hist_*`
    columns — using the exact mart formula `(n·rate_raw + 50·global)/(n+50)`,
    validated to reproduce the mart on the whole window to ~1e-8 — was run
    through the XGB-vs-LightGBM validation selection (fit on the fit-set, score
    on the val slice):

    | | XGBoost val PR-AUC | LightGBM val PR-AUC |
    |---|---|---|
    | leaky (mart, whole-window) | 0.514590 | 0.516557 |
    | clean (fit-window-only) | 0.513799 | 0.516361 |

    The leak inflates validation PR-AUC by **+0.00079 (XGBoost) vs +0.00020
    (LightGBM)** — a shift difference of ~0.0006, so it **can** distort the
    relative ranking. **On this selection it did not change the winner**:
    LightGBM wins both ways, and removing the leak *widens* its margin
    (0.0020 → 0.0026) because the leak had flattered XGBoost more. Verified
    immaterial to the current outcome — but for a closer comparison or a wider
    search it could matter, so the rigorous form remains to **re-derive rates
    as-of the validation cutoff** for any selection carved from the training
    window. The reported tuned-vs-untuned **test** deltas are unaffected (test
    `hist_*` never include test-window flights). The `ml/tuning.py` header's
    "common-mode" wording is corrected to point here.

### E. Calibration

11. **Probability calibration is fit on a training-window validation slice,
    never on the test set.** `build_calibration` (`ml/calibration.py`) fits the
    calibrator on `(p_val, y_val)`; it reads the raw test scores and test labels
    **only** for reporting and for the AUC-preservation gate (the gate raises if
    the served monotonic map moves ROC/PR-AUC on test) — it never *fits* on
    test. The in-sample optimism of fitting on the (in-sample) validation slice
    was measured and is negligible; documented in the module header.

### F. Serving / inference

12. **Serving uses only pre-departure inputs and never peeks at test-window
    data.** `ml/serving.py`/`ml/api.py` do not evaluate against a test set, so
    the selection rules (6–8) are N/A; instead: the weather FORECAST at the
    scheduled departure hour substitutes the observation training used; `hist_*`
    are read from the mart by explicit name (constant within an entity → the
    training value); and every serve-time ESTIMATE (departure density, the
    typical-rotation profile) is aggregated **`where is_training_row`** only, so
    the test window is never used to build a serving feature. The assembled
    frame's columns are asserted equal to `f.FEATURES` (and to each booster's
    stored schema) before any prediction, so a forbidden or drifted column
    cannot reach a model. Serving relies on the **train-time** audit + the dbt
    standing guards + these schema assertions rather than re-running `run_audit`
    itself (it does no fitting).

---

## Per-path audit (2026-07-31)

| Path | 1 audit gate | 4–5 split from `SPLIT_COL` | 6 select on val, not test | 7 test scored once, winner only | 8 spw from fit rows | Verdict |
|---|---|---|---|---|---|---|
| **`train.py`** (`run_training`) | ✅ `:198` (after `load_mart :195`, before any fit) | ✅ `:203`; `split_report :97` | — no selection (uses the tuned config from Stage 3) | ✅ scores clf/reg/logreg on test once each (`:266`, `:335`, `:242`); slices `:302` are post-hoc **reports** on the same final model | ✅ `spw` from `train_mask` `:253` | **PASS** |
| **`tuning.py`** (`run_tuning`) | ✅ `:228` | ✅ `carve :229` | ✅ `_search` fits on `fit`, scores `val`; test scoring **deferred** until after val selection `:247` | ⚠️ **WARN** — test scoring is deferred, but `run_tuning` then scores the **untuned baseline (`:257`) AND the tuned winner (`:283`)** on test and the keep/adopt split compares those test scores. So this is **not** "test scored once, winner only" — it is **test-informed config selection for BOTH models** (documented, mild: a two-way keep/adopt per model; rule 7) | ✅ `spw_fit :234` / `spw_full :235` | **WARN** (test-informed split; rule 7) |
| **`experiments.py`** (`compare_classifiers`) | ✅ `:124` **(fixed, PR #23 `18a78df`)** | ✅ `carve :128`; `split_report :125` | ✅ **(fixed, PR #24)** fit on `fit`, score `val` `:148`; winner = max val PR-AUC `:174` | ⚠️ within-run only the winner is scored on test `:177`, BUT the printed guidance **previously framed** a challenger's **test** score vs `0.7389/0.4652` as the adoption criterion — test-informed (documented, mild); **corrected in this PR** to recommend the val winner with test as confirmation only | ✅ `spw_fit :131` / `spw_full :132` | **PASS** (within-run; guidance fixed in this PR) |
| **`calibration.py`** (`build_calibration`) | via caller (`train.py` runs the audit) | N/A (receives masks-worth of scores) | N/A — ships the fixed Platt map, no model selection | reads test **only** for reporting + the AUC gate (`:206-224`); **fits on `p_val` only** `:204` | N/A | **PASS** |
| **`serving.py` / `api.py`** | N/A — no training/fitting | N/A | N/A | N/A | N/A | **PASS** (item 12: pre-departure inputs; estimates `where is_training_row`; schema gate) |

**Result: clean within-run, with two documented cross-cutting caveats** (added
after Codex's re-review of this doc; do not read the table as unqualified PASS):

1. **The FINAL config selection was test-informed** (rule 7 caveat). Stage 3's
   keep-classifier-default-vs-adopt-regressor-tuned decision compared candidates'
   **test** scores across runs (and `experiments.py`'s printed guidance did too,
   now corrected). Mild (a two-way keep/adopt decision, not a search) and the
   reported metrics stay held-out — but not an unqualified PASS. The
   `experiments.py` guidance is corrected (recommend the validation winner; test
   is a one-time confirmation, not an adoption gate).
2. **The `hist_*` val-slice leak is real and not common-mode** (rule 10). Fixed
   within-run selection now happens on the validation slice, but that slice's
   `hist_*` still carry whole-window (leaky) rates. Measured: it shifts XGBoost's
   validation PR-AUC ~4× more than LightGBM's (+0.00079 vs +0.00020) yet did
   **not** flip the current XGB-vs-LightGBM winner. Verified immaterial here;
   re-derive fit-window rates for any closer/wider future selection.

The gaps Codex originally found in `experiments.py` (no audit gate; scoring every
candidate on test) are **fixed and merged** — the audit gate in PR #23
(`18a78df`), the validation-selection + winner-only-on-test fix in **PR #24
(`5b9f765`)**. No mart-touching path has an *un*documented leak, and nothing
beyond `experiments.py` needed a code fix.

### Notes carried out of the audit

- **`run_audit` runs after `load_mart`, by construction** (it needs that
  client). Safe because `load_mart` selects only audited columns (rule 1). If a
  future change makes `load_mart` read more broadly (e.g. `SELECT *`), move the
  audit ahead of the read or re-audit the loaded frame.
- **Serving has no leakage self-audit of its own.** It reads `hist_*` by
  explicit name and hard-asserts the assembled schema against `f.FEATURES`, and
  the models it loads were trained under the audit — so a forbidden column
  cannot reach a model — but it leans on the mart's dbt guards for provenance
  rather than re-checking at startup. Acceptable given it does no fitting;
  revisit if serving ever computes a feature from raw, un-guarded mart columns.
- **`train.py` does no SELECTION** — it fits the config Stage 3 already selected
  (`CLASSIFIER_PARAMS` / `REGRESSOR_PARAMS`) and scores it on test once. That is
  why rule 6 is "—" and not a violation: there is no candidate search to leak.
