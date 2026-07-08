# ml/

Trains two models from the **gold wide-flat feature mart** (`gold/ml`).

- **Classification:** `ArrDel15` (delayed ≥15 min).
- **Regression:** `ArrDelayMinutes`.
- **Split:** time-based (train earlier dates, test later) — never random.
- **Leakage rule (CLAUDE.md §9):** predictors use **only** information knowable
  **before departure**. The mart already enforces this; the training code must
  not reintroduce at/after-departure columns as features.

Not implemented yet — planned modules:

| Module (planned)      | Responsibility                                            |
|-----------------------|----------------------------------------------------------|
| `data.py`             | Load feature mart from BigQuery (ADC), apply time split   |
| `train_classifier.py` | Fit + evaluate the `ArrDel15` classifier                  |
| `train_regressor.py`  | Fit + evaluate the `ArrDelayMinutes` regressor            |
| `evaluate.py`         | Metrics/plots on the held-out later period                |

Model artifacts go to `ml/artifacts/` (git-ignored). Install deps:
`uv sync --extra ml`.
