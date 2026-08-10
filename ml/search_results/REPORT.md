# Overnight model search — validation results

- fit rows searched on: **120,000** · val rows: **60,000**
- validation window starts **2024-05-06**
- `hist_*` re-derived on the fit window (rule 10): **yes**
- recompute verified against the mart: max rate diff **4.72e-16** over 8,144 entities
- **the held-out test set was NOT touched** — this is a validation ranking, not an adoption
- elapsed: 6.2 min

| rank | family | val PR-AUC | val ROC-AUC | params |
|---|---|---|---|---|
| 1 | catboost | 0.50295 | 0.71946 | `{'depth': 8, 'learning_rate': 0.05, 'l2_leaf_reg': 6, 'iterations': 600}` |
| 2 | catboost | 0.50259 | 0.71884 | `{'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 1, 'iterations': 600}` |
| 3 | catboost | 0.49976 | 0.71644 | `{'depth': 6, 'learning_rate': 0.1, 'l2_leaf_reg': 3, 'iterations': 600}` |
| 4 | catboost | 0.49881 | 0.71564 | `{'depth': 10, 'learning_rate': 0.05, 'l2_leaf_reg': 6, 'iterations': 600}` |
| 5 | catboost | 0.49736 | 0.71306 | `{'depth': 8, 'learning_rate': 0.1, 'l2_leaf_reg': 3, 'iterations': 600}` |
| 6 | ensemble_rank_blend | 0.48551 | 0.71540 | `{'members': ['catboost', 'extra_trees', 'random_forest']}` |
| 7 | random_forest | 0.48472 | 0.71387 | `{'n_estimators': 300, 'max_depth': 30, 'min_samples_leaf': 20, 'max_features': 'sqrt'}` |
| 8 | random_forest | 0.48158 | 0.71316 | `{'n_estimators': 300, 'max_depth': 20, 'min_samples_leaf': 50, 'max_features': 'sqrt'}` |
| 9 | extra_trees | 0.47853 | 0.71015 | `{'n_estimators': 300, 'max_depth': 30, 'min_samples_leaf': 20, 'max_features': 'sqrt'}` |
| 10 | random_forest | 0.47809 | 0.71072 | `{'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 100, 'max_features': 0.3}` |
| 11 | extra_trees | 0.47071 | 0.70682 | `{'n_estimators': 300, 'max_depth': 20, 'min_samples_leaf': 50, 'max_features': 'sqrt'}` |
| 12 | extra_trees | 0.46983 | 0.70602 | `{'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 100, 'max_features': 0.3}` |

## Failures

- `hist_gbdt` {'max_depth': None, 'learning_rate': 0.1, 'max_iter': 400, 'min_samples_leaf': 20}: Categorical feature 'origin' is expected to have a cardinality <= 255 but actually has a cardinality of 364.
- `hist_gbdt` {'max_depth': 8, 'learning_rate': 0.1, 'max_iter': 400, 'min_samples_leaf': 20}: Categorical feature 'origin' is expected to have a cardinality <= 255 but actually has a cardinality of 364.
- `hist_gbdt` {'max_depth': 12, 'learning_rate': 0.05, 'max_iter': 400, 'min_samples_leaf': 50}: Categorical feature 'origin' is expected to have a cardinality <= 255 but actually has a cardinality of 364.
- `hist_gbdt` {'max_depth': None, 'learning_rate': 0.05, 'max_iter': 400, 'min_samples_leaf': 100}: Categorical feature 'origin' is expected to have a cardinality <= 255 but actually has a cardinality of 364.
