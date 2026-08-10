# Overnight model search — validation results

- fit rows searched on: **2,500,000** · val rows: **800,000**
- validation window starts **2024-05-06**
- `hist_*` re-derived on the fit window (rule 10): **yes**
- recompute verified against the mart: max rate diff **4.72e-16** over 8,144 entities
- **the held-out test set was NOT touched** — this is a validation ranking, not an adoption
- elapsed: 42.7 min

| rank | family | val PR-AUC | val ROC-AUC | params |
|---|---|---|---|---|
| 1 | catboost | 0.51474 | 0.72732 | `{'depth': 8, 'learning_rate': 0.1, 'l2_leaf_reg': 3, 'iterations': 600}` |
| 2 | catboost | 0.51455 | 0.72758 | `{'depth': 10, 'learning_rate': 0.05, 'l2_leaf_reg': 6, 'iterations': 600}` |
| 3 | catboost | 0.51238 | 0.72610 | `{'depth': 6, 'learning_rate': 0.1, 'l2_leaf_reg': 3, 'iterations': 600}` |
| 4 | catboost | 0.51210 | 0.72587 | `{'depth': 8, 'learning_rate': 0.05, 'l2_leaf_reg': 6, 'iterations': 600}` |
| 5 | hist_gbdt | 0.51137 | 0.72450 | `{'max_depth': None, 'learning_rate': 0.1, 'max_iter': 400, 'min_samples_leaf': 20}` |
| 6 | random_forest | 0.51134 | 0.72809 | `{'n_estimators': 300, 'max_depth': 30, 'min_samples_leaf': 20, 'max_features': 'sqrt'}` |
| 7 | hist_gbdt | 0.51116 | 0.72432 | `{'max_depth': 8, 'learning_rate': 0.1, 'max_iter': 400, 'min_samples_leaf': 20}` |
| 8 | hist_gbdt | 0.51023 | 0.72398 | `{'max_depth': None, 'learning_rate': 0.05, 'max_iter': 400, 'min_samples_leaf': 100}` |
| 9 | hist_gbdt | 0.50975 | 0.72366 | `{'max_depth': 12, 'learning_rate': 0.05, 'max_iter': 400, 'min_samples_leaf': 50}` |
| 10 | ensemble_rank_blend | 0.50925 | 0.72712 | `{'members': ['catboost', 'extra_trees', 'hist_gbdt', 'random_forest']}` |
| 11 | catboost | 0.50863 | 0.72388 | `{'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 1, 'iterations': 600}` |
| 12 | random_forest | 0.50854 | 0.72722 | `{'n_estimators': 300, 'max_depth': 20, 'min_samples_leaf': 50, 'max_features': 'sqrt'}` |
| 13 | random_forest | 0.50795 | 0.72660 | `{'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 100, 'max_features': 0.3}` |
| 14 | extra_trees | 0.50324 | 0.72370 | `{'n_estimators': 300, 'max_depth': 30, 'min_samples_leaf': 20, 'max_features': 'sqrt'}` |
| 15 | extra_trees | 0.49788 | 0.72142 | `{'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 100, 'max_features': 0.3}` |
| 16 | extra_trees | 0.49544 | 0.71938 | `{'n_estimators': 300, 'max_depth': 20, 'min_samples_leaf': 50, 'max_features': 'sqrt'}` |
