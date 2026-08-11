# Overnight model search — validation results

- fit rows searched on: **2,500,000** · val rows: **800,000**
- validation window starts **2024-05-06**
- `hist_*` re-derived on the fit window (rule 10): **yes**
- recompute verified against the mart: max rate diff **4.72e-16** over 8,144 entities
- **the held-out test set was NOT touched** — this is a validation ranking, not an adoption
- elapsed: 90.4 min

| rank | family | val PR-AUC | val ROC-AUC | params |
|---|---|---|---|---|
| 1 | catboost | 0.51460 | 0.72684 | `{'depth': 10, 'learning_rate': 0.05, 'l2_leaf_reg': 6, 'iterations': 1200}` |
| 2 | catboost | 0.51433 | 0.72663 | `{'depth': 8, 'learning_rate': 0.1, 'l2_leaf_reg': 3, 'iterations': 1200}` |
| 3 | catboost | 0.51285 | 0.72611 | `{'depth': 8, 'learning_rate': 0.1, 'l2_leaf_reg': 3, 'iterations': 600}` |
| 4 | catboost | 0.51225 | 0.72562 | `{'depth': 10, 'learning_rate': 0.05, 'l2_leaf_reg': 6, 'iterations': 600}` |
| 5 | ensemble_rank_blend | 0.51126 | 0.72700 | `{'members': ['catboost', 'extra_trees', 'hist_gbdt', 'lightgbm', 'random_forest', 'xgb_incumbent']}` |
| 6 | catboost | 0.51040 | 0.72444 | `{'depth': 6, 'learning_rate': 0.1, 'l2_leaf_reg': 3, 'iterations': 600}` |
| 7 | catboost | 0.51008 | 0.72449 | `{'depth': 8, 'learning_rate': 0.05, 'l2_leaf_reg': 6, 'iterations': 600}` |
| 8 | hist_gbdt | 0.50982 | 0.72362 | `{'max_depth': 8, 'learning_rate': 0.1, 'max_iter': 400, 'min_samples_leaf': 20}` |
| 9 | hist_gbdt | 0.50977 | 0.72320 | `{'max_depth': None, 'learning_rate': 0.1, 'max_iter': 400, 'min_samples_leaf': 20}` |
| 10 | random_forest | 0.50929 | 0.72627 | `{'n_estimators': 300, 'max_depth': 30, 'min_samples_leaf': 20, 'max_features': 'sqrt'}` |
| 11 | hist_gbdt | 0.50775 | 0.72244 | `{'max_depth': 12, 'learning_rate': 0.05, 'max_iter': 400, 'min_samples_leaf': 50}` |
| 12 | hist_gbdt | 0.50763 | 0.72196 | `{'max_depth': None, 'learning_rate': 0.05, 'max_iter': 400, 'min_samples_leaf': 100}` |
| 13 | random_forest | 0.50696 | 0.72576 | `{'n_estimators': 300, 'max_depth': 20, 'min_samples_leaf': 50, 'max_features': 'sqrt'}` |
| 14 | catboost | 0.50683 | 0.72222 | `{'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 1, 'iterations': 600}` |
| 15 | random_forest | 0.50606 | 0.72517 | `{'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 100, 'max_features': 0.3}` |
| 16 | extra_trees | 0.50182 | 0.72249 | `{'n_estimators': 300, 'max_depth': 30, 'min_samples_leaf': 20, 'max_features': 'sqrt'}` |
| 17 | xgb_incumbent | 0.50102 | 0.71609 | `{'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 6}` |
| 18 | xgb_incumbent | 0.49834 | 0.71239 | `{'n_estimators': 600, 'learning_rate': 0.05, 'max_depth': 8}` |
| 19 | xgb_incumbent | 0.49805 | 0.71269 | `{'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 8}` |
| 20 | extra_trees | 0.49626 | 0.72011 | `{'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 100, 'max_features': 0.3}` |
