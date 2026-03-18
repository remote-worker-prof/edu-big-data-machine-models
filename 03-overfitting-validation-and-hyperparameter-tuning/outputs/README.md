# Outputs

Сюда ноутбуки сохраняют промежуточные и итоговые таблицы:
- `generalization_audit.csv` — сравнение качества на `train` и `validation` для базовых моделей.
- `validation_curve_results.csv` — результаты простых validation curves по одному параметру на модель.
- `gridsearch_results_top.csv` — топ-конфигурации из `GridSearchCV` для каждой пары `dataset + model`.
- `baseline_vs_tuned_test_results.csv` — финальное сравнение `baseline_default` и `tuned_best` на `test`.

Локально в ноутбуках также могут формироваться вспомогательные DataFrame
для графиков и narrative-анализа, но в git они не требуются.
