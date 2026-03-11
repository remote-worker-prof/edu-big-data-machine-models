# Outputs

Сюда ноутбуки сохраняют промежуточные и итоговые таблицы:
- `feature_ranking_filter_methods.csv` — ранги признаков из filter-методов.
- `shortlist_filter.json` — shortlist после filter-этапа.
- `feature_ranking_wrapper_embedded.csv` — ранги признаков из wrapper/embedded-методов.
- `feature_sets_wrapper_embedded.json` — candidate feature sets (`set_A`, `set_B`, `set_C` и т.д.).
- `model_results.csv` — итоговое сравнение моделей.

Обязательные артефакты самостоятельных блоков:
- `filter_stability_grid.csv` — устойчивость shortlist по сетке параметров filter-отбора.
- `method_agreement_long.csv` — согласованность методов отбора (pairwise overlap/Jaccard).
- `selection_stability.csv` — стабильность признаков по random_state/ресемплингу.
- `threshold_tuning_results.csv` — результаты тюнинга порога классификации.
- `cv_stability_results.csv` — проверка стабильности по кросс-валидации.
- `error_by_segment.csv` — сегментный анализ ошибок.
