# Шаблон отчета по ЛР 01: Feature Importance and Selection

## 1. Контекст
- ФИО студента:
- Группа:
- Дата выполнения:
- Используемая среда (OS, версия Python):

## 2. Данные и постановка
- Какой таргет предсказывается в медицинском датасете:
- Какой таргет предсказывается в финансовом датасете:
- Какие признаки оказались наиболее интуитивно важными до эксперимента (гипотеза):

## 3. Сравнение методов значимости признаков
Для каждого датасета кратко заполните таблицу.

| Dataset | Метод | Топ-5 признаков | Краткий комментарий |
|---|---|---|---|
| medical | VarianceThreshold |  |  |
| medical | Correlation |  |  |
| medical | Mutual Information |  |  |
| medical | ANOVA F-test |  |  |
| medical | RFE |  |  |
| medical | SFS |  |  |
| medical | L1 |  |  |
| medical | RF Importance |  |  |
| medical | Permutation |  |  |
| finance | VarianceThreshold |  |  |
| finance | Correlation |  |  |
| finance | Mutual Information |  |  |
| finance | ANOVA F-test |  |  |
| finance | RFE |  |  |
| finance | SFS |  |  |
| finance | L1 |  |  |
| finance | RF Importance |  |  |
| finance | Permutation |  |  |

## 4. Влияние отбора признаков на качество моделей
Используйте `model_results` и заполните таблицу с лучшими результатами.

| Dataset | Feature set | Model | Accuracy | F1 | ROC-AUC | Fit time (sec) |
|---|---|---|---:|---:|---:|---:|
| medical |  |  |  |  |  |  |
| finance |  |  |  |  |  |  |

## 5. Интерпретация
- Какие признаки стабильно важны для обоих подходов (filter/wrapper/embedded)?
- Где отбор признаков дал прирост метрик, а где ухудшил результат?
- Как изменилось время обучения после уменьшения числа признаков?

## 6. Практическая рекомендация
- Финальный рекомендуемый feature set для medical:
- Финальный рекомендуемый feature set для finance:
- Аргументация (метрики + интерпретируемость + скорость):

## 7. Обязательные самостоятельные задания (без образца в solutions)
Заполните краткие итоги по дополнительным обязательным блокам:

### 7.1 Устойчивость filter-ранжирования
- Как менялись shortlist при разных `variance threshold` и `top_n`?
- Какие конфигурации дали наибольший overlap/Jaccard?
- Файл: `outputs/filter_stability_grid.csv`.
- Минимальные колонки: `dataset`, `variance_threshold`, `top_n`, `shortlist_json`, `overlap_with_baseline`.

### 7.2 Согласованность wrapper/embedded методов
- Какой уровень согласованности между `rfe`, `sfs_forward`, `l1_logreg`, `rf_importance`, `permutation`?
- Какие признаки вошли в `set_D_robust` и почему?
- Файлы: `outputs/method_agreement_long.csv`, `outputs/selection_stability.csv`.
- Минимальные колонки:
  - `method_agreement_long`: `dataset`, `method_a`, `method_b`, `top_k`, `overlap_count`, `jaccard`.
  - `selection_stability`: `dataset`, `method`, `feature`, `selected_count`, `total_runs`, `stability_rate`.

### 7.3 Порог, CV и сегментный анализ ошибок
- Что изменилось после тюнинга порога у лучшей пары `dataset+model`?
- Насколько стабилен финальный feature set по CV?
- В каких сегментах (например, `age`, `credit_score`) ошибок больше всего?
- Файлы: `outputs/threshold_tuning_results.csv`, `outputs/cv_stability_results.csv`, `outputs/error_by_segment.csv`.
- Минимальные колонки:
  - `threshold_tuning_results`: `dataset`, `model`, `feature_set`, `threshold`, `precision`, `recall`, `f1`.
  - `cv_stability_results`: `dataset`, `model`, `feature_set`, `fold`, `accuracy`, `f1`, `roc_auc`.
  - `error_by_segment`: `dataset`, `segment_feature`, `segment`, `n`, `error_rate`, `false_positive_rate`, `false_negative_rate`.

## 8. Проверка понимания
Кратко ответьте (3-5 предложений на пункт):
1. Почему важно делать отбор признаков только на train-части?
2. Почему разные методы значимости могут давать разные топы признаков?
3. Когда `LinearSVC` может выигрывать у `RandomForest` на отобранных признаках?

## 9. Что бы вы улучшили в следующей итерации
- Какие эксперименты вы добавили бы (или уже добавили) на расширенном треке.
