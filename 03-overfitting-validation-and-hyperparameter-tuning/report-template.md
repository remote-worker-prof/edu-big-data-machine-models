# Шаблон отчета по ЛР 03: Overfitting, Validation and Honest Hyperparameter Tuning

Оценивание выполняется по единому rubric: `../RUBRIC_TEMPLATE.md`.


## 1. Контекст
- ФИО автора работы:
- Группа:
- Дата выполнения:
- Используемая среда (OS, версия Python):

## 2. Candidate feature set из ЛР 01
- Какие candidate feature set пришли для `medical`:
- Какие candidate feature set пришли для `finance`:
- Почему в ЛР 03 эти наборы считаются гипотезами, а не готовым winner:

## 2.1 Какой feature set выбрала каждая модель
Заполните по `outputs/model_feature_set_decisions.csv`.

| Dataset | Model | Selected feature set | Train F1 | Validation F1 | F1 gap | Abs F1 gap | Причина выбора |
|---|---|---|---:|---:|---:|---:|---|
| medical | LogisticRegression |  |  |  |  |  |  |
| medical | RandomForest |  |  |  |  |  |  |
| finance | LogisticRegression |  |  |  |  |  |  |
| finance | RandomForest |  |  |  |  |  |  |

## 2.2 Глоссарий незнакомых терминов (обязательно)
- Ссылка на `study-notes/glossary.md`:
- Сколько новых терминов добавлено по ходу всей ЛР:
- Минимум 3 примера терминов и почему они были важны:

## 3. Где модель переобучается
Используйте `generalization_audit` и заполните таблицу.

| Dataset | Feature set | Model | Train F1 | Validation F1 | F1 gap | Train ROC-AUC | Validation ROC-AUC | ROC-AUC gap | Краткий вывод |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| medical |  |  |  |  |  |  |  |  |  |
| finance |  |  |  |  |  |  |  |  |  |

### Что изучено по ходу выполнения (обязательно)
- Где вы увидели самый заметный `generalization gap`:
- В каком случае train-метрика оказалась слишком оптимистичной:
- Где validation получилась не хуже train и почему это не является ошибкой само по себе:
- Ссылки на источники (URL и/или `study-notes/*.md`):
- Какие термины из `study-notes/glossary.md` использовали:

## 4. Что показали validation curves
Используйте `validation_curve_results` и кратко заполните таблицу.

| Dataset | Model | Feature set | Hyperparameter | Лучшее значение по validation F1 | Что происходило при слишком слабой настройке | Что происходило при слишком сильной настройке |
|---|---|---|---|---|---|---|
| medical | LogisticRegression |  | C |  |  |  |
| medical | RandomForest |  | max_depth |  |  |  |
| finance | LogisticRegression |  | C |  |  |  |
| finance | RandomForest |  | max_depth |  |  |  |

### Что изучено по ходу выполнения (обязательно)
- Какой график оказался самым понятным для объяснения переобучения:
- Где изменение параметра реально улучшало validation-метрику, а где уже вредило:
- Ссылки на источники (URL и/или `study-notes/*.md`):
- Какие термины из `study-notes/glossary.md` использовали:

## 5. Что выбрал GridSearchCV
Используйте `gridsearch_results_top` и кратко опишите лучшие конфигурации.

| Dataset | Model | Feature set | Лучшая конфигурация | Mean CV F1 | Mean CV ROC-AUC | Краткий комментарий |
|---|---|---|---|---:|---:|---|
| medical | LogisticRegression |  |  |  |  |  |
| medical | RandomForest |  |  |  |  |  |
| finance | LogisticRegression |  |  |  |  |  |
| finance | RandomForest |  |  |  |  |  |

### Что изучено по ходу выполнения (обязательно)
- Почему поиск параметров велся только на `train`:
- Почему `Pipeline` помогает избежать leakage только тогда, когда все обучаемые preprocessing steps находятся внутри CV:
- Ссылки на источники (URL и/или `study-notes/*.md`):
- Какие термины из `study-notes/glossary.md` использовали:

## 6. Baseline vs Tuned на test
Используйте `baseline_vs_tuned_test_results` и заполните таблицу.

| Dataset | Model | Feature set | Variant | Accuracy | F1 | ROC-AUC | Fit time (sec) | Краткий вывод |
|---|---|---|---|---:|---:|---:|---:|---|
| medical |  |  | baseline_default |  |  |  |  |  |
| medical |  |  | tuned_best |  |  |  |  |  |
| finance |  |  | baseline_default |  |  |  |  |  |
| finance |  |  | tuned_best |  |  |  |  |  |

### Что изучено по ходу выполнения (обязательно)
- Насколько tuned-модель реально лучше baseline на `test`:
- Где более сложная настройка не дала ожидаемого выигрыша:
- Почему tuned-модель может не выиграть у baseline на `test`, даже если отбор был честным:
- Ссылки на источники (URL и/или `study-notes/*.md`):
- Какие термины из `study-notes/glossary.md` использовали:

## 7. Практическая рекомендация
- Какую модель вы рекомендуете для `medical` и почему:
- Какую модель вы рекомендуете для `finance` и почему:
- Какой feature set оказался наиболее удачным для каждой модели:
- Как бы вы объяснили свой выбор человеку, который смотрит только на train-метрику:

## 8. Проверка понимания
Кратко ответьте (3-5 предложений на пункт):
1. Почему нельзя выбирать гиперпараметры по `test`?
2. Почему train-метрика часто, но не всегда выше, чем validation?
3. Почему `Pipeline` внутри CV полезен, но не магически гарантирует отсутствие leakage сам по себе?
4. Почему более сложная модель или более долгий тюнинг не всегда побеждают на `test`?
5. Какое didactic shortcut есть в этой ЛР и как вы бы усилили его на продвинутом треке?

## 9. Что бы вы улучшили в следующей итерации
- Какие эксперименты вы бы добавили на расширенном треке.