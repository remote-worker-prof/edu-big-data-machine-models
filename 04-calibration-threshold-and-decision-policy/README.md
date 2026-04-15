# Лабораторная работа 04: калибровка вероятностей и выбор порога под цену ошибок

## О чем эта работа
После ЛР 03 у нас уже есть выбранные модели и наборы признаков.
Следующий шаг для новичка: научиться принимать решение не только по «красивой метрике», но и по цене ошибок.

В этой ЛР вы:
- сравните некалиброванные и калиброванные вероятности;
- увидите, как калибровка влияет на вероятностные метрики;
- подберете порог `threshold` по ожидаемой стоимости ошибок;
- примените ограничение-страховку (guardrail) по полноте;
- выполните одну финальную проверку на `test` для уже выбранного правила решения (policy).

## Формат
- 1 теоретический ноутбук + 2 практических ноутбука.
- Те же 2 бинарных набора данных: `medical`, `finance`.
- Локальный запуск на CPU.
- Стек: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.
- В каждом ноутбуке обязательны графики и короткая математическая опора.

## Зависимости от ЛР 03
Входной контракт:
- `../03-overfitting-validation-and-hyperparameter-tuning/outputs/baseline_vs_tuned_test_results.csv`

Этот файл задает стартовую гипотезу `dataset -> model + feature_set` для ЛР 04.

## Контракт по данным
- `train`: обучение модели и калибровки;
- `validation`: выбор варианта калибровки и порога;
- `test`: одна финальная проверка только в ноутбуке 2 после выбора правила решения.

## Структура папки
- `theory-notebooks/` — теоретический ноутбук, который объясняет все приемы из ЛР 04.
- `notebooks/` — версии для студентов в формате пошагового заполнения (Guided Fill) (`TODO(обязательно)`).
- `solutions/` — версии с полным решением.
- `outputs/` — CSV-артефакты ЛР 04.
- `study-notes/` — заметки и глоссарий.
- `report-template.md` — шаблон итогового отчета.
- `lab_utils.py` — общие утилиты.
- `scripts/verify_lab04.py` — внутренний скрипт быстрой проверки.
- `tests/` — юнит-тесты `lab_utils.py`.

## Порядок прохождения
1. `theory-notebooks/01_theory_calibration_threshold_decision_policy.ipynb`
- цель: спокойно разобрать всю теорию перед практикой;
- внутри каждого раздела идет единый шаблон: «Идея -> Формула -> Мини-пример -> Как читать результат/график -> Где это в практическом ноутбуке».
2. `notebooks/01_calibration_basics_todo.ipynb`
- сравнение `uncalibrated`, `calibrated_sigmoid`, `calibrated_isotonic`;
- расчет `brier`, `log_loss`, `roc_auc`, `pr_auc`, `ece` на `validation`;
- анализ надежности вероятностей и обязательные графики;
- выбор `calibrated_best`.
3. `notebooks/02_threshold_policy_todo.ipynb`
- базовая модель стоимости ошибок `FP=1`, `FN=5`;
- перебор порога на `validation`;
- выбор правила решения по `min expected_cost` при `recall >= 0.60`;
- одна финальная проверка на `test` и сегментный аудит.

## Контракты артефактов
`calibration_audit.csv`:
- `dataset`, `model`, `variant`, `split`
- `brier`, `log_loss`, `roc_auc`, `pr_auc`, `ece`
- для ЛР 04: `split` только `validation`

`threshold_policy_grid.csv`:
- `dataset`, `model`, `variant`, `threshold`
- `precision`, `recall`, `f1`, `fp_rate`, `fn_rate`, `expected_cost`

`policy_test_report.csv`:
- `dataset`, `model`, `variant`, `policy_name`, `threshold`
- `accuracy`, `f1`, `roc_auc`, `pr_auc`, `expected_cost`, `cost_per_100`

`segment_policy_audit.csv`:
- `dataset`, `segment_feature`, `segment`, `n`
- `fp_rate`, `fn_rate`, `expected_cost_per_100`

## Что обязательно сделать студенту
- заполнить блоки `TODO(обязательно)` в обоих ноутбуках;
- сохранить все 4 CSV в `outputs/`;
- обновить `study-notes/glossary.md`;
- добавить минимум 1 заметку в `study-notes/*.md` со ссылками на источники.

## Запуск
Команды выполняются из папки `04-calibration-threshold-and-decision-policy`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name lab04-policy --display-name "Python (.venv) Lab 04"
.venv/bin/python -m jupyter notebook
```

Опционально можно использовать `source .venv/bin/activate` и запускать команды через `python`.

## Проверка для преподавателя
```bash
.venv/bin/python scripts/verify_lab04.py
```

Скрипт выполняет оба `solution`-ноутбука, проверяет теоретический ноутбук, структуру практики, графики, контракт по данным и CSV-артефакты.

## Предсдача Для Студента
```bash
.venv/bin/python ../scripts/check_submission.py --lab 04
```

## Что Делать, Если Не Запускается
1. Ошибка `ModuleNotFoundError`:
   - выполните `.venv/bin/python -m pip install -r requirements.txt`.
2. Ошибка про отсутствующий `baseline_vs_tuned_test_results.csv`:
   - выполните `.venv/bin/python ../03-overfitting-validation-and-hyperparameter-tuning/scripts/verify_lab03.py`;
   - или заново выполните export в ЛР03.
3. Ошибка про отсутствующий `feature_sets_wrapper_embedded.json`:
   - выполните `.venv/bin/python ../01-feature-importance-and-selection/scripts/verify_lab01.py`.
4. Для полной диагностики среды из корня репозитория:
   - `.venv/bin/python scripts/doctor_env.py`.

## Траектории Прохождения
- Легкий трек (минимально обязательный маршрут):
  - пройти теоретический ноутбук и два `todo`-ноутбука;
  - сформировать 4 обязательных CSV;
  - заполнить отчет и глоссарий.
- Углубленный трек:
  - сравнить разные наборы cost-коэффициентов (`FP/FN`) и их влияние на policy;
  - добавить альтернативные guardrail-ограничения и сравнить последствия;
  - провести расширенный сегментный аудит с дополнительными признаками сегментации.
