# Лабораторная работа 02: Model Interpretability and Explainability

## О чем эта работа
Эта лабораторная продолжает ЛР 01: после отбора признаков и сравнения моделей
нужно научиться объяснять, почему модель принимает решения и где эти объяснения
согласуются или расходятся с качеством.

## Формат
- 2 обязательных Jupyter-ноутбука.
- Те же 2 прикладных бинарных датасета, что и в ЛР 01:
  - медицина: прогноз сердечно-сосудистого риска;
  - финансы: прогноз кредитного риска.
- Локальный запуск на CPU без тяжелых explainability-библиотек.
- Основа для feature set берется из артефактов ЛР 01.

## Зависимости от ЛР 01
Новая ЛР ожидает, что доступны артефакты из:
- `../01-feature-importance-and-selection/outputs/feature_sets_wrapper_embedded.json`
- `../01-feature-importance-and-selection/outputs/model_results.csv`

Если ЛР 01 уже была пройдена по текущему репозиторию, ноутбуки подхватят
эти файлы автоматически.

## Вход Для Новичка (Перед Практикой)
- Что уже нужно знать:
  - разницу между `train` и `test`;
  - базовый смысл метрик `f1` и `roc_auc`;
  - как читать таблицу важности признаков.
- Что можно пропустить на первом проходе:
  - глубокую теорию интерпретируемости beyond notebook-подсказок;
  - расширенные эксперименты по устойчивости объяснений.
- Какие артефакты должны лежать на диске:
  - `../01-feature-importance-and-selection/outputs/feature_sets_wrapper_embedded.json`;
  - `../01-feature-importance-and-selection/outputs/model_results.csv`.

## Структура папки
- `notebooks/` — шаблоны заданий с TODO и обязательными самостоятельными блоками.
- `solutions/` — решения базового маршрута без ответов на самостоятельные блоки.
- `study-notes/` — заметки и глоссарий по методам интерпретации.
- `outputs/` — промежуточные и итоговые таблицы.
- `report-template.md` — шаблон итогового отчета.
- `requirements.txt` — зависимости.
- `lab_utils.py` — общие утилиты для двух ноутбуков.

## Ноутбуки и порядок прохождения
1. `notebooks/01_global_explanations_todo.ipynb` (90 минут)
   - выбор лучшего неполного feature set из ЛР 01;
   - глобальная интерпретация `LogisticRegression` и `RandomForestClassifier`;
   - сравнение native importance и `permutation importance`;
   - one-way partial dependence для числовых признаков.
2. `notebooks/02_local_error_analysis_todo.ipynb` (90 минут)
   - выбор лучшей пары `model + feature_set` из ЛР 01;
   - локальные объяснения ошибок через perturbation-анализ;
   - разбор false positive / false negative;
   - сегментный анализ ошибок и практические рекомендации.

## Что сдавать
- заполненные ноутбуки с выполненными ячейками и выводами;
- обязательные CSV-артефакты из самостоятельных блоков;
- narrative-блоки по самостоятельному изучению методов;
- обновленный `study-notes/glossary.md`:
  минимум 3 новых термина на каждый ноутбук;
- отчет по шаблону `report-template.md`.

## Формат промежуточных таблиц
`global_importance_comparison`:
- `dataset`, `model`, `feature_set`, `method`, `feature`, `score`, `rank`

`partial_dependence_summary`:
- `dataset`, `model`, `feature_set`, `raw_feature`
- `grid_min`, `grid_max`, `score_min`, `score_max`, `score_delta`, `trend`

`error_case_explanations`:
- `dataset`, `model`, `feature_set`, `case_group_index`, `error_type`
- `y_true`, `y_pred`, `score`, `score_source`
- `explanation_method`, `feature`, `importance_value`, `detail_a`, `detail_b`

## Обязательные самостоятельные блоки
- экспорт `outputs/global_importance_comparison.csv`
- экспорт `outputs/partial_dependence_summary.csv`
- экспорт `outputs/error_case_explanations.csv`
- обновление narrative-блоков и `study-notes/glossary.md`

## Запуск
Команды выполняются из папки `02-model-interpretability-and-explainability`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name interpretability-lab --display-name "Python (.venv) Interpretability Lab"
.venv/bin/python -m jupyter notebook
```

Опционально можно использовать `source .venv/bin/activate` и запускать команды через `python`.

## Проверка Для Преподавателя/Разработчика
Из папки ЛР 02:

```bash
.venv/bin/python scripts/verify_lab02.py
```

Скрипт исполняет `solution`-ноутбуки и проверяет структуру `todo/solution`, upstream-контракт с ЛР 01 и CSV-артефакты в `outputs/`.

## Предсдача Для Студента
```bash
.venv/bin/python ../scripts/check_submission.py --lab 02
```

## Что Делать, Если Не Запускается
1. Ошибка `ModuleNotFoundError`:
   - выполните `.venv/bin/python -m pip install -r requirements.txt`.
2. Ошибка про отсутствующий upstream-файл из ЛР01:
   - выполните `.venv/bin/python ../01-feature-importance-and-selection/scripts/verify_lab01.py`;
   - или заново выполните export-ячейку в ЛР01.
3. Запустили из неверной директории:
   - команды для ЛР02 выполняйте из папки `02-model-interpretability-and-explainability`.
4. Для полной диагностики среды из корня репозитория:
   - `.venv/bin/python scripts/doctor_env.py`.

## Траектории Прохождения
- Легкий трек (минимально обязательный маршрут):
  - пройти оба `todo`-ноутбука и закрыть все `TODO(обязательно)`;
  - сформировать 3 обязательных CSV в `outputs/`;
  - заполнить отчет и базовый глоссарий.
- Углубленный трек:
  - сравнить объяснения на `full` против выбранного feature set;
  - проверить, как меняются локальные объяснения при разных `random_state`;
  - сделать отдельный decision memo: где лучше объяснимость, где лучше метрики.
