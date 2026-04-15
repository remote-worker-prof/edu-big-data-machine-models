# Лабораторная работа 03: Overfitting, Validation and Honest Hyperparameter Tuning

## О чем эта работа
Эта лабораторная отвечает на очень практичный вопрос:
почему модель может выглядеть отлично на обучающей выборке,
но заметно хуже работать на новых данных.

Здесь мы не "накручиваем метрику", а учимся улучшать модель честно:
- отделять `train`, `validation`, `test`;
- замечать переобучение по разрыву между качеством на обучении и проверке;
- подбирать гиперпараметры через `GridSearchCV`, не используя `test` для выбора;
- сравнивать candidate feature set заново на текущем split, а не доверять старому победителю.

ЛР 03 продолжает ЛР 01:
- берем те же датасеты `medical` и `finance`;
- используем candidate feature set из `../01-feature-importance-and-selection/outputs/feature_sets_wrapper_embedded.json`;
- относимся к этим наборам как к гипотезам, которые нужно переоценить в новом эксперименте.

ЛР 02 остается смысловым мостом:
мы уже учились объяснять поведение модели, а теперь учимся улучшать ее без самообмана.

## Формат
- 2 обязательных Jupyter-ноутбука.
- Те же 2 прикладных бинарных датасета:
  - медицина: прогноз сердечно-сосудистого риска;
  - финансы: прогноз кредитного риска.
- Локальный запуск на CPU.
- Только знакомый стек: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.
- Без nested CV, новых тяжелых библиотек и сложной теории ради теории.

## Зависимости от ЛР 01
Новая ЛР ожидает только один upstream-артефакт:
- `../01-feature-importance-and-selection/outputs/feature_sets_wrapper_embedded.json`

Если ЛР 01 уже была пройдена в текущем репозитории, ноутбуки подхватят этот файл автоматически.

## Вход Для Новичка (Перед Практикой)
- Что уже нужно знать:
  - базовое разделение `train/validation/test`;
  - что означает переобучение и почему train-метрика может быть оптимистичной;
  - как читать таблицу с метриками.
- Что можно пропустить на первом проходе:
  - nested CV и сложные схемы model selection;
  - расширенный анализ устойчивости при разных seeds.
- Какие артефакты должны лежать на диске:
  - `../01-feature-importance-and-selection/outputs/feature_sets_wrapper_embedded.json`.

## Data Usage Contract
В этой ЛР важно не только запустить код, но и честно разделить роли частей данных:
- `train`: fit baseline-моделей, validation curves и `GridSearchCV`;
- `validation`: выбор feature set для каждой модели, сравнение лучших tuned-конфигураций и выбор финального winner;
- `test`: один финальный `baseline_default` vs `tuned_best` check после всех решений.

Если tuned-модель не улучшает `test`, это не провал лабораторной.
Это нормальный результат, который показывает, что честная процедура важнее красивой цифры.

## Что Эта ЛР Упрощает
Базовый маршрут этой лабораторной специально упрощен, чтобы остаться понятным новичку:
- мы переиспользуем один и тот же `validation` для нескольких последовательных решений;
- мы не вводим nested CV в обязательную часть;
- мы не добавляем отдельный selection split сверх схемы `train/validation/test`.

Это didactic shortcut, а не production gold standard.
При этом базовый маршрут остается честным по отношению к `test`, потому что `test` не участвует в выборе feature set, гиперпараметров и финальной модели.
На продвинутом треке этот workflow можно усилить через nested CV или отдельный selection split.

## Структура папки
- `notebooks/` — шаблоны заданий с TODO и обязательными самостоятельными блоками.
- `solutions/` — учебные walkthrough-версии с тем же маршрутом, но с пояснениями и образцом reasoning.
- `study-notes/` — заметки и глоссарий по переобучению, валидации и тюнингу.
- `outputs/` — промежуточные и итоговые таблицы.
- `report-template.md` — шаблон итогового отчета.
- `requirements.txt` — зависимости.
- `lab_utils.py` — общие утилиты для обоих ноутбуков.

## Ноутбуки и порядок прохождения
1. `notebooks/01_train_validation_overfitting_todo.ipynb` (90 минут)
   - загрузка всех candidate feature set из ЛР 01;
   - сравнение `full` и всех неполных наборов на одном и том же split;
   - оценка `LogisticRegression` и `RandomForestClassifier` на `train` и `validation`;
   - вычисление `generalization gap`;
   - выбор feature set отдельно для каждой модели;
   - простые validation curves по одному гиперпараметру на модель.
2. `notebooks/02_gridsearch_and_final_choice_todo.ipynb` (90 минут)
   - чтение `model_feature_set_decisions.csv` из первого ноутбука как явного входного контракта;
   - честный `GridSearchCV` через `Pipeline` для каждой пары `dataset + model + selected_feature_set`;
   - сравнение лучших tuned-конфигураций на `validation`;
   - финальное сравнение `baseline_default` против `tuned_best` на `test`.

### Workflow: base + mandatory independent
- **Базовый маршрут**: закрывается по `solutions/*_solution.ipynb`.
- **Обязательные самостоятельные блоки**: выполняются только в `notebooks/*_todo.ipynb`.
- В `todo`-ноутбуках самостоятельные места помечены как `TODO(обязательно)` и сопровождаются блоками `Как интерпретировать результат` и `Проверь себя`.
- В `solutions` показан тот же маршрут, но уже как пошаговый walkthrough: что делаем, зачем, как читать результат и какой вывод считается корректным.
- По ходу работы обязательно фиксируйте:
  - где модель переобучается;
  - как feature set влияет на разные model families;
  - где tuning помогает, а где почти ничего не меняет;
  - почему итоговый выбор делается по `validation`, а не по train-оптимизму.

## Как Понять, Где Решать Самостоятельно
В ноутбуках ЛР 03 ищите три типа student-facing маркеров:
- `TODO(обязательно)` — место, где нужно дописать код, заполнить выводы или сохранить артефакт.
- `Как интерпретировать результат` — подсказка, на что смотреть в таблице или графике и какой вывод ожидается.
- `Проверь себя` — короткий чеклист, который помогает понять, что шаг завершен корректно.

Практическое правило:
- если видите только код, не спешите его разбирать построчно;
- сначала прочитайте markdown-блок перед ним;
- затем выполните код;
- потом ответьте на вопросы в `TODO(обязательно)` и сверяйтесь с `Проверь себя`.

## Как `todo` Связан С `solution`
- `todo` показывает тот же маршрут, но оставляет студенту обязательные mini-task по шагам.
- `solution` не подменяет отчет и не дает готовый текст сдачи, а показывает образец хода рассуждения.
- Сопоставление простое: шаги и артефакты в `todo` и `solution` идут в одинаковом порядке.

## Что сдавать
- заполненные ноутбуки с выполненными ячейками и выводами;
- обязательные CSV-артефакты из самостоятельных блоков;
- narrative-блоки по самостоятельному изучению методов;
- обновленный `study-notes/glossary.md`:
  минимум 3 новых термина на каждый ноутбук;
- отчет по шаблону `report-template.md`.

## Формат промежуточных таблиц
`generalization_audit`:
- `dataset`, `feature_set`, `model`, `split`
- `accuracy`, `f1`, `roc_auc`, `fit_time_sec`

`model_feature_set_decisions`:
- `dataset`, `model`, `selected_feature_set`
- `train_f1`, `validation_f1`, `f1_gap`, `abs_f1_gap`, `tie_break_reason`

`validation_curve_results`:
- `dataset`, `feature_set`, `model`, `hyperparameter`, `param_value`, `split`
- `accuracy`, `f1`, `roc_auc`

`gridsearch_results_top`:
- `dataset`, `feature_set`, `model`, `rank`, `params_json`
- `mean_cv_f1`, `std_cv_f1`, `mean_cv_roc_auc`, `mean_cv_accuracy`, `mean_fit_time_sec`

`baseline_vs_tuned_test_results`:
- `dataset`, `feature_set`, `model`, `variant`
- `accuracy`, `f1`, `roc_auc`, `fit_time_sec`

Замечание: в `validation_curve_results` и `gridsearch_results_top` feature set выбирается отдельно для каждой модели, поэтому внутри одного dataset у `LogisticRegression` и `RandomForest` он может отличаться.

## Обязательные самостоятельные блоки
- экспорт `outputs/generalization_audit.csv`
- экспорт `outputs/model_feature_set_decisions.csv`
- экспорт `outputs/validation_curve_results.csv`
- экспорт `outputs/gridsearch_results_top.csv`
- экспорт `outputs/baseline_vs_tuned_test_results.csv`
- обновление narrative-блоков и `study-notes/glossary.md`

## Запуск
Команды выполняются из папки `03-overfitting-validation-and-hyperparameter-tuning`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name lab03-overfitting --display-name "Python (.venv) Lab 03"
.venv/bin/python -m jupyter notebook
```

Опционально можно использовать `source .venv/bin/activate` и запускать команды через `python`.

## Рекомендуемый маршрут выполнения
1. В первом ноутбуке сравнить `full` и все candidate feature set на `train` и `validation`.
2. Явно выписать, где виден `generalization gap`.
3. Зафиксировать, что `train` часто, но не всегда лучше `validation`.
4. Для каждой модели выбрать свой feature set по правилу `max validation f1 -> min abs f1 gap -> prefer non-full -> lexicographic`.
5. Сохранить `model_feature_set_decisions.csv` как явный вход во второй ноутбук.
6. Построить validation curves и зафиксировать, где модель усложняется или становится устойчивее.
7. Во втором ноутбуке запустить `GridSearchCV` только на `train`.
8. Сравнить лучшие tuned-конфигурации на `validation` и выбрать финальную модель.
9. Один раз проверить `baseline_default` и `tuned_best` на `test`.
10. Обновлять `study-notes/glossary.md` по ходу работы, а не в конце.
11. Заполнить отчет по `report-template.md`.

## Связка С Отчетом
- Notebook 1, шаг 1: входные candidate feature set и split идут в разделы `2` и `2.2`.
- Notebook 1, шаги 2-3: `generalization_audit` и `model_feature_set_decisions` идут в разделы `2.1` и `3`.
- Notebook 1, шаг 4: `validation_curve_results` идет в раздел `4`.
- Notebook 2, шаги 1-3: `gridsearch_results_top` и выбор tuned winner идут в раздел `5`.
- Notebook 2, шаг 4: `baseline_vs_tuned_test_results` и итоговая рекомендация идут в разделы `6` и `7`.

## После Каждого Ноутбука
После `notebooks/01_train_validation_overfitting_todo.ipynb` в `outputs/` должны появиться:
- `generalization_audit.csv`
- `model_feature_set_decisions.csv`
- `validation_curve_results.csv`

После `notebooks/02_gridsearch_and_final_choice_todo.ipynb` в `outputs/` должны появиться:
- `gridsearch_results_top.csv`
- `baseline_vs_tuned_test_results.csv`

## Если Notebook 2 Не Стартует
Проверьте по порядку:
- первый ноутбук действительно дошел до export-cell без `NotImplementedError`;
- в `outputs/` лежат все три CSV из notebook 1;
- `model_feature_set_decisions.csv` не редактировался вручную и содержит по одной строке на каждую пару `dataset + model`;
- notebook 1 и notebook 2 запускаются из одной и той же папки модуля и из одного и того же `.venv`.

Если `model_feature_set_decisions.csv` поврежден или устарел, не правьте его вручную: просто заново выполните экспортную ячейку в notebook 1.

## Submission Checklist
- Оба `todo`-ноутбука выполнены и содержат ваши выводы.
- В `outputs/` лежат все 5 обязательных CSV.
- `study-notes/glossary.md` обновлялся по ходу работы.
- Narrative-блоки и отчет по `report-template.md` заполнены.
- В финальном сравнении `test` использован только один раз.

## Проверка Для Преподавателя/Разработчика
После настройки окружения можно прогнать полный smoke-check ЛР:

```bash
.venv/bin/python scripts/verify_lab03.py
```

Скрипт выполняет оба `solution`-ноутбука и проверяет контракты всех обязательных CSV-артефактов.

## Предсдача Для Студента
```bash
.venv/bin/python ../scripts/check_submission.py --lab 03
```

## Что Делать, Если Не Запускается
1. Ошибка `ModuleNotFoundError`:
   - выполните `.venv/bin/python -m pip install -r requirements.txt`.
2. Ошибка `Не найден feature_sets_wrapper_embedded.json`:
   - выполните `.venv/bin/python ../01-feature-importance-and-selection/scripts/verify_lab01.py`;
   - или заново выполните export в ЛР01.
3. Notebook 2 не стартует из-за отсутствующих файлов из notebook 1:
   - выполните notebook 1 до export-ячейки;
   - проверьте `outputs/generalization_audit.csv`, `outputs/model_feature_set_decisions.csv`, `outputs/validation_curve_results.csv`.
4. Для полной диагностики среды из корня репозитория:
   - `.venv/bin/python scripts/doctor_env.py`.

## Траектории Прохождения
- Легкий трек (минимально обязательный маршрут):
  - пройти оба `todo`-ноутбука;
  - сформировать 5 обязательных CSV;
  - заполнить отчет и базовый глоссарий.
- Углубленный трек:
  - добавить `LinearSVC` как третью модель и сравнить ее с базовыми двумя;
  - попробовать `RandomizedSearchCV` и сравнить его с полным `GridSearchCV`;
  - сравнить выбор гиперпараметров по `f1` против выбора по `roc_auc`;
  - проверить, как меняется итог при другом `random_state`.
