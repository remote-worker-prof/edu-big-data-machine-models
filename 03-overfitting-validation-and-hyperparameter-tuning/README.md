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
- `solutions/` — решения базового маршрута без ответов на narrative-блоки.
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
- В `solutions` нет ответов на narrative-блоки и самостоятельные задания намеренно.
- По ходу работы обязательно фиксируйте:
  - где модель переобучается;
  - как feature set влияет на разные model families;
  - где tuning помогает, а где почти ничего не меняет;
  - почему итоговый выбор делается по `validation`, а не по train-оптимизму.

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
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m ipykernel install --user --name lab03-overfitting --display-name "Python (.venv) Lab 03"
jupyter notebook
```

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
python scripts/verify_lab03.py
```

Скрипт выполняет оба `solution`-ноутбука и проверяет контракты всех обязательных CSV-артефактов.

## Расширения на 1-2 дня
- добавить `LinearSVC` как третью модель и сравнить ее с базовыми двумя;
- попробовать `RandomizedSearchCV` и сравнить его с полным `GridSearchCV`;
- сравнить выбор гиперпараметров по `f1` против выбора по `roc_auc`;
- проверить, как меняется итог при другом `random_state`.
