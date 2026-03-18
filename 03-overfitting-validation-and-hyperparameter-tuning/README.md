# Лабораторная работа 03: Overfitting, Validation and Honest Hyperparameter Tuning

## О чем эта работа
Эта лабораторная отвечает на очень практичный вопрос:
почему модель может выглядеть отлично на обучающей выборке,
но заметно хуже работать на новых данных.

Здесь мы не "накручиваем метрику", а учимся улучшать модель честно:
- отделять `train`, `validation`, `test`;
- замечать переобучение по разрыву между качеством на обучении и проверке;
- подбирать гиперпараметры через `GridSearchCV`, не используя `test` для выбора.

ЛР 03 продолжает ЛР 01:
- берем те же датасеты `medical` и `finance`;
- используем candidate feature set из `../01-feature-importance-and-selection/outputs/feature_sets_wrapper_embedded.json`;
- опираемся на `../01-feature-importance-and-selection/outputs/model_results.csv`,
  чтобы взять лучший неполный feature set как стартовую точку.

ЛР 02 остается смысловым мостом:
мы уже учились объяснять поведение модели, а теперь учимся улучшать ее без самообмана.

## Формат
- 2 обязательных Jupyter-ноутбука.
- Те же 2 прикладных бинарных датасета:
  - медицина: прогноз сердечно-сосудистого риска;
  - финансы: прогноз кредитного риска.
- Локальный запуск на CPU.
- Только знакомый стек: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.
- Без новых тяжелых библиотек и без сложной теории ради теории.

## Зависимости от ЛР 01
Новая ЛР ожидает, что доступны артефакты из:
- `../01-feature-importance-and-selection/outputs/feature_sets_wrapper_embedded.json`
- `../01-feature-importance-and-selection/outputs/model_results.csv`

Если ЛР 01 уже была пройдена в текущем репозитории, ноутбуки подхватят эти файлы автоматически.

## Структура папки
- `notebooks/` — шаблоны заданий с TODO и обязательными самостоятельными блоками.
- `solutions/` — решения базового маршрута без ответов на самостоятельные narrative-блоки.
- `study-notes/` — заметки и глоссарий по переобучению, валидации и тюнингу.
- `outputs/` — промежуточные и итоговые таблицы.
- `report-template.md` — шаблон итогового отчета.
- `requirements.txt` — зависимости.
- `lab_utils.py` — общие утилиты для обоих ноутбуков.

## Ноутбуки и порядок прохождения
1. `notebooks/01_train_validation_overfitting_todo.ipynb` (90 минут)
   - загрузка feature set из ЛР 01;
   - сравнение `full` против лучшего неполного feature set;
   - оценка `LogisticRegression` и `RandomForestClassifier` на `train` и `validation`;
   - вычисление `generalization gap`;
   - простые validation curves по одному гиперпараметру на модель.
2. `notebooks/02_gridsearch_and_final_choice_todo.ipynb` (90 минут)
   - выбор одного feature set на основе результатов первого ноутбука;
   - честный `GridSearchCV` через `Pipeline`;
   - сравнение лучших конфигураций на `validation`;
   - финальное сравнение `baseline_default` против `tuned_best` на `test`.

### Workflow: base + mandatory independent
- **Базовый маршрут**: закрывается по `solutions/*_solution.ipynb`.
- **Обязательные самостоятельные блоки**: выполняются только в `notebooks/*_todo.ipynb`.
- В `solutions` нет ответов на narrative-блоки и самостоятельные задания намеренно.
- По ходу работы обязательно фиксируйте:
  - где модель переобучается;
  - какой гиперпараметр на что влияет;
  - почему итоговый выбор делается именно так, а не по одной красивой цифре.

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

`validation_curve_results`:
- `dataset`, `feature_set`, `model`, `hyperparameter`, `param_value`, `split`
- `accuracy`, `f1`, `roc_auc`

`gridsearch_results_top`:
- `dataset`, `feature_set`, `model`, `rank`, `params_json`
- `mean_cv_f1`, `std_cv_f1`, `mean_cv_roc_auc`, `mean_cv_accuracy`, `mean_fit_time_sec`

`baseline_vs_tuned_test_results`:
- `dataset`, `feature_set`, `model`, `variant`
- `accuracy`, `f1`, `roc_auc`, `fit_time_sec`

## Обязательные самостоятельные блоки
- экспорт `outputs/generalization_audit.csv`
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
1. В первом ноутбуке сравнить `full` и лучший неполный feature set на `train` и `validation`.
2. Явно выписать, где виден `generalization gap`.
3. Построить validation curves и зафиксировать, где модель усложняется или становится устойчивее.
4. Во втором ноутбуке запустить `GridSearchCV` только на `train`.
5. Сравнить лучшие конфигурации на `validation` и выбрать финальную модель.
6. Один раз проверить `baseline_default` и `tuned_best` на `test`.
7. Обновлять `study-notes/glossary.md` по ходу работы, а не в конце.
8. Заполнить отчет по `report-template.md`.

## Расширения на 1-2 дня
- добавить `LinearSVC` как третью модель и сравнить ее с базовыми двумя;
- попробовать `RandomizedSearchCV` и сравнить его с полным `GridSearchCV`;
- сравнить выбор гиперпараметров по `f1` против выбора по `roc_auc`;
- проверить, как меняется итог при другом `random_state`.
