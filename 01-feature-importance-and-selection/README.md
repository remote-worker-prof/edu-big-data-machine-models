# Лабораторная работа 01: Feature Importance and Selection

## О чем эта работа
Эта лабораторная знакомит с практическим отбором признаков без углубления в тяжелые математические доказательства.
Вы последовательно сравните несколько подходов к оценке значимости признаков и проверите, как отбор влияет на качество моделей.

## Формат
- 3 обязательных Jupyter-ноутбука.
- 2 прикладных бинарных классификационных датасета:
  - медицина: прогноз сердечно-сосудистого риска;
  - финансы: прогноз кредитного риска.
- Локальный запуск на CPU.
- Данные в CSV лежат внутри репозитория.

## Структура папки
- `data/` — исходные датасеты и краткое описание колонок.
- `notebooks/` — шаблоны заданий с TODO + обязательные самостоятельные блоки (без образцов решений).
- `solutions/` — готовые решения только для базового маршрута ноутбуков.
- `outputs/` — промежуточные и итоговые таблицы (`feature_ranking`, `model_results`).
- `report-template.md` — шаблон итогового отчета студента.
- `requirements.txt` — зависимости.

## Ноутбуки и порядок прохождения
1. `notebooks/01_filter_methods_todo.ipynb` (90 минут)
   - базовый pipeline;
   - `VarianceThreshold`, корреляционный анализ, `mutual_info_classif`, `f_classif`;
   - формирование `feature_ranking` и shortlist.
2. `notebooks/02_wrapper_embedded_todo.ipynb` (90 минут)
   - `RFE`, `SequentialFeatureSelector`, L1-регуляризация;
   - важности деревьев и permutation importance;
   - формирование 2-3 кандидатных feature set.
3. `notebooks/03_model_comparison_todo.ipynb` (60 минут)
   - сравнение `LogisticRegression`, `RandomForest`, `LinearSVC`;
   - опциональный блок `MLPClassifier`;
   - формирование `model_results` и практических выводов.

### Workflow: base + mandatory independent
- **Базовый маршрут**: закрывается по `solutions/*_solution.ipynb`.
- **Обязательные самостоятельные блоки**: выполняются только в `notebooks/*_todo.ipynb`.
- В `solutions` нет ответов на самостоятельные блоки намеренно.

## Тайминг
- Ядро: 4 академических часа (90 + 90 + 60 минут).
- Расширенный трек: 1-2 дня (опциональные эксперименты и более глубокая интерпретация).

## Что сдавать
- Заполненные ноутбуки (выполненные ячейки и комментарии с выводами).
- Все дополнительные CSV-артефакты из обязательных самостоятельных блоков.
- Краткий отчет по шаблону `report-template.md`.

## Формат промежуточных таблиц
`feature_ranking`:
- `dataset` — название датасета;
- `method` — метод отбора/оценки;
- `feature` — признак;
- `score` — числовой скор;
- `rank` — ранг внутри `dataset + method`.

`model_results`:
- `dataset` — название датасета;
- `feature_set` — набор признаков (`full`, `set_A`, ...);
- `model` — название модели;
- `metric` — метрика (`accuracy`, `f1`, `roc_auc`);
- `value` — значение метрики;
- `fit_time_sec` — время обучения модели в секундах.

## Обязательные самостоятельные блоки (только в notebooks)
В каждом `*_todo.ipynb` добавлен раздел `Обязательные самостоятельные задания (без образца в solutions)`.

Важно:
- эти задания **обязательны** к выполнению;
- в шаблонах стоят intentional-stop ячейки с `NotImplementedError`;
- пока студент не заменит шаблон своим кодом, ноутбук останавливается в этом разделе.

Дополнительные обязательные артефакты:
- `outputs/filter_stability_grid.csv`
- `outputs/method_agreement_long.csv`
- `outputs/selection_stability.csv`
- `outputs/threshold_tuning_results.csv`
- `outputs/cv_stability_results.csv`
- `outputs/error_by_segment.csv`

## Запуск
Команды выполняются из папки `01-feature-importance-and-selection`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name feature-lab --display-name "Python (.venv) Feature Lab"
jupyter notebook
```

## Рекомендуемый маршрут для студента
1. Пройти базовые шаги `01_filter_methods_todo.ipynb`, затем выполнить обязательные самостоятельные задания в конце.
2. Пройти базовые шаги `02_wrapper_embedded_todo.ipynb`, затем выполнить обязательные самостоятельные задания в конце.
3. Пройти базовые шаги `03_model_comparison_todo.ipynb`, затем выполнить обязательные самостоятельные задания в конце.
4. Проверить, что все обязательные CSV-артефакты сохранены в `outputs/`.
5. Заполнить отчет по `report-template.md`.

## Расширения на 1-2 дня
- Повторить эксперименты с другими гиперпараметрами отбора (число признаков, пороги).
- Добавить кросс-валидацию и доверительные интервалы метрик.
- Активировать опциональный блок `MLPClassifier` и сравнить с классическими моделями.
