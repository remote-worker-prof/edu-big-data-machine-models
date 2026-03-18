# edu-big-data-machine-models

Практический учебный репозиторий по дисциплине  
`Математические основы анализа больших данных и моделей машинного обучения`.

## Структура
- `01-feature-importance-and-selection/` — ЛР 01: значимость и отбор признаков.
- `02-model-interpretability-and-explainability/` — ЛР 02: интерпретация и объяснение моделей.
- `03-overfitting-validation-and-hyperparameter-tuning/` — ЛР 03: переобучение, validation и честный подбор гиперпараметров.
- `.venv/` — единое локальное окружение Python для проекта (не коммитится).

## Текущая лабораторная
Материалы ЛР 01 находятся в:
- [01-feature-importance-and-selection/README.md](./01-feature-importance-and-selection/README.md)

Материалы ЛР 02 находятся в:
- [02-model-interpretability-and-explainability/README.md](./02-model-interpretability-and-explainability/README.md)

Материалы ЛР 03 находятся в:
- [03-overfitting-validation-and-hyperparameter-tuning/README.md](./03-overfitting-validation-and-hyperparameter-tuning/README.md)

Внутри ЛР 01:
- `notebooks/` — ноутбуки с TODO и обязательными самостоятельными блоками;
- `solutions/` — решения только базового маршрута;
- `study-notes/` — заметки и глоссарий по ходу выполнения;
- `report-template.md` — шаблон отчета.

## Быстрый старт
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r 01-feature-importance-and-selection/requirements.txt
jupyter notebook
```

## Git-политика
- Generated outputs (`outputs/*.csv`, `outputs/*.json`) не добавляются в git.
- В репозиторий идут исходные данные, код, ноутбуки и документация.
