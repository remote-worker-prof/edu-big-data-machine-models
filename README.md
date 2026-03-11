# edu-big-data-machine-models

Практический учебный репозиторий по дисциплине  
`Математические основы анализа больших данных и моделей машинного обучения`.

## Структура
- `01-feature-importance-and-selection/` — ЛР 01: значимость и отбор признаков.
- `.venv/` — единое локальное окружение Python для проекта (не коммитится).

## Текущая лабораторная
Материалы ЛР 01 находятся в:
- [01-feature-importance-and-selection/README.md](./01-feature-importance-and-selection/README.md)

Внутри ЛР 01:
- `notebooks/` — ноутбуки с TODO и обязательными самостоятельными блоками;
- `solutions/` — решения только базового маршрута;
- `study-notes/` — заметки и глоссарий по ходу выполнения;
- `report-template.md` — шаблон отчета.

## Быстрый старт
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r 01-feature-importance-and-selection/requirements.txt
jupyter notebook
```

## Git-политика
- Generated outputs (`outputs/*.csv`, `outputs/*.json`) не добавляются в git.
- В репозиторий идут исходные данные, код, ноутбуки и документация.
