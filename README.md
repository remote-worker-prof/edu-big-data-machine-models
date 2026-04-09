# edu-big-data-machine-models

Практический учебный репозиторий по дисциплине  
`Математические основы анализа больших данных и моделей машинного обучения`.

## Структура
- `01-feature-importance-and-selection/` — ЛР 01: значимость и отбор признаков.
- `02-model-interpretability-and-explainability/` — ЛР 02: интерпретация и объяснение моделей.
- `03-overfitting-validation-and-hyperparameter-tuning/` — ЛР 03: переобучение, validation и честный подбор гиперпараметров.
  Внутри ЛР 03 есть отдельный maintainers smoke-check: `python 03-overfitting-validation-and-hyperparameter-tuning/scripts/verify_lab03.py`.
- `04-calibration-threshold-and-decision-policy/` — ЛР 04: калибровка вероятностей, выбор порога и cost-sensitive policy.
  Внутри ЛР 04 есть отдельный maintainers smoke-check: `python 04-calibration-threshold-and-decision-policy/scripts/verify_lab04.py`.
- `05-drift-monitoring-and-retraining-policy/` — ЛР 05: мониторинг дрейфа данных/качества и policy решения о переобучении.
  Внутри ЛР 05 есть отдельный maintainers smoke-check: `python 05-drift-monitoring-and-retraining-policy/scripts/verify_lab05.py`.
- `.venv/` — единое локальное окружение Python для проекта (не коммитится).

## Статус аудита (2026-04-09)
- Выполнен полный аудит структуры, учебных контрактов и проверочных сценариев проекта.
- `Lab 04` присутствует и полностью укомплектована: `README`, `notebooks`, `solutions`, `tests`, `scripts/verify_lab04.py`.
- Smoke-check пройден для ЛР 03 и ЛР 04: `verify_lab03.py` и `verify_lab04.py`.
- Юнит-тесты `lab_utils` пройдены для ЛР 03 и ЛР 04.
- Проверка стиля комментариев ноутбуков пройдена: `scripts/verify_notebook_comment_style.py`.
- Локальные артефакты `outputs/*` присутствуют, но не коммитятся; политика `.gitignore` соблюдается.
- API, форматы данных и кодовые контракты не изменялись: обновлена только верхнеуровневая документация.

## Текущая лабораторная
Материалы ЛР 01 находятся в:
- [01-feature-importance-and-selection/README.md](./01-feature-importance-and-selection/README.md)

Материалы ЛР 02 находятся в:
- [02-model-interpretability-and-explainability/README.md](./02-model-interpretability-and-explainability/README.md)

Материалы ЛР 03 находятся в:
- [03-overfitting-validation-and-hyperparameter-tuning/README.md](./03-overfitting-validation-and-hyperparameter-tuning/README.md)

Материалы ЛР 04 находятся в:
- [04-calibration-threshold-and-decision-policy/README.md](./04-calibration-threshold-and-decision-policy/README.md)

Материалы ЛР 05 находятся в:
- [05-drift-monitoring-and-retraining-policy/README.md](./05-drift-monitoring-and-retraining-policy/README.md)

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
