# edu-big-data-machine-models

Практический учебный репозиторий по дисциплине  
`Математические основы анализа больших данных и моделей машинного обучения`.

## Структура
- `01-feature-importance-and-selection/` — ЛР 01: значимость и отбор признаков.
- Внутри ЛР 01 есть отдельный maintainers smoke-check: `python 01-feature-importance-and-selection/scripts/verify_lab01.py`.
- `02-model-interpretability-and-explainability/` — ЛР 02: интерпретация и объяснение моделей.
- Внутри ЛР 02 есть отдельный maintainers smoke-check: `python 02-model-interpretability-and-explainability/scripts/verify_lab02.py`.
- `03-overfitting-validation-and-hyperparameter-tuning/` — ЛР 03: переобучение, validation и честный подбор гиперпараметров.
- Внутри ЛР 03 есть отдельный maintainers smoke-check: `python 03-overfitting-validation-and-hyperparameter-tuning/scripts/verify_lab03.py`.
- `04-calibration-threshold-and-decision-policy/` — ЛР 04: калибровка вероятностей, выбор порога и cost-sensitive policy.
- Внутри ЛР 04 есть отдельный maintainers smoke-check: `python 04-calibration-threshold-and-decision-policy/scripts/verify_lab04.py`.
- `05-drift-monitoring-and-retraining-policy/` — ЛР 05: мониторинг дрейфа данных/качества и policy решения о переобучении.
- Внутри ЛР 05 есть отдельный maintainers smoke-check: `python 05-drift-monitoring-and-retraining-policy/scripts/verify_lab05.py`.
- `scripts/verify_all_labs.py` — единый запускатор проверок курса (ЛР 01-05 + comment-style + unit-tests).
- `scripts/doctor_env.py` — диагностика среды и upstream-артефактов с точными командами исправления.
- `scripts/check_submission.py` — предсдачная проверка обязательных артефактов по ЛР.
- `.venv/` — единое локальное окружение Python для проекта (не коммитится).

## Статус аудита (2026-04-15)
- Выполнен полный аудит структуры, учебных контрактов и проверочных сценариев проекта.
- Добавлены smoke-check для ЛР 01 и ЛР 02: `verify_lab01.py`, `verify_lab02.py`.
- Smoke-check пройден для ЛР 01-05.
- Юнит-тесты `lab_utils` пройдены для ЛР 03-05.
- Проверка стиля комментариев ноутбуков пройдена: `scripts/verify_notebook_comment_style.py`.
- Добавлен unified quality gate: `scripts/verify_all_labs.py` и CI workflow `.github/workflows/quality-gate.yml`.
- Добавлены инструменты student UX: `scripts/doctor_env.py` и `scripts/check_submission.py`.
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

## Официальный Onboarding И Проверка Среды
Команды выполняются из корня репозитория.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r 01-feature-importance-and-selection/requirements.txt
.venv/bin/python -m pip install -r 02-model-interpretability-and-explainability/requirements.txt
.venv/bin/python -m pip install -r 03-overfitting-validation-and-hyperparameter-tuning/requirements.txt
.venv/bin/python -m pip install -r 04-calibration-threshold-and-decision-policy/requirements.txt
.venv/bin/python -m pip install -r 05-drift-monitoring-and-retraining-policy/requirements.txt
.venv/bin/python -c "import pandas, numpy, sklearn, scipy, nbconvert; print('Environment check passed')"
```

Опционально можно использовать `activate`:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Быстрый Запуск Jupyter
```bash
.venv/bin/python -m jupyter notebook
```

## Единый QA Gate
```bash
.venv/bin/python scripts/verify_all_labs.py
```

Скрипт запускает:
- smoke-check ЛР 01-05;
- `scripts/verify_notebook_comment_style.py`;
- `python -m unittest` для `lab_utils` в ЛР 03-05.

## Диагностика Среды
```bash
.venv/bin/python scripts/doctor_env.py
```

Скрипт показывает:
- состояние зависимостей;
- доступность `python -m jupyter`;
- наличие upstream-артефактов для ЛР02-04;
- точные команды исправления.

## Предсдачная Проверка
```bash
.venv/bin/python scripts/check_submission.py --lab 01
.venv/bin/python scripts/check_submission.py --lab 02
.venv/bin/python scripts/check_submission.py --lab 03
.venv/bin/python scripts/check_submission.py --lab 04
.venv/bin/python scripts/check_submission.py --lab 05
```

Опционально:
```bash
.venv/bin/python scripts/check_submission.py --lab all
```

## Проверка По Лабораторным Отдельно
- ЛР 01: `.venv/bin/python 01-feature-importance-and-selection/scripts/verify_lab01.py`
- ЛР 02: `.venv/bin/python 02-model-interpretability-and-explainability/scripts/verify_lab02.py`
- ЛР 03: `.venv/bin/python 03-overfitting-validation-and-hyperparameter-tuning/scripts/verify_lab03.py`
- ЛР 04: `.venv/bin/python 04-calibration-threshold-and-decision-policy/scripts/verify_lab04.py`
- ЛР 05: `.venv/bin/python 05-drift-monitoring-and-retraining-policy/scripts/verify_lab05.py`

## CI Quality Gate
- В репозитории настроен workflow: `.github/workflows/quality-gate.yml`.
- Для блокировки merge настройте branch protection с required status check `verify-all-labs`.
- Предсеместровый чеклист преподавателя: [RELEASE_CHECKLIST.md](./RELEASE_CHECKLIST.md)

## Единый Rubric Отчетов
- Шаблон rubric для ЛР 01-05: [RUBRIC_TEMPLATE.md](./RUBRIC_TEMPLATE.md)

## Лекционный Long-Read
- Подробный академический конспект для лекционного показа (ЛР 01-05, 90 минут): [LECTURE_LONGREAD_01_05.md](./LECTURE_LONGREAD_01_05.md)

## Git-политика
- Generated outputs (`outputs/*.csv`, `outputs/*.json`) не добавляются в git.
- В репозиторий идут исходные данные, код, ноутбуки и документация.
