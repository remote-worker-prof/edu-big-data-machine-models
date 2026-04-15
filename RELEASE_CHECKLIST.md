# Release Checklist Для Преподавателя

Короткий предсеместровый чеклист перед публикацией/обновлением материалов курса.

## 1. Подготовка Ветки И Ревью
1. Все изменения оформлены в PR с читаемым описанием (что меняется в учебном процессе).
2. Есть минимум один ревьюер (методист/преподаватель/maintainer).
3. В PR отмечены возможные риски для студентов (breaking points, новые зависимости).

## 2. Локальный Quality Gate
Команды выполнять из корня репозитория:

```bash
.venv/bin/python scripts/doctor_env.py
.venv/bin/python scripts/verify_all_labs.py
```

Ожидаемый результат:
- `doctor_env.py` без `FAIL`;
- `verify_all_labs.py` с `QA summary: PASS`.

## 3. Контракт Upstream-Артефактов
Перед публикацией убедиться, что доступны ключевые входные файлы:
- ЛР02: `01-feature-importance-and-selection/outputs/feature_sets_wrapper_embedded.json`, `model_results.csv`;
- ЛР03: `01-feature-importance-and-selection/outputs/feature_sets_wrapper_embedded.json`;
- ЛР04: `03-overfitting-validation-and-hyperparameter-tuning/outputs/baseline_vs_tuned_test_results.csv`.

Если upstream отсутствует, это должно быть явно отражено в README и troubleshooting.

## 4. Студенческий UX
1. Во всех ЛР есть единые разделы:
   - запуск через `.venv/bin/python`;
   - “Что делать, если не запускается”;
   - четкий список обязательных артефактов сдачи.
2. Для ЛР01–03 есть блок “Вход для новичка”.
3. Для каждой ЛР указан легкий и углубленный трек.

## 5. Branch Protection (GitHub)
В настройках репозитория для основной ветки:
1. Включить `Require a pull request before merging`.
2. Включить `Require status checks to pass before merging`.
3. Добавить required check: `verify-all-labs`.
4. Запретить прямой push в main (по политике команды).

## 6. После Merge
1. Проверить, что workflow `Quality Gate` прошел на merge-коммите.
2. Обновить changelog/анонс для студентов:
   - что изменилось;
   - что делать, если старые ноутбуки не совпадают с новыми требованиями.
