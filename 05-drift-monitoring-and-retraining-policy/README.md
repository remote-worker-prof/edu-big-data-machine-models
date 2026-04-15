# Лабораторная работа 05: Drift Monitoring и Retraining Policy

## О чем эта работа
ЛР05 — завершающая лабораторная по курсу. Здесь вы учитесь не просто считать метрики, а принимать управленческое решение по модели на новых данных:
- наблюдать дальше (`observe`);
- или запускать переобучение (`retrain`).

Ключевая логика ЛР05:
`изменения в данных -> статистические сигналы -> изменение качества/стоимости -> policy-решение`.

## Для кого и что можно не знать
- Работа рассчитана на студента, который начинает "с нуля".
- Не требуется глубокая матстатистика, доказательства теорем или продвинутый MLOps.
- Достаточно базово понимать: признак, таргет, бинарная классификация.

## Что получится в конце (конкретный результат)
Вы сформируете и интерпретируете 4 обязательных артефакта:
1. `drift_detection_audit.csv`
2. `monitoring_quality_audit.csv`
3. `retraining_policy_decisions.csv`
4. `post_retrain_comparison.csv`

Итоговый практический результат: по каждому сценарию у вас будет обоснованное решение `observe` или `retrain`.

## Быстрый маршрут новичка
1. Прочитайте теоретический notebook полностью (особенно разделы 0-5).
2. Выполните практику 1 и сохраните 2 CSV (`drift_detection_audit`, `monitoring_quality_audit`).
3. Выполните практику 2 и сохраните еще 2 CSV (`retraining_policy_decisions`, `post_retrain_comparison`).
4. Заполните отчет по шаблону и объясните решение простыми словами.

Если застряли:
1. Вернитесь к блоку `Перед началом` в текущем notebook.
2. Проверьте блоки `Вход/Выход` и `Проверь себя` текущего шага.
3. Сверьте интерпретацию с форматом: `что вижу в таблице -> что это значит -> что делаю дальше`.

## Нулевой статистический минимум
Перед практикой зафиксируйте смысл терминов:
- `выборка`: ограниченный срез данных, который мы реально анализируем;
- `распределение`: как часто встречаются разные значения признака;
- `H0/H1`: формальная пара гипотез "отличия нет" / "отличие есть";
- `alpha`: порог для `p-value` (в ЛР05: `0.05`);
- ошибки I/II рода: ложная тревога и пропуск реального сигнала;
- `PSI`: сила сдвига (насколько отличие велико практически);
- `confusion matrix` и метрики `precision/recall/f1`: структура и баланс ошибок;
- `threshold`: граница перевода вероятности в класс;
- `Brier`, `ECE`: качество вероятностей (калибровка);
- `expected_cost`: практическая цена ошибок.

## Карманный словарь латиницы (где это видно в CSV)
| Термин | Простое русское имя | Где смотреть в артефактах |
|---|---|---|
| `covariate` | сдвиг профиля признаков | `drift_detection_audit.csv` -> `scenario` |
| `prior` | сдвиг доли целевого класса | `drift_detection_audit.csv` -> `scenario` |
| `combined` | комбинированный сдвиг | `drift_detection_audit.csv` -> `scenario` |
| `KS` / `chi2` | статистические тесты различий | `drift_detection_audit.csv` -> `detector`, `p_value` |
| `p-value` | статистическая заметность | `drift_detection_audit.csv` -> `p_value` |
| `PSI` | сила эффекта сдвига | `drift_detection_audit.csv` -> `effect_size` |
| `drift_flag` | итоговый флаг сдвига | `drift_detection_audit.csv` -> `drift_flag` |
| `f1` | баланс точности и полноты | `monitoring_quality_audit.csv` -> `f1`, `delta_f1_vs_reference` |
| `expected_cost` | стоимость ошибок | `monitoring_quality_audit.csv` -> `expected_cost`, `delta_cost_vs_reference` |
| `policy_action` | итоговое действие | `retraining_policy_decisions.csv` -> `policy_action` |
| `trigger_reason` | причина действия | `retraining_policy_decisions.csv` -> `trigger_reason` |
| `before_retrain/after_retrain` | фазы до/после переобучения | `post_retrain_comparison.csv` -> `phase` |

## Формат и структура
- `theory-notebooks/` — 1 теоретический notebook.
- `notebooks/` — 2 практических notebook с `TODO(обязательно)`.
- `solutions/` — 2 полностью заполненных решения.
- `outputs/` — CSV-артефакты (локальные, не коммитятся по `.gitignore`).
- `study-notes/` — заметки и расширенный глоссарий.
- `report-template.md` — шаблон отчета.
- `lab_utils.py` — утилиты ЛР05.
- `scripts/verify_lab05.py` — smoke-check.
- `tests/` — unit-тесты.

## Фиксированные policy-пороги (без изменений)
- `alpha = 0.05`
- `psi_warn = 0.10`
- `psi_alert = 0.25`
- `retrain_f1_drop = 0.05`
- `retrain_cost_increase = 0.15`
- Правило решения:
  - `retrain`, если `drift_feature_share >= 0.30`, или `delta_f1_vs_reference <= -0.05`, или `delta_cost_vs_reference >= +0.15`;
  - иначе `observe`.

## Контракты артефактов
`drift_detection_audit.csv`:
- `dataset`, `window_id`, `scenario`, `feature`, `feature_type`, `detector`, `statistic`, `p_value`, `effect_size`, `drift_flag`

`monitoring_quality_audit.csv`:
- `dataset`, `window_id`, `scenario`, `model_variant`, `accuracy`, `f1`, `roc_auc`, `pr_auc`, `brier`, `ece`, `expected_cost`, `delta_f1_vs_reference`, `delta_cost_vs_reference`

`retraining_policy_decisions.csv`:
- `dataset`, `window_id`, `scenario`, `drift_feature_share`, `delta_f1_vs_reference`, `delta_cost_vs_reference`, `policy_action`, `trigger_reason`

`post_retrain_comparison.csv`:
- `dataset`, `scenario`, `phase`, `accuracy`, `f1`, `roc_auc`, `pr_auc`, `brier`, `ece`, `expected_cost`

## Запуск
Рекомендуется общий `.venv` в корне репозитория для всех лабораторных.

```bash
python3 -m venv .venv  # если окружение еще не создано
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r 05-drift-monitoring-and-retraining-policy/requirements.txt
.venv/bin/python -m ipykernel install --user --name lab05-monitoring --display-name "Python (.venv) Lab 05"
.venv/bin/python -m jupyter notebook 05-drift-monitoring-and-retraining-policy
```

Опционально можно использовать `source .venv/bin/activate` и запускать команды через `python`.

## Быстрая проверка (для студента)
Из папки ЛР05:

```bash
.venv/bin/python scripts/verify_lab05.py
```

Если скрипт завершился с `Lab 05 smoke-check passed.`, работа оформлена корректно.

## Предсдача Для Студента
```bash
.venv/bin/python ../scripts/check_submission.py --lab 05
```

## Расширенные проверки (для преподавателя/автора курса)
Из корня репозитория при подготовке к публикации:

```bash
.venv/bin/python -m unittest 05-drift-monitoring-and-retraining-policy/tests/test_lab_utils.py
.venv/bin/python scripts/verify_notebook_comment_style.py
```

## Что Делать, Если Не Запускается
1. Ошибка `ModuleNotFoundError`:
   - выполните `.venv/bin/python -m pip install -r 05-drift-monitoring-and-retraining-policy/requirements.txt`.
2. Ошибка про отсутствие `medical_cardiovascular_risk.csv` или `finance_credit_risk.csv`:
   - проверьте наличие файлов в `../01-feature-importance-and-selection/data/`.
3. Запуск из неверной директории:
   - команды ЛР05 выполняйте из папки `05-drift-monitoring-and-retraining-policy` или из корня с полным путем.
4. Для полной диагностики среды из корня репозитория:
   - `.venv/bin/python scripts/doctor_env.py`.

## Траектории Прохождения
- Легкий трек (минимально обязательный маршрут):
  - пройти теоретический ноутбук и два `todo`-ноутбука;
  - сформировать 4 обязательных CSV;
  - заполнить отчет простыми объяснениями решения `observe/retrain`.
- Углубленный трек:
  - протестировать альтернативные policy-пороги и сравнить итоговые решения;
  - добавить дополнительные monitoring-сценарии и оценить устойчивость policy;
  - расширить анализ post-retrain сравнением нескольких model variants.
