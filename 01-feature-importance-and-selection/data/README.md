# Датасеты для ЛР 01

Оба датасета имеют бинарный целевой столбец `target` (0/1).

## 1) `medical_cardiovascular_risk.csv`
Практический кейс по оценке риска сердечно-сосудистого события.

### Основные признаки
- `age`, `sex`, `bmi`, `systolic_bp`, `diastolic_bp`
- `cholesterol`, `glucose`, `resting_heart_rate`
- `smoking_status`, `family_history`
- `physical_activity_hours`, `stress_level`, `alcohol_units_weekly`
- `target` — 1 означает повышенный риск.

## 2) `finance_credit_risk.csv`
Практический кейс по оценке риска дефолта по кредиту.

### Основные признаки
- `age`, `annual_income`, `loan_amount`, `loan_to_income`
- `credit_score`, `employment_years`, `delinquency_count`
- `open_credit_lines`, `utilization_ratio`, `savings_balance`
- `housing_status`, `employment_type`, `previous_default`, `purpose`
- `target` — 1 означает повышенный кредитный риск.

## Методические замечания
- Наборы сделаны в формате, удобном для CPU-обучения и прохождения ЛР в аудитории.
- В данных присутствуют пропуски, чтобы отработать базовый препроцессинг.
- Все методы отбора признаков применяются только к train-части.
