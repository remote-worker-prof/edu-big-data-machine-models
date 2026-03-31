"""Утилиты для ЛР 04: calibration, threshold policy и cost-sensitive decisions.

ЛР 04 опирается на результаты ЛР 03 и показывает, как перейти
от "хороших метрик" к практической policy принятия решения.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
LAB01_DIR = BASE_DIR.parent / "01-feature-importance-and-selection"
LAB03_DIR = BASE_DIR.parent / "03-overfitting-validation-and-hyperparameter-tuning"
DATA_DIR = LAB01_DIR / "data"
LAB01_OUTPUT_DIR = LAB01_DIR / "outputs"
LAB03_OUTPUT_DIR = LAB03_DIR / "outputs"
OUTPUT_DIR = BASE_DIR / "outputs"

SEED = 42
DEFAULT_FP_COST = 1.0
DEFAULT_FN_COST = 5.0
DEFAULT_MIN_RECALL = 0.60
DEFAULT_THRESHOLD_GRID = np.round(np.linspace(0.05, 0.95, 19), 2)

DATASET_PATHS = {
    "medical": DATA_DIR / "medical_cardiovascular_risk.csv",
    "finance": DATA_DIR / "finance_credit_risk.csv",
}

SEGMENT_FEATURES = {
    "medical": ["age", "cholesterol", "smoking_status"],
    "finance": ["credit_score", "loan_to_income", "previous_default"],
}

CALIBRATION_AUDIT_COLUMNS = [
    "dataset",
    "model",
    "variant",
    "split",
    "brier",
    "log_loss",
    "roc_auc",
    "pr_auc",
    "ece",
]

THRESHOLD_POLICY_GRID_COLUMNS = [
    "dataset",
    "model",
    "variant",
    "threshold",
    "precision",
    "recall",
    "f1",
    "fp_rate",
    "fn_rate",
    "expected_cost",
]

POLICY_TEST_REPORT_COLUMNS = [
    "dataset",
    "model",
    "variant",
    "policy_name",
    "threshold",
    "accuracy",
    "f1",
    "roc_auc",
    "pr_auc",
    "expected_cost",
    "cost_per_100",
]

SEGMENT_POLICY_AUDIT_COLUMNS = [
    "dataset",
    "segment_feature",
    "segment",
    "n",
    "fp_rate",
    "fn_rate",
    "expected_cost_per_100",
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Загружает CSV и проверяет наличие колонки `target`."""

    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError(f"В датасете {path} отсутствует колонка 'target'.")
    return df


def load_course_datasets() -> Dict[str, pd.DataFrame]:
    """Возвращает оба датасета курса в виде словаря."""

    return {name: load_dataset(path) for name, path in DATASET_PATHS.items()}


def split_xy(df: pd.DataFrame, target: str = "target") -> Tuple[pd.DataFrame, pd.Series]:
    """Разделяет признаки и таргет."""

    x = df.drop(columns=[target]).copy()
    y = df[target].astype(int).copy()
    return x, y


def train_valid_test_split_stratified(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Делит данные на train/validation/test со стратификацией (60/20/20)."""

    if not np.isclose(test_size + valid_size, 0.4):
        raise ValueError("Ожидалась схема 60/20/20: test_size + valid_size должны давать 0.4.")

    x_train_valid, x_test, y_train_valid, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    relative_valid_size = valid_size / (1.0 - test_size)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_valid,
        y_train_valid,
        test_size=relative_valid_size,
        random_state=random_state,
        stratify=y_train_valid,
    )

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def infer_feature_types(x: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Возвращает списки числовых и категориальных признаков."""

    numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in x.columns if col not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Строит единый препроцессор: impute + scale + one-hot."""

    numeric_features, categorical_features = infer_feature_types(x)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def transform_with_names(
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    x_test: pd.DataFrame,
):
    """Фитит препроцессор на train и трансформирует train/valid/test."""

    x_train_t = preprocessor.fit_transform(x_train)
    x_valid_t = preprocessor.transform(x_valid)
    x_test_t = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return x_train_t, x_valid_t, x_test_t, feature_names


def select_columns(matrix, column_indices: Sequence[int]):
    """Выбирает колонки из dense/sparse-матрицы."""

    if sparse.issparse(matrix):
        return matrix[:, list(column_indices)]
    return np.asarray(matrix)[:, list(column_indices)]


def resolve_feature_indices(feature_names: Sequence[str], selected_features: Sequence[str] | None) -> List[int]:
    """Преобразует имена признаков в индексы колонок после препроцессинга."""

    if selected_features is None:
        return list(range(len(feature_names)))

    feature_to_index = {name: idx for idx, name in enumerate(feature_names)}
    indices = [feature_to_index[name] for name in selected_features if name in feature_to_index]
    missing = [name for name in selected_features if name not in feature_to_index]

    if not indices:
        raise ValueError(
            "Ни один признак из selected_features не найден после препроцессинга. "
            "Проверьте feature_set и совместимость с текущими данными."
        )

    if missing:
        missing_preview = ", ".join(missing[:5])
        raise ValueError(
            "Часть признаков из feature_set отсутствует после препроцессинга: "
            f"{missing_preview}. Сначала проверьте входной feature_set."
        )

    return indices


def prepare_selected_matrices(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    x_test: pd.DataFrame,
    selected_features: Sequence[str] | None,
):
    """Готовит матрицы train/valid/test c учетом выбранного feature set."""

    preprocessor = build_preprocessor(x_train)
    x_train_t, x_valid_t, x_test_t, feature_names = transform_with_names(
        preprocessor,
        x_train,
        x_valid,
        x_test,
    )

    selected_indices = resolve_feature_indices(feature_names, selected_features)
    selected_names = [feature_names[idx] for idx in selected_indices]

    return (
        select_columns(x_train_t, selected_indices),
        select_columns(x_valid_t, selected_indices),
        select_columns(x_test_t, selected_indices),
        selected_names,
    )


def make_base_model(model_name: str, random_state: int = SEED):
    """Создает базовую модель по имени."""

    if model_name == "LogisticRegression":
        return LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        )

    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Неподдерживаемая модель: {model_name}")


def train_model_variants(
    model_name: str,
    x_train,
    y_train,
    random_state: int = SEED,
):
    """Обучает uncalibrated + calibrated variants."""

    uncalibrated = make_base_model(model_name=model_name, random_state=random_state)
    uncalibrated.fit(x_train, y_train)

    calibrated_sigmoid = CalibratedClassifierCV(
        estimator=make_base_model(model_name=model_name, random_state=random_state),
        method="sigmoid",
        cv=3,
    )
    calibrated_sigmoid.fit(x_train, y_train)

    calibrated_isotonic = CalibratedClassifierCV(
        estimator=make_base_model(model_name=model_name, random_state=random_state),
        method="isotonic",
        cv=3,
    )
    calibrated_isotonic.fit(x_train, y_train)

    return {
        "uncalibrated": uncalibrated,
        "calibrated_sigmoid": calibrated_sigmoid,
        "calibrated_isotonic": calibrated_isotonic,
    }


def get_binary_score_vector(model, x_data) -> np.ndarray:
    """Возвращает score-вектор бинарной классификации в диапазоне [0, 1]."""

    if hasattr(model, "predict_proba"):
        score = np.asarray(model.predict_proba(x_data)[:, 1], dtype=float)
        return np.clip(score, 0.0, 1.0)

    if hasattr(model, "decision_function"):
        margin = np.asarray(model.decision_function(x_data), dtype=float)
        margin = np.clip(margin, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-margin))

    pred = np.asarray(model.predict(x_data), dtype=float)
    return np.clip(pred, 0.0, 1.0)


def safe_roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """ROC-AUC с fallback в NaN для вырожденных случаев."""

    y_true_arr = np.asarray(y_true).astype(int)
    if np.unique(y_true_arr).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true_arr, y_score))


def safe_pr_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """PR-AUC с fallback в NaN для вырожденных случаев."""

    y_true_arr = np.asarray(y_true).astype(int)
    if np.unique(y_true_arr).size < 2:
        return float("nan")
    return float(average_precision_score(y_true_arr, y_score))


def compute_ece(y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10) -> float:
    """Считает Expected Calibration Error (ECE)."""

    if n_bins <= 1:
        raise ValueError("n_bins должен быть больше 1.")

    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_prob_arr = np.clip(y_prob_arr, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_arr, bins, right=True)

    total = len(y_true_arr)
    if total == 0:
        return 0.0

    ece = 0.0
    for bin_id in range(1, n_bins + 1):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        prob_mean = float(np.mean(y_prob_arr[mask]))
        target_mean = float(np.mean(y_true_arr[mask]))
        ece += (np.sum(mask) / total) * abs(prob_mean - target_mean)

    return float(ece)


def compute_expected_cost(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
    normalize: bool = True,
) -> float:
    """Считает expected cost по заданным штрафам FP/FN."""

    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true и y_pred должны иметь одинаковую длину.")

    fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
    total_cost = float(fp_cost * fp + fn_cost * fn)

    if not normalize:
        return total_cost

    n = len(y_true_arr)
    return total_cost / n if n > 0 else 0.0


def sweep_thresholds(
    y_true: Sequence[int],
    y_score: Sequence[float],
    thresholds: Iterable[float] | None = None,
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
) -> pd.DataFrame:
    """Перебирает пороги и считает метрики + expected cost."""

    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.clip(np.asarray(y_score, dtype=float), 0.0, 1.0)

    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError("y_true и y_score должны иметь одинаковую длину.")

    if thresholds is None:
        threshold_values = DEFAULT_THRESHOLD_GRID
    else:
        threshold_values = np.asarray(list(thresholds), dtype=float)
        threshold_values = np.clip(threshold_values, 0.0, 1.0)
        threshold_values = np.unique(np.round(threshold_values, 6))

    rows = []
    n = len(y_true_arr)

    for threshold in threshold_values:
        y_pred = (y_score_arr >= threshold).astype(int)

        fp = int(((y_true_arr == 0) & (y_pred == 1)).sum())
        fn = int(((y_true_arr == 1) & (y_pred == 0)).sum())

        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
                "fp_rate": float(fp / n) if n > 0 else 0.0,
                "fn_rate": float(fn / n) if n > 0 else 0.0,
                "expected_cost": float(
                    compute_expected_cost(
                        y_true=y_true_arr,
                        y_pred=y_pred,
                        fp_cost=fp_cost,
                        fn_cost=fn_cost,
                        normalize=True,
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def choose_threshold_policy(
    threshold_grid: pd.DataFrame,
    min_recall: float = DEFAULT_MIN_RECALL,
) -> pd.Series:
    """Выбирает policy по минимальному expected cost с recall-guardrail."""

    required_columns = {
        "threshold",
        "precision",
        "recall",
        "f1",
        "fp_rate",
        "fn_rate",
        "expected_cost",
    }
    if not required_columns.issubset(set(threshold_grid.columns)):
        raise ValueError(
            "threshold_grid не содержит обязательные колонки: "
            f"{sorted(required_columns)}"
        )

    eligible = threshold_grid[threshold_grid["recall"] >= min_recall].copy()
    guardrail_passed = len(eligible) > 0

    candidates = eligible if guardrail_passed else threshold_grid.copy()
    if len(candidates) == 0:
        raise ValueError("threshold_grid пустой: невозможно выбрать policy.")

    winner = (
        candidates.sort_values(
            ["expected_cost", "fn_rate", "f1", "recall", "threshold"],
            ascending=[True, True, False, False, True],
        )
        .iloc[0]
        .copy()
    )

    winner["policy_name"] = (
        f"min_cost_recall_ge_{min_recall:.2f}"
        if guardrail_passed
        else "min_cost_without_recall_guardrail"
    )
    winner["guardrail_passed"] = bool(guardrail_passed)
    return winner


def build_calibration_record(
    dataset_name: str,
    model_name: str,
    variant: str,
    split: str,
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> Dict[str, float | str]:
    """Собирает одну строку `calibration_audit`."""

    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.clip(np.asarray(y_score, dtype=float), 1e-6, 1 - 1e-6)

    return {
        "dataset": dataset_name,
        "model": model_name,
        "variant": variant,
        "split": split,
        "brier": float(np.mean((y_score_arr - y_true_arr) ** 2)),
        "log_loss": float(log_loss(y_true_arr, y_score_arr, labels=[0, 1])),
        "roc_auc": safe_roc_auc(y_true_arr, y_score_arr),
        "pr_auc": safe_pr_auc(y_true_arr, y_score_arr),
        "ece": compute_ece(y_true_arr, y_score_arr, n_bins=10),
    }


def choose_best_calibrated_variant(
    calibration_audit: pd.DataFrame,
    dataset_name: str,
) -> str:
    """Выбирает лучший calibrated-вариант по validation-метрикам."""

    subset = calibration_audit[
        (calibration_audit["dataset"] == dataset_name)
        & (calibration_audit["split"] == "validation")
        & (calibration_audit["variant"].isin(["calibrated_sigmoid", "calibrated_isotonic"]))
    ].copy()

    if subset.empty:
        raise ValueError(
            "Не найдены calibrated-результаты для dataset="
            f"{dataset_name}. Проверьте calibration_audit.csv."
        )

    winner = (
        subset.sort_values(["brier", "log_loss", "ece", "pr_auc"], ascending=[True, True, True, False])
        .iloc[0]
    )
    return str(winner["variant"])


def evaluate_policy_on_split(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float,
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
) -> Dict[str, float]:
    """Считает итоговые метрики policy на фиксированном split."""

    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.clip(np.asarray(y_score, dtype=float), 0.0, 1.0)
    y_pred = (y_score_arr >= float(threshold)).astype(int)

    expected_cost = compute_expected_cost(
        y_true=y_true_arr,
        y_pred=y_pred,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        normalize=True,
    )

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true_arr, y_score_arr),
        "pr_auc": safe_pr_auc(y_true_arr, y_score_arr),
        "expected_cost": float(expected_cost),
        "cost_per_100": float(expected_cost * 100.0),
    }


def build_segment_policy_audit(
    dataset_name: str,
    segment_feature: str,
    segment_values: Sequence,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
    n_bins: int = 4,
) -> pd.DataFrame:
    """Строит segment-level аудит ошибок и expected cost."""

    segment_series = pd.Series(segment_values).reset_index(drop=True)
    y_true_s = pd.Series(y_true).astype(int).reset_index(drop=True)
    y_pred_s = pd.Series(y_pred).astype(int).reset_index(drop=True)

    if not (len(segment_series) == len(y_true_s) == len(y_pred_s)):
        raise ValueError("segment_values, y_true и y_pred должны иметь одинаковую длину.")

    if pd.api.types.is_numeric_dtype(segment_series):
        n_unique = int(segment_series.nunique(dropna=True))
        if n_unique >= 2:
            q = min(max(2, n_bins), n_unique)
            bins = pd.qcut(segment_series, q=q, duplicates="drop")
            segment_labels = bins.astype(str).fillna("missing")
        else:
            segment_labels = pd.Series(["all"] * len(segment_series))
    else:
        segment_labels = segment_series.fillna("missing").astype(str)

    frame = pd.DataFrame(
        {
            "dataset": dataset_name,
            "segment_feature": segment_feature,
            "segment": segment_labels,
            "y_true": y_true_s,
            "y_pred": y_pred_s,
        }
    )

    rows: List[Dict[str, object]] = []
    for segment_value, group in frame.groupby("segment", dropna=False):
        n = int(len(group))
        fp = int(((group["y_true"] == 0) & (group["y_pred"] == 1)).sum())
        fn = int(((group["y_true"] == 1) & (group["y_pred"] == 0)).sum())

        group_cost = compute_expected_cost(
            y_true=group["y_true"].to_numpy(),
            y_pred=group["y_pred"].to_numpy(),
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            normalize=True,
        )

        rows.append(
            {
                "dataset": dataset_name,
                "segment_feature": segment_feature,
                "segment": str(segment_value),
                "n": n,
                "fp_rate": float(fp / n) if n > 0 else 0.0,
                "fn_rate": float(fn / n) if n > 0 else 0.0,
                "expected_cost_per_100": float(group_cost * 100.0),
            }
        )

    return pd.DataFrame(rows)


def load_feature_sets(path: Path | None = None) -> Dict[str, Dict[str, List[str]]]:
    """Загружает candidate feature set из ЛР 01."""

    feature_sets_path = path or (LAB01_OUTPUT_DIR / "feature_sets_wrapper_embedded.json")
    if not feature_sets_path.exists():
        raise FileNotFoundError(
            "Не найден feature_sets_wrapper_embedded.json из ЛР 01. "
            "Сначала выполните базовый маршрут 01-feature-importance-and-selection "
            "или убедитесь, что файл лежит в ../01-feature-importance-and-selection/outputs/."
        )

    with open(feature_sets_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_set_features(
    feature_sets: Dict[str, Dict[str, List[str]]],
    dataset_name: str,
    feature_set_name: str,
) -> List[str] | None:
    """Возвращает список признаков для feature_set или None для `full`."""

    if feature_set_name == "full":
        return None

    dataset_sets = feature_sets.get(dataset_name, {})
    if feature_set_name not in dataset_sets:
        known_sets = sorted(dataset_sets) or ["<none>"]
        raise ValueError(
            f"Для dataset={dataset_name} нет feature_set={feature_set_name}. "
            f"Доступны: {known_sets}."
        )

    return list(dataset_sets[feature_set_name])


def load_lab03_hypotheses(path: Path | None = None) -> pd.DataFrame:
    """Загружает гипотезы model+feature_set из ЛР 03."""

    baseline_path = path or (LAB03_OUTPUT_DIR / "baseline_vs_tuned_test_results.csv")
    if not baseline_path.exists():
        raise FileNotFoundError(
            "Не найден baseline_vs_tuned_test_results.csv из ЛР 03. "
            "Сначала завершите ЛР 03 или убедитесь, что файл лежит в ../03-overfitting-validation-and-hyperparameter-tuning/outputs/."
        )

    baseline = pd.read_csv(baseline_path)
    required_columns = {"dataset", "model", "feature_set", "variant", "f1"}
    if not required_columns.issubset(set(baseline.columns)):
        raise ValueError(
            "baseline_vs_tuned_test_results.csv имеет неверный формат. "
            f"Ожидались колонки: {sorted(required_columns)}."
        )

    subset = baseline[baseline["variant"] == "tuned_best"].copy()
    if subset.empty:
        subset = (
            baseline.sort_values(["dataset", "f1"], ascending=[True, False])
            .drop_duplicates(["dataset"], keep="first")
            .copy()
        )

    subset = (
        subset.sort_values(["dataset", "f1"], ascending=[True, False])
        .drop_duplicates(["dataset"], keep="first")
        .loc[:, ["dataset", "model", "feature_set"]]
        .reset_index(drop=True)
    )

    expected_datasets = sorted(DATASET_PATHS)
    observed_datasets = sorted(subset["dataset"].unique().tolist())
    if observed_datasets != expected_datasets:
        raise ValueError(
            "Гипотезы из ЛР 03 неполные или содержат неожиданные dataset. "
            f"Ожидались: {expected_datasets}, получены: {observed_datasets}."
        )

    return subset


def load_feature_sets_raw(path: Path | None = None) -> Dict[str, Dict[str, List[str]]]:
    """Публичный helper для чтения feature set JSON из ЛР 01."""

    return load_feature_sets(path=path)
