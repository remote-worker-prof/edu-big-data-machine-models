"""Утилиты для ЛР 03 по переобучению, validation и тюнингу.

Модуль продолжает workflow ЛР 01: берет candidate feature set, делит данные
на train/validation/test и помогает честно сравнивать baseline и tuned-модели.
Главная методическая цель: показать студенту, почему train-метрика не равна
качеству на новых данных и как улучшать модель без утечки информации.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
LAB01_DIR = BASE_DIR.parent / "01-feature-importance-and-selection"
DATA_DIR = LAB01_DIR / "data"
LAB01_OUTPUT_DIR = LAB01_DIR / "outputs"
OUTPUT_DIR = BASE_DIR / "outputs"
SEED = 42

DATASET_PATHS = {
    "medical": DATA_DIR / "medical_cardiovascular_risk.csv",
    "finance": DATA_DIR / "finance_credit_risk.csv",
}

VALIDATION_CURVE_GRIDS = {
    "LogisticRegression": ("C", [0.01, 0.1, 1.0, 10.0, 100.0]),
    "RandomForest": ("max_depth", [2, 4, 6, 8, None]),
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Загружает CSV и проверяет наличие таргета."""

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
    """Делит данные на train/validation/test со стратификацией.

    По умолчанию получаем схему 60/20/20.
    """

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


def infer_category_levels(x: pd.DataFrame) -> Dict[str, Tuple[object, ...]]:
    """Фиксирует набор категорий для object-колонок.

    Это помогает сохранить стабильные имена one-hot колонок между фолдами
    в GridSearchCV и не потерять признаки, выбранные в ЛР 01.
    """

    _, categorical_features = infer_feature_types(x)
    levels: Dict[str, Tuple[object, ...]] = {}
    for col in categorical_features:
        values = pd.Series(x[col]).dropna().tolist()
        unique_values = tuple(dict.fromkeys(values).keys())
        levels[col] = unique_values
    return levels


def build_preprocessor(
    x: pd.DataFrame,
    category_levels: Dict[str, Tuple[object, ...]] | None = None,
) -> ColumnTransformer:
    """Строит единый препроцессор для train-таблицы."""

    numeric_features, categorical_features = infer_feature_types(x)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    encoder_kwargs = {
        "handle_unknown": "ignore",
        "sparse_output": True,
    }
    if category_levels is not None and categorical_features:
        encoder_kwargs["categories"] = [list(category_levels[col]) for col in categorical_features]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(**encoder_kwargs)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def to_dense(matrix):
    """Преобразует sparse-матрицу в dense, если требуется."""

    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def select_columns(matrix, column_indices: Sequence[int]):
    """Выбирает колонки из dense/sparse-матрицы."""

    if sparse.issparse(matrix):
        return matrix[:, list(column_indices)]
    return np.asarray(matrix)[:, list(column_indices)]


def transform_with_names(
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    x_test: pd.DataFrame,
):
    """Фитит препроцессор на train и возвращает train/valid/test + имена признаков."""

    x_train_t = preprocessor.fit_transform(x_train)
    x_valid_t = preprocessor.transform(x_valid)
    x_test_t = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return x_train_t, x_valid_t, x_test_t, feature_names


def resolve_feature_indices(feature_names: Sequence[str], selected_features: Sequence[str]) -> List[int]:
    """Находит индексы выбранных признаков по именам."""

    position_map = {name: idx for idx, name in enumerate(feature_names)}
    missing = [feature for feature in selected_features if feature not in position_map]
    if missing:
        raise ValueError(
            "Часть признаков из feature set не найдена после препроцессинга: "
            f"{missing[:5]}"
        )
    return [position_map[feature] for feature in selected_features]


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


def load_lab01_model_results(path: Path | None = None) -> pd.DataFrame:
    """Загружает model_results.csv из ЛР 01."""

    results_path = path or (LAB01_OUTPUT_DIR / "model_results.csv")
    if not results_path.exists():
        raise FileNotFoundError(
            "Не найден model_results.csv из ЛР 01. "
            "Сначала выполните 03_model_comparison первой лабораторной "
            "или положите файл в ../01-feature-importance-and-selection/outputs/."
        )
    return pd.read_csv(results_path)


def load_generalization_audit(path: Path | None = None) -> pd.DataFrame:
    """Загружает results первого ноутбука ЛР 03."""

    audit_path = path or (OUTPUT_DIR / "generalization_audit.csv")
    if not audit_path.exists():
        raise FileNotFoundError(
            "Не найден generalization_audit.csv из первого ноутбука ЛР 03. "
            "Сначала выполните 01_train_validation_overfitting или убедитесь, "
            "что файл лежит в outputs/."
        )
    return pd.read_csv(audit_path)


def choose_best_nonfull_feature_set(
    model_results: pd.DataFrame,
    feature_sets: Dict[str, Dict[str, List[str]]],
    dataset_name: str,
) -> str:
    """Выбирает лучший неполный feature set по roc_auc, затем f1, затем accuracy."""

    subset = model_results[model_results["dataset"] == dataset_name].copy()
    summary = (
        subset.pivot_table(
            index=["feature_set", "model"],
            columns="metric",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values(["roc_auc", "f1", "accuracy"], ascending=[False, False, False])
    )

    available_sets = set(feature_sets.get(dataset_name, {}))
    for _, row in summary.iterrows():
        feature_set_name = row["feature_set"]
        if feature_set_name != "full" and feature_set_name in available_sets:
            return feature_set_name

    fallback = next(iter(feature_sets.get(dataset_name, {})), None)
    if fallback is None:
        raise ValueError(f"Для dataset={dataset_name} не найдено candidate feature set в ЛР 01.")
    return fallback


def summarize_predictions(y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float]) -> Dict[str, float]:
    """Считает базовые метрики бинарной классификации."""

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)

    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_score_arr))
    except Exception:
        roc_auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "roc_auc": roc_auc,
    }


def get_binary_score_vector(model, x_data) -> Tuple[np.ndarray, str]:
    """Получает score-вектор в диапазоне [0, 1]."""

    if hasattr(model, "predict_proba"):
        score = np.asarray(model.predict_proba(x_data)[:, 1], dtype=float)
        return np.clip(score, 0.0, 1.0), "predict_proba"

    if hasattr(model, "decision_function"):
        margin = np.asarray(model.decision_function(x_data), dtype=float)
        margin = np.clip(margin, -40.0, 40.0)
        score = 1.0 / (1.0 + np.exp(-margin))
        return score, "decision_function_sigmoid"

    fallback_pred = np.asarray(model.predict(x_data), dtype=float)
    return np.clip(fallback_pred, 0.0, 1.0), "predict"


def evaluate_fitted_model(model, x_data, y_true) -> Dict[str, float]:
    """Оценивает уже обученную модель на заданном split."""

    y_pred = model.predict(x_data)
    y_score, _ = get_binary_score_vector(model, x_data)
    return summarize_predictions(y_true, y_pred, y_score)


def measure_fit_and_split_metrics(model, x_train, y_train, x_valid, y_valid):
    """Обучает модель один раз и возвращает метрики для train и validation."""

    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_time_sec = float(time.perf_counter() - start)

    train_metrics = evaluate_fitted_model(model, x_train, y_train)
    valid_metrics = evaluate_fitted_model(model, x_valid, y_valid)
    return model, fit_time_sec, train_metrics, valid_metrics


def make_default_models() -> Dict[str, object]:
    """Возвращает базовые модели ЛР 03."""

    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=350,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def make_tuning_models() -> Dict[str, object]:
    """Возвращает модели для GridSearchCV."""

    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2500,
            random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=350,
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def make_param_grids() -> Dict[str, Dict[str, List[object]]]:
    """Возвращает фиксированные сетки гиперпараметров."""

    return {
        "LogisticRegression": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
        "RandomForest": {
            "model__max_depth": [4, 8, None],
            "model__min_samples_leaf": [1, 5, 10],
            "model__class_weight": [None, "balanced_subsample"],
        },
    }


def generalization_gap(train_value: float, valid_value: float) -> float:
    """Считает разрыв между train и validation."""

    return float(train_value - valid_value)


def choose_lab03_feature_set(generalization_audit: pd.DataFrame, dataset_name: str) -> str:
    """Выбирает feature set для второго ноутбука.

    Правило:
    - максимум validation f1;
    - затем минимум f1 gap;
    - затем предпочесть неполный набор признаков при равенстве.
    """

    subset = generalization_audit[generalization_audit["dataset"] == dataset_name].copy()
    feature_rows = (
        subset.groupby(["feature_set", "split"], as_index=False)["f1"]
        .mean()
        .pivot(index="feature_set", columns="split", values="f1")
        .reset_index()
        .rename_axis(None, axis=1)
        .rename(columns={"train": "train_f1", "validation": "validation_f1"})
    )
    feature_rows["f1_gap"] = feature_rows["train_f1"] - feature_rows["validation_f1"]
    feature_rows["full_penalty"] = (feature_rows["feature_set"] == "full").astype(int)

    ordered = feature_rows.sort_values(
        ["validation_f1", "f1_gap", "full_penalty"],
        ascending=[False, True, True],
    )
    return str(ordered.iloc[0]["feature_set"])


def build_generalization_selection_summary(generalization_audit: pd.DataFrame) -> pd.DataFrame:
    """Готовит компактную summary по feature set для narrative-части."""

    rows = []
    for dataset_name in sorted(generalization_audit["dataset"].unique()):
        subset = generalization_audit[generalization_audit["dataset"] == dataset_name].copy()
        for (feature_set_name, model_name), group in subset.groupby(["feature_set", "model"]):
            metric_by_split = (
                group.pivot_table(
                    index="split",
                    values=["accuracy", "f1", "roc_auc"],
                    aggfunc="mean",
                )
                .reset_index()
                .set_index("split")
            )
            train_f1 = float(metric_by_split.loc["train", "f1"])
            valid_f1 = float(metric_by_split.loc["validation", "f1"])
            rows.append(
                {
                    "dataset": dataset_name,
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "train_f1": train_f1,
                    "validation_f1": valid_f1,
                    "f1_gap": generalization_gap(train_f1, valid_f1),
                    "train_roc_auc": float(metric_by_split.loc["train", "roc_auc"]),
                    "validation_roc_auc": float(metric_by_split.loc["validation", "roc_auc"]),
                    "roc_auc_gap": generalization_gap(
                        float(metric_by_split.loc["train", "roc_auc"]),
                        float(metric_by_split.loc["validation", "roc_auc"]),
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "validation_f1", "f1_gap"],
        ascending=[True, False, True],
    )


class PreprocessedFeatureSelector(BaseEstimator, TransformerMixin):
    """Фитит препроцессор и оставляет только выбранные transformed features."""

    def __init__(
        self,
        selected_features: Sequence[str] | None = None,
        category_levels: Dict[str, Tuple[object, ...]] | None = None,
    ):
        self.selected_features = selected_features
        self.category_levels = category_levels

    def fit(self, X, y=None):
        x_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        self.preprocessor_ = build_preprocessor(x_df, category_levels=self.category_levels)
        self.preprocessor_.fit(x_df, y)
        self.feature_names_ = self.preprocessor_.get_feature_names_out().tolist()

        if self.selected_features is None:
            self.selected_feature_names_ = list(self.feature_names_)
            self.selected_indices_ = list(range(len(self.selected_feature_names_)))
            self.missing_mask_ = [False] * len(self.selected_feature_names_)
        else:
            self.selected_feature_names_ = list(self.selected_features)
            position_map = {name: idx for idx, name in enumerate(self.feature_names_)}
            self.selected_indices_ = [
                position_map.get(name, -1) for name in self.selected_feature_names_
            ]
            self.missing_mask_ = [index == -1 for index in self.selected_indices_]
        return self

    def transform(self, X):
        x_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        transformed = self.preprocessor_.transform(x_df)
        dense_transformed = to_dense(transformed)

        if self.selected_features is None:
            return dense_transformed

        columns = []
        n_rows = dense_transformed.shape[0]
        for column_index in self.selected_indices_:
            if column_index == -1:
                columns.append(np.zeros((n_rows, 1), dtype=float))
            else:
                columns.append(dense_transformed[:, [column_index]])

        if not columns:
            return np.empty((n_rows, 0), dtype=float)
        return np.hstack(columns)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_feature_names_, dtype=object)


def build_model_pipeline(
    model,
    selected_features: Sequence[str] | None,
    category_levels: Dict[str, Tuple[object, ...]],
) -> Pipeline:
    """Строит честный Pipeline для GridSearchCV."""

    return Pipeline(
        steps=[
            (
                "features",
                PreprocessedFeatureSelector(
                    selected_features=selected_features,
                    category_levels=category_levels,
                ),
            ),
            ("model", model),
        ]
    )


def build_full_feature_pipeline(
    model,
    x_reference: pd.DataFrame,
) -> Pipeline:
    """Строит Pipeline без отбора признаков для baseline на raw-данных."""

    category_levels = infer_category_levels(x_reference)
    return Pipeline(
        steps=[
            (
                "features",
                PreprocessedFeatureSelector(
                    selected_features=None,
                    category_levels=category_levels,
                ),
            ),
            ("model", model),
        ]
    )


def top_gridsearch_rows(
    cv_results: pd.DataFrame,
    dataset_name: str,
    feature_set_name: str,
    model_name: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """Формирует top-N строк из cv_results_ в компактном формате."""

    ranked = (
        cv_results.sort_values(
            ["rank_test_f1", "mean_test_f1", "mean_test_roc_auc"],
            ascending=[True, False, False],
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    rows = []
    for rank, row in enumerate(ranked.itertuples(index=False), start=1):
        params_json = json.dumps(row.params, ensure_ascii=False, sort_keys=True, default=str)
        rows.append(
            {
                "dataset": dataset_name,
                "feature_set": feature_set_name,
                "model": model_name,
                "rank": rank,
                "params_json": params_json,
                "mean_cv_f1": float(row.mean_test_f1),
                "std_cv_f1": float(row.std_test_f1),
                "mean_cv_roc_auc": float(row.mean_test_roc_auc),
                "mean_cv_accuracy": float(row.mean_test_accuracy),
                "mean_fit_time_sec": float(row.mean_fit_time),
            }
        )
    return pd.DataFrame(rows)


def choose_validation_winner(validation_summary: pd.DataFrame, dataset_name: str) -> pd.Series:
    """Выбирает итоговую модель по validation f1, затем roc_auc, затем simplicity."""

    subset = validation_summary[validation_summary["dataset"] == dataset_name].copy()
    subset["model_priority"] = subset["model"].map({"LogisticRegression": 0, "RandomForest": 1}).fillna(99)
    ordered = subset.sort_values(
        ["validation_f1", "validation_roc_auc", "model_priority"],
        ascending=[False, False, True],
    )
    return ordered.iloc[0]


def fit_and_evaluate_pipeline(estimator, x_train, y_train, x_eval, y_eval):
    """Фитит estimator и возвращает fit_time и метрики на eval."""

    start = time.perf_counter()
    estimator.fit(x_train, y_train)
    fit_time_sec = float(time.perf_counter() - start)
    metrics = evaluate_fitted_model(estimator, x_eval, y_eval)
    metrics["fit_time_sec"] = fit_time_sec
    return estimator, metrics


def format_param_value(value: object) -> str:
    """Удобное строковое представление param_value для CSV/графиков."""

    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)
