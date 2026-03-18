"""Утилиты для ЛР 03 по переобучению, validation и честному тюнингу.

ЛР 03 продолжает курс после ЛР 01, но принимает все решения заново
на текущем split train/validation/test. Candidate feature set из ЛР 01
используются как гипотезы, а не как готовый победитель.
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

MODEL_FEATURE_SET_DECISION_COLUMNS = [
    "dataset",
    "model",
    "selected_feature_set",
    "train_f1",
    "validation_f1",
    "f1_gap",
    "abs_f1_gap",
    "tie_break_reason",
]


def expected_model_feature_set_decision_pairs(
    expected_datasets: Sequence[str] | None = None,
    expected_models: Sequence[str] | None = None,
) -> List[Tuple[str, str]]:
    """Возвращает ожидаемые пары dataset + model для decisions CSV."""

    dataset_names = list(expected_datasets) if expected_datasets is not None else sorted(DATASET_PATHS)
    model_names = list(expected_models) if expected_models is not None else sorted(make_tuning_models())
    return sorted((dataset_name, model_name) for dataset_name in dataset_names for model_name in model_names)


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


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Строит единый препроцессор, который обучается на текущем fit."""

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
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
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


def load_model_feature_set_decisions(
    path: Path | None = None,
    expected_datasets: Sequence[str] | None = None,
    expected_models: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Загружает и валидирует итоговый выбор feature set из первого ноутбука ЛР 03."""

    decisions_path = path or (OUTPUT_DIR / "model_feature_set_decisions.csv")
    if not decisions_path.exists():
        _raise_missing_model_feature_set_decisions(path)

    return validate_model_feature_set_decisions(
        decisions=pd.read_csv(decisions_path),
        expected_datasets=expected_datasets,
        expected_models=expected_models,
    )


def _raise_missing_model_feature_set_decisions(path: Path | None = None):
    decisions_path = path or (OUTPUT_DIR / "model_feature_set_decisions.csv")
    raise FileNotFoundError(
        "Не найден model_feature_set_decisions.csv из первого ноутбука ЛР 03. "
        "Сначала выполните 01_train_validation_overfitting до экспортной ячейки "
        f"или убедитесь, что файл лежит в {decisions_path.parent}/."
    )


def validate_model_feature_set_decisions(
    decisions: pd.DataFrame,
    expected_datasets: Sequence[str] | None = None,
    expected_models: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Проверяет контракт model_feature_set_decisions.csv и возвращает нормализованный DataFrame."""

    expected_columns = set(MODEL_FEATURE_SET_DECISION_COLUMNS)
    actual_columns = set(decisions.columns)
    if actual_columns != expected_columns:
        missing_columns = sorted(expected_columns - actual_columns)
        extra_columns = sorted(actual_columns - expected_columns)
        raise ValueError(
            "model_feature_set_decisions.csv имеет неверные колонки. "
            f"Ожидались {sorted(expected_columns)}, получены {list(decisions.columns)}. "
            f"Отсутствуют: {missing_columns or 'нет'}. Лишние: {extra_columns or 'нет'}. "
            "Сначала завершите 01_train_validation_overfitting до экспортной ячейки "
            "или пересохраните CSV без ручного редактирования."
        )

    duplicate_mask = decisions.duplicated(["dataset", "model"], keep=False)
    if duplicate_mask.any():
        duplicate_pairs = sorted(
            {
                f"{row.dataset}/{row.model}"
                for row in decisions.loc[duplicate_mask, ["dataset", "model"]].itertuples(index=False)
            }
        )
        raise ValueError(
            "model_feature_set_decisions.csv должен содержать уникальные пары dataset + model, "
            f"но найдены дубликаты: {duplicate_pairs}. "
            "Сначала завершите 01_train_validation_overfitting заново или пересохраните CSV."
        )

    expected_pairs = set(
        expected_model_feature_set_decision_pairs(
            expected_datasets=expected_datasets,
            expected_models=expected_models,
        )
    )
    observed_pairs = {
        (row.dataset, row.model) for row in decisions.loc[:, ["dataset", "model"]].itertuples(index=False)
    }
    missing_pairs = sorted(expected_pairs - observed_pairs)
    if missing_pairs:
        missing_pairs_text = ", ".join(f"{dataset}/{model}" for dataset, model in missing_pairs)
        raise ValueError(
            "model_feature_set_decisions.csv неполон: отсутствуют строки для "
            f"{missing_pairs_text}. "
            "Сначала завершите 01_train_validation_overfitting до экспортной ячейки "
            "или пересохраните CSV после полного выполнения первого ноутбука."
        )

    return (
        decisions.loc[:, MODEL_FEATURE_SET_DECISION_COLUMNS]
        .sort_values(["dataset", "model"])
        .reset_index(drop=True)
    )


def get_model_feature_set_decision(
    decisions: pd.DataFrame,
    dataset_name: str,
    model_name: str,
) -> pd.Series:
    """Возвращает одну валидную строку decisions для пары dataset + model."""

    subset = decisions[
        (decisions["dataset"] == dataset_name) & (decisions["model"] == model_name)
    ].copy()
    if subset.empty:
        raise ValueError(
            "В model_feature_set_decisions.csv нет строки для "
            f"{dataset_name}/{model_name}. "
            "Сначала завершите 01_train_validation_overfitting до экспортной ячейки "
            "или пересохраните CSV."
        )
    if len(subset) != 1:
        raise ValueError(
            "Для пары "
            f"{dataset_name}/{model_name} ожидалась ровно одна строка в model_feature_set_decisions.csv. "
            "Сначала пересохраните CSV из первого ноутбука без ручного редактирования."
        )
    return subset.iloc[0]


def list_feature_set_names(
    feature_sets: Dict[str, Dict[str, List[str]]],
    dataset_name: str,
) -> List[str]:
    """Возвращает все candidate feature set для dataset вместе с `full`."""

    dataset_feature_sets = feature_sets.get(dataset_name, {})
    if not dataset_feature_sets:
        raise ValueError(f"Для dataset={dataset_name} не найдено candidate feature set в ЛР 01.")
    return ["full"] + sorted(dataset_feature_sets)


def get_feature_set_features(
    feature_sets: Dict[str, Dict[str, List[str]]],
    dataset_name: str,
    feature_set_name: str,
) -> Sequence[str] | None:
    """Возвращает transformed features для заданного feature set.

    Для `full` возвращается `None`, что означает использовать все признаки
    после текущего preprocessing.
    """

    if feature_set_name == "full":
        return None
    return list(feature_sets[dataset_name][feature_set_name])


def summarize_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
) -> Dict[str, float]:
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


def _feature_set_summary_for_model(
    generalization_audit: pd.DataFrame,
    dataset_name: str,
    model_name: str,
) -> pd.DataFrame:
    """Готовит сводную таблицу по feature set для dataset + model."""

    subset = generalization_audit[
        (generalization_audit["dataset"] == dataset_name)
        & (generalization_audit["model"] == model_name)
    ].copy()
    if subset.empty:
        raise ValueError(
            f"В generalization_audit нет строк для dataset={dataset_name} и model={model_name}."
        )

    summary = (
        subset.pivot_table(
            index="feature_set",
            columns="split",
            values=["accuracy", "f1", "roc_auc"],
            aggfunc="mean",
        )
        .sort_index(axis=1)
        .reset_index()
    )
    summary.columns = [
        "feature_set" if column == ("feature_set", "") else f"{column[1]}_{column[0]}"
        for column in summary.columns
    ]
    summary["f1_gap"] = summary["train_f1"] - summary["validation_f1"]
    summary["abs_f1_gap"] = summary["f1_gap"].abs()
    summary["full_penalty"] = (summary["feature_set"] == "full").astype(int)
    return summary


def explain_feature_set_tie_break(feature_rows: pd.DataFrame) -> str:
    """Кратко объясняет, каким правилом был выбран winner."""

    top_validation_f1 = feature_rows["validation_f1"].max()
    remaining = feature_rows[np.isclose(feature_rows["validation_f1"], top_validation_f1)].copy()
    if len(remaining) == 1:
        return "best validation_f1"

    best_abs_gap = remaining["abs_f1_gap"].min()
    remaining = remaining[np.isclose(remaining["abs_f1_gap"], best_abs_gap)].copy()
    if len(remaining) == 1:
        return "tie on validation_f1 -> min abs_f1_gap"

    best_full_penalty = remaining["full_penalty"].min()
    remaining = remaining[remaining["full_penalty"] == best_full_penalty].copy()
    if len(remaining) == 1:
        return "tie on validation_f1 and abs_f1_gap -> prefer non-full"

    return "tie on validation_f1, abs_f1_gap and full/non-full -> lexicographic order"


def _select_feature_set_winner_row(feature_rows: pd.DataFrame) -> pd.Series:
    """Применяет tie-break rules последовательно и устойчиво к float-шуму."""

    remaining = feature_rows.copy()

    top_validation_f1 = remaining["validation_f1"].max()
    remaining = remaining[np.isclose(remaining["validation_f1"], top_validation_f1)].copy()

    best_abs_gap = remaining["abs_f1_gap"].min()
    remaining = remaining[np.isclose(remaining["abs_f1_gap"], best_abs_gap)].copy()

    best_full_penalty = remaining["full_penalty"].min()
    remaining = remaining[remaining["full_penalty"] == best_full_penalty].copy()

    ordered = remaining.sort_values("feature_set", ascending=True).reset_index(drop=True)
    return ordered.iloc[0]


def choose_feature_set_for_model(
    generalization_audit: pd.DataFrame,
    dataset_name: str,
    model_name: str,
) -> str:
    """Выбирает feature set для заданной пары dataset + model.

    Правило:
    - максимум validation f1;
    - затем минимум abs(train f1 - validation f1);
    - затем предпочесть неполный набор признаков;
    - затем лексикографически меньший feature set.
    """

    feature_rows = _feature_set_summary_for_model(
        generalization_audit=generalization_audit,
        dataset_name=dataset_name,
        model_name=model_name,
    )
    winner = _select_feature_set_winner_row(feature_rows)
    return str(winner["feature_set"])


def build_generalization_selection_summary(generalization_audit: pd.DataFrame) -> pd.DataFrame:
    """Готовит компактную summary по всем feature set для narrative-части."""

    rows = []
    for dataset_name in sorted(generalization_audit["dataset"].unique()):
        dataset_subset = generalization_audit[generalization_audit["dataset"] == dataset_name]
        for model_name in sorted(dataset_subset["model"].unique()):
            feature_rows = _feature_set_summary_for_model(
                generalization_audit=generalization_audit,
                dataset_name=dataset_name,
                model_name=model_name,
            )
            for row in feature_rows.to_dict("records"):
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "feature_set": row["feature_set"],
                        "train_f1": float(row["train_f1"]),
                        "validation_f1": float(row["validation_f1"]),
                        "f1_gap": float(row["f1_gap"]),
                        "abs_f1_gap": float(row["abs_f1_gap"]),
                        "train_roc_auc": float(row["train_roc_auc"]),
                        "validation_roc_auc": float(row["validation_roc_auc"]),
                        "roc_auc_gap": float(row["train_roc_auc"] - row["validation_roc_auc"]),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "model", "validation_f1", "abs_f1_gap", "feature_set"],
        ascending=[True, True, False, True, True],
    )


def build_model_feature_set_decisions(generalization_audit: pd.DataFrame) -> pd.DataFrame:
    """Возвращает итоговый выбор feature set для каждой пары dataset + model."""

    rows = []
    for dataset_name in sorted(generalization_audit["dataset"].unique()):
        dataset_subset = generalization_audit[generalization_audit["dataset"] == dataset_name]
        for model_name in sorted(dataset_subset["model"].unique()):
            feature_rows = _feature_set_summary_for_model(
                generalization_audit=generalization_audit,
                dataset_name=dataset_name,
                model_name=model_name,
            )
            winner = _select_feature_set_winner_row(feature_rows)
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "selected_feature_set": str(winner["feature_set"]),
                    "train_f1": float(winner["train_f1"]),
                    "validation_f1": float(winner["validation_f1"]),
                    "f1_gap": float(winner["f1_gap"]),
                    "abs_f1_gap": float(winner["abs_f1_gap"]),
                    "tie_break_reason": explain_feature_set_tie_break(feature_rows),
                }
            )
    return (
        pd.DataFrame(rows, columns=MODEL_FEATURE_SET_DECISION_COLUMNS)
        .sort_values(["dataset", "model"])
        .reset_index(drop=True)
    )


class PreprocessedFeatureSelector(BaseEstimator, TransformerMixin):
    """Фитит препроцессор и оставляет только выбранные transformed features."""

    def __init__(self, selected_features: Sequence[str] | None = None):
        self.selected_features = selected_features

    def fit(self, X, y=None):
        x_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        self.preprocessor_ = build_preprocessor(x_df)
        self.preprocessor_.fit(x_df, y)
        self.feature_names_ = self.preprocessor_.get_feature_names_out().tolist()

        if self.selected_features is None:
            self.selected_feature_names_ = list(self.feature_names_)
            self.selected_indices_ = list(range(len(self.selected_feature_names_)))
        else:
            self.selected_feature_names_ = list(self.selected_features)
            position_map = {name: idx for idx, name in enumerate(self.feature_names_)}
            self.selected_indices_ = [
                position_map.get(name, -1) for name in self.selected_feature_names_
            ]
        return self

    def transform(self, X):
        x_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        dense_transformed = to_dense(self.preprocessor_.transform(x_df))

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


def build_model_pipeline(model, selected_features: Sequence[str] | None) -> Pipeline:
    """Строит Pipeline, в котором preprocessing обучается внутри fit."""

    return Pipeline(
        steps=[
            ("features", PreprocessedFeatureSelector(selected_features=selected_features)),
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
    subset["model_priority"] = (
        subset["model"].map({"LogisticRegression": 0, "RandomForest": 1}).fillna(99)
    )
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
