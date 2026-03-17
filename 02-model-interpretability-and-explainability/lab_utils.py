"""Утилиты для ЛР 02 по интерпретации моделей.

Модуль продолжает workflow первой лабораторной: берет уже выбранные
feature set из ЛР 01 и помогает разбирать поведение моделей на глобальном
и локальном уровнях без тяжелых внешних библиотек.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
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

SEGMENT_FEATURES = {
    "medical": ["age", "cholesterol", "smoking_status"],
    "finance": ["credit_score", "loan_to_income", "previous_default"],
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


def train_test_split_stratified(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Делит данные на train/test со стратификацией."""

    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def infer_feature_types(x: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Возвращает числовые и категориальные признаки."""

    numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in x.columns if col not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Строит единый препроцессор для train-таблицы."""

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
    x_test: pd.DataFrame,
):
    """Фитит препроцессор и возвращает train/test + имена признаков."""

    x_train_t = preprocessor.fit_transform(x_train)
    x_test_t = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return x_train_t, x_test_t, feature_names


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


def build_segment_error_table(
    dataset_name: str,
    segment_feature: str,
    segment_values: Sequence,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_bins: int = 4,
) -> pd.DataFrame:
    """Строит компактный сегментный анализ ошибок."""

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
        errors = int((group["y_true"] != group["y_pred"]).sum())
        rows.append(
            {
                "dataset": dataset_name,
                "segment_feature": segment_feature,
                "segment": str(segment_value),
                "n": n,
                "error_rate": float(errors / n),
                "false_positive_rate": float(fp / n),
                "false_negative_rate": float(fn / n),
            }
        )

    return pd.DataFrame(rows)


def select_columns(matrix, column_indices: Sequence[int]):
    """Выбирает колонки из dense/sparse-матрицы."""

    if sparse.issparse(matrix):
        return matrix[:, list(column_indices)]
    return np.asarray(matrix)[:, list(column_indices)]


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
    """Загружает `model_results.csv` из ЛР 01."""

    results_path = path or (LAB01_OUTPUT_DIR / "model_results.csv")
    if not results_path.exists():
        raise FileNotFoundError(
            "Не найден model_results.csv из ЛР 01. "
            "Сначала выполните Notebook 3 первой лабораторной "
            "или положите файл в ../01-feature-importance-and-selection/outputs/."
        )
    return pd.read_csv(results_path)


def choose_best_nonfull_feature_set(
    model_results: pd.DataFrame,
    feature_sets: Dict[str, Dict[str, List[str]]],
    dataset_name: str,
) -> str:
    """Выбирает лучший неполный feature set по roc_auc/f1/accuracy."""

    subset = model_results[model_results["dataset"] == dataset_name].copy()
    summary = (
        subset.pivot_table(
            index=["feature_set", "model"],
            columns="metric",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values(
            ["roc_auc", "f1", "accuracy"],
            ascending=[False, False, False],
        )
    )

    available_sets = set(feature_sets.get(dataset_name, {}))
    for _, row in summary.iterrows():
        feature_set_name = str(row["feature_set"])
        if feature_set_name != "full" and feature_set_name in available_sets:
            return feature_set_name

    fallback = next(iter(feature_sets.get(dataset_name, {})), None)
    if fallback is None:
        raise ValueError(f"Не найден ни один feature set для dataset={dataset_name}.")
    return fallback


def choose_best_model_config(
    model_results: pd.DataFrame,
    feature_sets: Dict[str, Dict[str, List[str]]],
    dataset_name: str,
) -> Dict[str, Any]:
    """Выбирает лучшую пару model + feature_set для локального анализа."""

    subset = model_results[model_results["dataset"] == dataset_name].copy()
    subset = subset[subset["feature_set"] != "full"].copy()
    summary = (
        subset.pivot_table(
            index=["feature_set", "model"],
            columns="metric",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values(
            ["roc_auc", "f1", "accuracy"],
            ascending=[False, False, False],
        )
    )

    available_sets = set(feature_sets.get(dataset_name, {}))
    for _, row in summary.iterrows():
        feature_set_name = str(row["feature_set"])
        if feature_set_name in available_sets:
            return {
                "dataset": dataset_name,
                "feature_set": feature_set_name,
                "model": str(row["model"]),
                "roc_auc": float(row["roc_auc"]),
                "f1": float(row["f1"]),
                "accuracy": float(row["accuracy"]),
            }

    feature_set_name = choose_best_nonfull_feature_set(model_results, feature_sets, dataset_name)
    return {
        "dataset": dataset_name,
        "feature_set": feature_set_name,
        "model": "LogisticRegression",
        "roc_auc": float("nan"),
        "f1": float("nan"),
        "accuracy": float("nan"),
    }


def _resolve_raw_column(feature_name: str, raw_columns: Sequence[str]) -> str:
    """Находит исходную raw-колонку для transformed feature name."""

    candidates = sorted(raw_columns, key=len, reverse=True)
    for col in candidates:
        if feature_name == f"num__{col}" or feature_name.startswith(f"cat__{col}_"):
            return col
    raise ValueError(f"Не удалось сопоставить transformed feature `{feature_name}` с raw column.")


def resolve_raw_columns(
    selected_features: Sequence[str],
    raw_columns: Sequence[str],
) -> List[str]:
    """Переходит от transformed feature names к raw columns без дубликатов."""

    resolved: List[str] = []
    for feature_name in selected_features:
        raw_col = _resolve_raw_column(feature_name, raw_columns)
        if raw_col not in resolved:
            resolved.append(raw_col)
    return resolved


def fit_selected_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected_features: Sequence[str],
    model,
) -> Dict[str, Any]:
    """Фитит модель на подмножестве transformed-признаков из ЛР 01."""

    raw_columns = resolve_raw_columns(selected_features, x_train.columns.tolist())
    x_train_subset = x_train[raw_columns].copy()
    x_test_subset = x_test[raw_columns].copy()

    preprocessor = build_preprocessor(x_train_subset)
    x_train_t_all, x_test_t_all, feature_names = transform_with_names(
        preprocessor,
        x_train_subset,
        x_test_subset,
    )

    missing = [feature for feature in selected_features if feature not in feature_names]
    if missing:
        raise ValueError(f"Не найдены transformed features после fit: {missing}")

    selected_idx = [feature_names.index(feature) for feature in selected_features]
    x_train_model = select_columns(x_train_t_all, selected_idx)
    x_test_model = select_columns(x_test_t_all, selected_idx)

    model.fit(x_train_model, y_train)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "raw_columns": raw_columns,
        "selected_idx": selected_idx,
        "all_feature_names": feature_names,
        "selected_feature_names": list(selected_features),
        "x_train_raw": x_train_subset.reset_index(drop=True),
        "x_test_raw": x_test_subset.reset_index(drop=True),
        "x_train_model": x_train_model,
        "x_test_model": x_test_model,
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
    }


def transform_raw_for_artifact(artifact: Dict[str, Any], x_raw: pd.DataFrame):
    """Применяет препроцессинг и отбор transformed columns к новым данным."""

    subset = x_raw[artifact["raw_columns"]].copy()
    x_t_all = artifact["preprocessor"].transform(subset)
    return select_columns(x_t_all, artifact["selected_idx"])


def score_raw_rows(artifact: Dict[str, Any], x_raw: pd.DataFrame) -> np.ndarray:
    """Считает score для raw-таблицы через сохраненный препроцессинг."""

    x_model = transform_raw_for_artifact(artifact, x_raw)
    scores, _ = get_binary_score_vector(artifact["model"], x_model)
    return scores


def rank_desc(scores: np.ndarray) -> np.ndarray:
    """Возвращает ранги при сортировке по убыванию."""

    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def _build_single_importance_frame(
    dataset_name: str,
    model_name: str,
    feature_set_name: str,
    method_name: str,
    feature_names: Sequence[str],
    scores: Sequence[float],
) -> pd.DataFrame:
    scores_arr = np.asarray(scores, dtype=float)
    frame = pd.DataFrame(
        {
            "dataset": dataset_name,
            "model": model_name,
            "feature_set": feature_set_name,
            "method": method_name,
            "feature": list(feature_names),
            "score": scores_arr,
            "rank": rank_desc(scores_arr),
        }
    )
    return frame.sort_values("rank").reset_index(drop=True)


def build_global_importance_table(
    artifact: Dict[str, Any],
    dataset_name: str,
    model_name: str,
    feature_set_name: str,
    n_repeats: int = 8,
) -> pd.DataFrame:
    """Строит global importance для native importance и permutation."""

    model = artifact["model"]
    feature_names = artifact["selected_feature_names"]
    frames: List[pd.DataFrame] = []

    if hasattr(model, "coef_"):
        coef_scores = np.abs(np.asarray(model.coef_).ravel())
        frames.append(
            _build_single_importance_frame(
                dataset_name,
                model_name,
                feature_set_name,
                "coef_abs",
                feature_names,
                coef_scores,
            )
        )

    if hasattr(model, "feature_importances_"):
        frames.append(
            _build_single_importance_frame(
                dataset_name,
                model_name,
                feature_set_name,
                "feature_importance",
                feature_names,
                np.asarray(model.feature_importances_, dtype=float),
            )
        )

    perm = permutation_importance(
        model,
        artifact["x_test_model"],
        artifact["y_test"],
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=SEED,
    )
    frames.append(
        _build_single_importance_frame(
            dataset_name,
            model_name,
            feature_set_name,
            "permutation",
            feature_names,
            perm.importances_mean,
        )
    )

    return pd.concat(frames, ignore_index=True)


def choose_numeric_features_for_pdp(
    artifact: Dict[str, Any],
    top_n: int = 3,
) -> List[str]:
    """Возвращает top-n raw numeric features, присутствующие в selected set."""

    numeric_cols, _ = infer_feature_types(artifact["x_train_raw"])
    selected_raw = artifact["raw_columns"]
    candidates = [col for col in selected_raw if col in numeric_cols]
    return candidates[:top_n]


def _trend_label(scores: Sequence[float]) -> str:
    deltas = np.diff(np.asarray(scores, dtype=float))
    if len(deltas) == 0:
        return "flat"
    if np.all(deltas >= -1e-9):
        return "non_decreasing"
    if np.all(deltas <= 1e-9):
        return "non_increasing"
    return "non_monotonic"


def build_partial_dependence_summary(
    artifact: Dict[str, Any],
    dataset_name: str,
    model_name: str,
    feature_set_name: str,
    raw_features: Sequence[str] | None = None,
    grid_points: int = 9,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Строит curve-таблицу и summary по one-way partial dependence."""

    features = list(raw_features) if raw_features is not None else choose_numeric_features_for_pdp(artifact)
    curve_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    x_reference = artifact["x_test_raw"].copy()

    for raw_feature in features:
        values = artifact["x_train_raw"][raw_feature].dropna()
        if values.empty:
            continue

        quantiles = np.linspace(0.1, 0.9, grid_points)
        grid = np.unique(np.quantile(values, quantiles))
        mean_scores: List[float] = []

        for grid_value in grid:
            modified = x_reference.copy()
            modified[raw_feature] = grid_value
            mean_score = float(score_raw_rows(artifact, modified).mean())
            mean_scores.append(mean_score)
            curve_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "raw_feature": raw_feature,
                    "grid_value": float(grid_value),
                    "mean_score": mean_score,
                }
            )

        summary_rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "feature_set": feature_set_name,
                "raw_feature": raw_feature,
                "grid_min": float(np.min(grid)),
                "grid_max": float(np.max(grid)),
                "score_min": float(np.min(mean_scores)),
                "score_max": float(np.max(mean_scores)),
                "score_delta": float(np.max(mean_scores) - np.min(mean_scores)),
                "trend": _trend_label(mean_scores),
            }
        )

    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)


def summarize_model_quality(artifact: Dict[str, Any]) -> Dict[str, float]:
    """Возвращает accuracy/f1/roc_auc на test-части."""

    preds = artifact["model"].predict(artifact["x_test_model"])
    scores, _ = get_binary_score_vector(artifact["model"], artifact["x_test_model"])
    return {
        "accuracy": float(accuracy_score(artifact["y_test"], preds)),
        "f1": float(f1_score(artifact["y_test"], preds)),
        "roc_auc": float(roc_auc_score(artifact["y_test"], scores)),
    }


def _reference_value(series: pd.Series) -> Any:
    if pd.api.types.is_numeric_dtype(series):
        return float(series.median())
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return "missing"


def build_reference_profile(x_train_raw: pd.DataFrame) -> Dict[str, Any]:
    """Формирует reference profile для perturbation-объяснений."""

    return {col: _reference_value(x_train_raw[col]) for col in x_train_raw.columns}


def explain_case_by_perturbation(
    artifact: Dict[str, Any],
    case_row: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Локальное объяснение через замену признака на reference value."""

    baseline_score = float(score_raw_rows(artifact, case_row).ravel()[0])
    reference_profile = build_reference_profile(artifact["x_train_raw"])
    rows: List[Dict[str, object]] = []

    for raw_feature in artifact["raw_columns"]:
        modified = case_row.copy()
        original_value = modified.iloc[0][raw_feature]
        reference_value = reference_profile[raw_feature]
        modified.loc[modified.index[0], raw_feature] = reference_value
        adjusted_score = float(score_raw_rows(artifact, modified).ravel()[0])
        rows.append(
            {
                "feature": raw_feature,
                "original_value": original_value,
                "reference_value": reference_value,
                "baseline_score": baseline_score,
                "adjusted_score": adjusted_score,
                "importance_value": float(abs(baseline_score - adjusted_score)),
            }
        )

    frame = pd.DataFrame(rows).sort_values("importance_value", ascending=False)
    return frame.head(top_n).reset_index(drop=True)


def explain_linear_case(
    artifact: Dict[str, Any],
    case_row: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Локальное объяснение через signed contribution для линейной модели."""

    model = artifact["model"]
    if not hasattr(model, "coef_"):
        return pd.DataFrame(
            columns=[
                "feature",
                "signed_contribution",
                "importance_value",
            ]
        )

    x_case = transform_raw_for_artifact(artifact, case_row)
    dense_case = x_case.toarray().ravel() if sparse.issparse(x_case) else np.asarray(x_case).ravel()
    coef = np.asarray(model.coef_).ravel()
    contributions = dense_case * coef
    frame = pd.DataFrame(
        {
            "feature": artifact["selected_feature_names"],
            "signed_contribution": contributions,
            "importance_value": np.abs(contributions),
        }
    )
    return frame.sort_values("importance_value", ascending=False).head(top_n).reset_index(drop=True)


def build_error_case_explanations(
    artifact: Dict[str, Any],
    dataset_name: str,
    model_name: str,
    feature_set_name: str,
    top_n_per_error_type: int = 3,
    top_features_per_case: int = 3,
) -> pd.DataFrame:
    """Собирает локальные объяснения для самых уверенных ошибок."""

    preds = artifact["model"].predict(artifact["x_test_model"])
    scores, score_source = get_binary_score_vector(artifact["model"], artifact["x_test_model"])

    frame = artifact["x_test_raw"].copy()
    frame["y_true"] = artifact["y_test"].to_numpy()
    frame["y_pred"] = np.asarray(preds, dtype=int)
    frame["score"] = np.asarray(scores, dtype=float)
    frame["confidence"] = np.abs(frame["score"] - 0.5)
    frame["error_type"] = np.where(
        (frame["y_true"] == 0) & (frame["y_pred"] == 1),
        "false_positive",
        np.where(
            (frame["y_true"] == 1) & (frame["y_pred"] == 0),
            "false_negative",
            "correct",
        ),
    )
    errors = frame[frame["error_type"] != "correct"].copy()

    rows: List[Dict[str, object]] = []
    for error_type in ["false_positive", "false_negative"]:
        subset = errors[errors["error_type"] == error_type].sort_values(
            "confidence",
            ascending=False,
        )
        for case_index, (_, row) in enumerate(subset.head(top_n_per_error_type).iterrows(), start=1):
            case_df = row[artifact["raw_columns"]].to_frame().T.reset_index(drop=True)
            perturb = explain_case_by_perturbation(artifact, case_df, top_n=top_features_per_case)
            for _, exp_row in perturb.iterrows():
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "case_group_index": case_index,
                        "error_type": error_type,
                        "y_true": int(row["y_true"]),
                        "y_pred": int(row["y_pred"]),
                        "score": float(row["score"]),
                        "score_source": score_source,
                        "explanation_method": "perturbation",
                        "feature": str(exp_row["feature"]),
                        "importance_value": float(exp_row["importance_value"]),
                        "detail_a": str(exp_row["original_value"]),
                        "detail_b": str(exp_row["reference_value"]),
                    }
                )

            linear = explain_linear_case(artifact, case_df, top_n=top_features_per_case)
            for _, exp_row in linear.iterrows():
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "case_group_index": case_index,
                        "error_type": error_type,
                        "y_true": int(row["y_true"]),
                        "y_pred": int(row["y_pred"]),
                        "score": float(row["score"]),
                        "score_source": score_source,
                        "explanation_method": "linear_contribution",
                        "feature": str(exp_row["feature"]),
                        "importance_value": float(exp_row["importance_value"]),
                        "detail_a": f"{float(exp_row['signed_contribution']):.6f}",
                        "detail_b": "",
                    }
                )

    return pd.DataFrame(rows)


def build_default_segment_tables(
    artifact: Dict[str, Any],
    dataset_name: str,
) -> pd.DataFrame:
    """Строит сегментный анализ для фиксированного набора учебных признаков."""

    preds = artifact["model"].predict(artifact["x_test_model"])
    rows: List[pd.DataFrame] = []
    for feature_name in SEGMENT_FEATURES.get(dataset_name, []):
        if feature_name not in artifact["x_test_raw"].columns:
            continue
        table = build_segment_error_table(
            dataset_name=dataset_name,
            segment_feature=feature_name,
            segment_values=artifact["x_test_raw"][feature_name],
            y_true=artifact["y_test"],
            y_pred=preds,
        )
        rows.append(table)
    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "segment_feature",
                "segment",
                "n",
                "error_rate",
                "false_positive_rate",
                "false_negative_rate",
            ]
        )
    return pd.concat(rows, ignore_index=True)
