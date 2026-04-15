#!/usr/bin/env python3
"""Smoke-check для ЛР 01.

Проверяет:
- структуру `todo/solution` ноутбуков;
- выполнение solution-ноутбуков;
- контракты обязательных артефактов в `outputs/`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"

NOTEBOOKS_TO_EXECUTE = [
    BASE_DIR / "solutions/01_filter_methods_solution.ipynb",
    BASE_DIR / "solutions/02_wrapper_embedded_solution.ipynb",
    BASE_DIR / "solutions/03_model_comparison_solution.ipynb",
]

NOTEBOOK_PAIRS = [
    (
        BASE_DIR / "notebooks/01_filter_methods_todo.ipynb",
        BASE_DIR / "solutions/01_filter_methods_solution.ipynb",
    ),
    (
        BASE_DIR / "notebooks/02_wrapper_embedded_todo.ipynb",
        BASE_DIR / "solutions/02_wrapper_embedded_solution.ipynb",
    ),
    (
        BASE_DIR / "notebooks/03_model_comparison_todo.ipynb",
        BASE_DIR / "solutions/03_model_comparison_solution.ipynb",
    ),
]

NOTEBOOK_STRUCTURE_RULES = {
    BASE_DIR / "notebooks/01_filter_methods_todo.ipynb": {
        "required_markers": [
            "## Цель",
            "## Что важно",
            "## Контрольные точки",
            "## Типичные ошибки",
            "## Финальные выводы",
            "## Самостоятельное изучение по ходу работы",
            "## Обязательные самостоятельные задания (без образца в solutions)",
            "TODO(обязательно)",
        ],
        "forbidden_markers": [],
        "min_todo_count": 3,
    },
    BASE_DIR / "notebooks/02_wrapper_embedded_todo.ipynb": {
        "required_markers": [
            "## Цель",
            "## Контрольные точки",
            "## Типичные ошибки",
            "## Финальные выводы",
            "## Самостоятельное изучение по ходу работы",
            "## Обязательные самостоятельные задания (без образца в solutions)",
            "TODO(обязательно)",
        ],
        "forbidden_markers": [],
        "min_todo_count": 3,
    },
    BASE_DIR / "notebooks/03_model_comparison_todo.ipynb": {
        "required_markers": [
            "## Цель",
            "## Контрольные точки",
            "## Типичные ошибки",
            "## Финальные выводы",
            "## Самостоятельное изучение по ходу работы",
            "## Обязательные самостоятельные задания (без образца в solutions)",
            "TODO(обязательно)",
        ],
        "forbidden_markers": [],
        "min_todo_count": 3,
    },
    BASE_DIR / "solutions/01_filter_methods_solution.ipynb": {
        "required_markers": [
            "## Цель",
            "## Что важно",
            "## Контрольные точки",
            "## Типичные ошибки",
            "## Финальные выводы",
        ],
        "forbidden_markers": ["TODO(обязательно)", "NotImplementedError"],
        "min_todo_count": 0,
    },
    BASE_DIR / "solutions/02_wrapper_embedded_solution.ipynb": {
        "required_markers": [
            "## Цель",
            "## Контрольные точки",
            "## Типичные ошибки",
            "## Финальные выводы",
        ],
        "forbidden_markers": ["TODO(обязательно)", "NotImplementedError"],
        "min_todo_count": 0,
    },
    BASE_DIR / "solutions/03_model_comparison_solution.ipynb": {
        "required_markers": [
            "## Цель",
            "## Контрольные точки",
            "## Типичные ошибки",
            "## Финальные выводы",
        ],
        "forbidden_markers": ["TODO(обязательно)", "NotImplementedError"],
        "min_todo_count": 0,
    },
}

REQUIRED_COMMENT_MARKERS = [
    "# Что делаем:",
    "# Зачем:",
    "# Как читать результат:",
    "# Типичные ошибки:",
]

EXPECTED_COLUMNS = {
    "feature_ranking_filter_methods.csv": {
        "dataset",
        "method",
        "feature",
        "score",
        "rank",
    },
    "feature_ranking_wrapper_embedded.csv": {
        "dataset",
        "method",
        "feature",
        "score",
        "rank",
    },
    "model_results.csv": {
        "dataset",
        "feature_set",
        "model",
        "metric",
        "value",
        "fit_time_sec",
    },
}

EXPECTED_DATASETS = {"medical", "finance"}
EXPECTED_FILTER_METHODS = {"variance", "abs_correlation", "mutual_info", "f_classif"}
EXPECTED_WRAPPER_METHODS = {
    "rfe",
    "sfs_forward",
    "l1_logreg",
    "rf_importance",
    "permutation",
}
EXPECTED_MODEL_METRICS = {"accuracy", "f1", "roc_auc"}


def run_solution_notebooks() -> None:
    with tempfile.TemporaryDirectory(prefix="lab01_verify_") as temp_dir:
        for notebook_path in NOTEBOOKS_TO_EXECUTE:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--output",
                    notebook_path.name,
                    "--output-dir",
                    temp_dir,
                    str(notebook_path.relative_to(BASE_DIR)),
                ],
                cwd=BASE_DIR,
                check=True,
            )


def load_notebook_text(notebook_path: Path) -> tuple[str, str]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    all_text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    markdown_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    return all_text, markdown_text


def extract_step_titles(markdown_text: str) -> list[str]:
    return [line.strip() for line in markdown_text.splitlines() if line.strip().startswith("## Шаг")]


def assert_notebook_structure(notebook_path: Path, rules: dict) -> None:
    all_text, _ = load_notebook_text(notebook_path)
    notebook_name = notebook_path.relative_to(BASE_DIR)

    for marker in rules["required_markers"]:
        if marker not in all_text:
            raise AssertionError(f"{notebook_name} должен содержать маркер `{marker}`.")

    for marker in rules["forbidden_markers"]:
        if marker in all_text:
            raise AssertionError(f"{notebook_name} не должен содержать маркер `{marker}`.")

    todo_count = all_text.count("TODO(обязательно)")
    if todo_count < rules["min_todo_count"]:
        raise AssertionError(
            f"{notebook_name} должен содержать минимум {rules['min_todo_count']} TODO-блоков."
        )

    for marker in REQUIRED_COMMENT_MARKERS:
        if all_text.count(marker) < 2:
            raise AssertionError(
                f"{notebook_name} должен содержать минимум 2 вхождения маркера `{marker}`."
            )


def assert_workflow_identity() -> None:
    for todo_path, solution_path in NOTEBOOK_PAIRS:
        _, todo_markdown = load_notebook_text(todo_path)
        _, solution_markdown = load_notebook_text(solution_path)

        todo_steps = extract_step_titles(todo_markdown)
        solution_steps = extract_step_titles(solution_markdown)
        if todo_steps and todo_steps != solution_steps:
            raise AssertionError(
                "Шаги workflow в todo/solution должны совпадать: "
                f"{todo_path.name} vs {solution_path.name}."
            )


def assert_static_conditions() -> None:
    required_files = [
        BASE_DIR / "study-notes/_template.md",
        BASE_DIR / "study-notes/glossary.md",
        BASE_DIR / "report-template.md",
    ]
    for path in required_files:
        if not path.exists():
            raise AssertionError(f"Отсутствует обязательный учебный файл: {path.name}")

    for notebook_path, rules in NOTEBOOK_STRUCTURE_RULES.items():
        if not notebook_path.exists():
            raise AssertionError(f"Не найден notebook: {notebook_path}")
        assert_notebook_structure(notebook_path, rules)

    assert_workflow_identity()


def load_output(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise AssertionError(f"Не найден обязательный артефакт: {path}")
    frame = pd.read_csv(path)
    if set(frame.columns) != EXPECTED_COLUMNS[name]:
        raise AssertionError(f"Неверные колонки в {name}: {list(frame.columns)}")
    if frame.empty:
        raise AssertionError(f"{name} не должен быть пустым.")
    return frame


def load_json_output(name: str) -> dict:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise AssertionError(f"Не найден обязательный JSON-артефакт: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise AssertionError(f"{name} должен быть JSON-объектом верхнего уровня.")
    return data


def assert_output_invariants() -> None:
    ranking_filter = load_output("feature_ranking_filter_methods.csv")
    ranking_wrapper = load_output("feature_ranking_wrapper_embedded.csv")
    model_results = load_output("model_results.csv")
    shortlist = load_json_output("shortlist_filter.json")
    feature_sets = load_json_output("feature_sets_wrapper_embedded.json")

    for name, frame in [
        ("feature_ranking_filter_methods.csv", ranking_filter),
        ("feature_ranking_wrapper_embedded.csv", ranking_wrapper),
        ("model_results.csv", model_results),
    ]:
        observed_datasets = set(frame["dataset"].unique())
        if observed_datasets != EXPECTED_DATASETS:
            raise AssertionError(
                f"{name}: ожидались dataset={sorted(EXPECTED_DATASETS)}, получено={sorted(observed_datasets)}."
            )

    filter_methods = set(ranking_filter["method"].unique())
    if not EXPECTED_FILTER_METHODS.issubset(filter_methods):
        raise AssertionError(
            "feature_ranking_filter_methods.csv не покрывает обязательные методы "
            f"{sorted(EXPECTED_FILTER_METHODS)}."
        )

    wrapper_methods = set(ranking_wrapper["method"].unique())
    if not EXPECTED_WRAPPER_METHODS.issubset(wrapper_methods):
        raise AssertionError(
            "feature_ranking_wrapper_embedded.csv не покрывает обязательные методы "
            f"{sorted(EXPECTED_WRAPPER_METHODS)}."
        )

    if (ranking_filter["rank"] < 1).any() or (ranking_wrapper["rank"] < 1).any():
        raise AssertionError("ranking-таблицы не должны содержать rank < 1.")

    metric_values = set(model_results["metric"].unique())
    if metric_values != EXPECTED_MODEL_METRICS:
        raise AssertionError(
            "model_results.csv должен содержать ровно метрики "
            f"{sorted(EXPECTED_MODEL_METRICS)}."
        )

    group_metric_counts = model_results.groupby(["dataset", "feature_set", "model"])["metric"].nunique()
    if (group_metric_counts != 3).any():
        raise AssertionError(
            "model_results.csv должен содержать по 3 метрики для каждой пары dataset+feature_set+model."
        )

    if "full" not in set(model_results["feature_set"]):
        raise AssertionError("model_results.csv должен содержать сравнение для feature_set='full'.")

    bounded = model_results["metric"].isin(["accuracy", "f1", "roc_auc"])
    invalid_metric_values = ~model_results.loc[bounded, "value"].between(0, 1, inclusive="both")
    if invalid_metric_values.any():
        raise AssertionError("model_results.csv содержит metric value вне диапазона [0, 1].")

    if (model_results["fit_time_sec"] < 0).any():
        raise AssertionError("model_results.csv содержит отрицательное время обучения.")

    if set(shortlist.keys()) != EXPECTED_DATASETS:
        raise AssertionError("shortlist_filter.json должен содержать только dataset: medical и finance.")

    for dataset_name, features in shortlist.items():
        if not isinstance(features, list) or len(features) < 3:
            raise AssertionError(
                f"shortlist_filter.json: dataset={dataset_name} должен содержать список минимум из 3 признаков."
            )
        if not all(isinstance(feature, str) and feature for feature in features):
            raise AssertionError(
                f"shortlist_filter.json: dataset={dataset_name} содержит некорректный список признаков."
            )

    if set(feature_sets.keys()) != EXPECTED_DATASETS:
        raise AssertionError("feature_sets_wrapper_embedded.json должен содержать dataset: medical и finance.")

    for dataset_name, sets in feature_sets.items():
        if not isinstance(sets, dict) or len(sets) < 2:
            raise AssertionError(
                f"feature_sets_wrapper_embedded.json: dataset={dataset_name} должен содержать минимум 2 feature set."
            )
        for feature_set_name, features in sets.items():
            if not feature_set_name:
                raise AssertionError("feature_sets_wrapper_embedded.json содержит пустое имя feature set.")
            if not isinstance(features, list) or len(features) == 0:
                raise AssertionError(
                    f"feature_sets_wrapper_embedded.json: {dataset_name}/{feature_set_name} не должен быть пустым."
                )
            if not all(isinstance(feature, str) and feature for feature in features):
                raise AssertionError(
                    f"feature_sets_wrapper_embedded.json: {dataset_name}/{feature_set_name} содержит некорректные признаки."
                )


def main() -> None:
    assert_static_conditions()
    run_solution_notebooks()
    assert_output_invariants()
    print("Lab 01 smoke-check passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"[FAIL] Ошибка исполнения notebook: {exc}")
        sys.exit(1)
