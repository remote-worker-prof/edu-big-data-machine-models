#!/usr/bin/env python3
"""Smoke-check для ЛР 02.

Проверяет:
- структуру `todo/solution` ноутбуков;
- использование upstream-артефактов из ЛР 01;
- выполнение solution-ноутбуков;
- контракты обязательных CSV-артефактов.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
MODULE_PATH = BASE_DIR / "lab_utils.py"

NOTEBOOKS_TO_EXECUTE = [
    BASE_DIR / "solutions/01_global_explanations_solution.ipynb",
    BASE_DIR / "solutions/02_local_error_analysis_solution.ipynb",
]

ALL_NOTEBOOKS = [
    BASE_DIR / "notebooks/01_global_explanations_todo.ipynb",
    BASE_DIR / "notebooks/02_local_error_analysis_todo.ipynb",
    BASE_DIR / "solutions/01_global_explanations_solution.ipynb",
    BASE_DIR / "solutions/02_local_error_analysis_solution.ipynb",
]

NOTEBOOK_STRUCTURE_RULES = {
    BASE_DIR / "notebooks/01_global_explanations_todo.ipynb": {
        "required_markers": [
            "## Цель",
            "## Что важно",
            "## Контрольные точки",
            "## Самостоятельное изучение по ходу работы",
            "## Обязательные самостоятельные задания (без образца в solutions)",
            "TODO(обязательно)",
        ],
        "forbidden_markers": [],
        "min_step_count": 4,
        "min_todo_count": 1,
    },
    BASE_DIR / "notebooks/02_local_error_analysis_todo.ipynb": {
        "required_markers": [
            "## Цель",
            "## Контрольные точки",
            "## Самостоятельное изучение по ходу работы",
            "## Обязательные самостоятельные задания (без образца в solutions)",
            "TODO(обязательно)",
        ],
        "forbidden_markers": [],
        "min_step_count": 3,
        "min_todo_count": 1,
    },
    BASE_DIR / "solutions/01_global_explanations_solution.ipynb": {
        "required_markers": [
            "# Что делаем:",
            "# Зачем:",
            "# Как читать результат:",
            "# Типичные ошибки:",
        ],
        "forbidden_markers": ["TODO(обязательно)", "NotImplementedError"],
        "min_step_count": 0,
        "min_todo_count": 0,
    },
    BASE_DIR / "solutions/02_local_error_analysis_solution.ipynb": {
        "required_markers": [
            "# Что делаем:",
            "# Зачем:",
            "# Как читать результат:",
            "# Типичные ошибки:",
        ],
        "forbidden_markers": ["TODO(обязательно)", "NotImplementedError"],
        "min_step_count": 0,
        "min_todo_count": 0,
    },
}

EXPECTED_COLUMNS = {
    "global_importance_comparison.csv": {
        "dataset",
        "model",
        "feature_set",
        "method",
        "feature",
        "score",
        "rank",
    },
    "partial_dependence_summary.csv": {
        "dataset",
        "model",
        "feature_set",
        "raw_feature",
        "grid_min",
        "grid_max",
        "score_min",
        "score_max",
        "score_delta",
        "trend",
    },
    "error_case_explanations.csv": {
        "dataset",
        "model",
        "feature_set",
        "case_group_index",
        "error_type",
        "y_true",
        "y_pred",
        "score",
        "score_source",
        "explanation_method",
        "feature",
        "importance_value",
        "detail_a",
        "detail_b",
    },
}

EXPECTED_TRENDS = {"flat", "non_decreasing", "non_increasing", "non_monotonic"}
EXPECTED_ERROR_TYPES = {"false_positive", "false_negative"}
EXPECTED_EXPLANATION_METHODS = {"perturbation", "linear_contribution"}


SPEC = importlib.util.spec_from_file_location("lab02_utils", MODULE_PATH)
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def run_solution_notebooks() -> None:
    with tempfile.TemporaryDirectory(prefix="lab02_verify_") as temp_dir:
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


def assert_notebook_structure(notebook_path: Path, rules: dict) -> None:
    all_text, markdown_text = load_notebook_text(notebook_path)
    notebook_name = notebook_path.relative_to(BASE_DIR)

    for marker in rules["required_markers"]:
        if marker not in all_text:
            raise AssertionError(f"{notebook_name} должен содержать маркер `{marker}`.")

    for marker in rules["forbidden_markers"]:
        if marker in all_text:
            raise AssertionError(f"{notebook_name} не должен содержать маркер `{marker}`.")

    step_count = markdown_text.count("## Шаг")
    if step_count < rules["min_step_count"]:
        raise AssertionError(
            f"{notebook_name} должен содержать минимум {rules['min_step_count']} шагов."
        )

    todo_count = all_text.count("TODO(обязательно)")
    if todo_count < rules["min_todo_count"]:
        raise AssertionError(
            f"{notebook_name} должен содержать минимум {rules['min_todo_count']} TODO-блоков."
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

    for notebook_path in ALL_NOTEBOOKS:
        notebook_text, _ = load_notebook_text(notebook_path)
        notebook_name = notebook_path.relative_to(BASE_DIR)
        if "load_feature_sets" not in notebook_text:
            raise AssertionError(f"{notebook_name} должен загружать feature set из ЛР 01.")
        if "load_lab01_model_results" not in notebook_text:
            raise AssertionError(f"{notebook_name} должен загружать model_results из ЛР 01.")


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


def assert_output_invariants() -> None:
    global_importance = load_output("global_importance_comparison.csv")
    pd_summary = load_output("partial_dependence_summary.csv")
    errors = load_output("error_case_explanations.csv")

    expected_datasets = sorted(lab.DATASET_PATHS)
    for name, frame in [
        ("global_importance_comparison.csv", global_importance),
        ("partial_dependence_summary.csv", pd_summary),
        ("error_case_explanations.csv", errors),
    ]:
        observed_datasets = sorted(frame["dataset"].unique().tolist())
        if observed_datasets != expected_datasets:
            raise AssertionError(
                f"{name}: ожидались dataset={expected_datasets}, получено={observed_datasets}."
            )

    if (global_importance["rank"] < 1).any():
        raise AssertionError("global_importance_comparison.csv не должен содержать rank < 1.")

    if (global_importance["feature"].astype(str).str.len() == 0).any():
        raise AssertionError("global_importance_comparison.csv содержит пустые названия признаков.")

    if "permutation" not in set(global_importance["method"].unique()):
        raise AssertionError("global_importance_comparison.csv должен содержать метод permutation.")

    if set(pd_summary["trend"].unique()) - EXPECTED_TRENDS:
        raise AssertionError("partial_dependence_summary.csv содержит неожиданные значения trend.")

    if (pd_summary["score_delta"] < 0).any():
        raise AssertionError("partial_dependence_summary.csv содержит отрицательный score_delta.")

    if set(errors["error_type"].unique()) - EXPECTED_ERROR_TYPES:
        raise AssertionError("error_case_explanations.csv содержит неожиданный error_type.")

    if set(errors["explanation_method"].unique()) - EXPECTED_EXPLANATION_METHODS:
        raise AssertionError("error_case_explanations.csv содержит неожиданный explanation_method.")

    if (errors["importance_value"] < 0).any():
        raise AssertionError("error_case_explanations.csv содержит importance_value < 0.")

    if (errors["case_group_index"] < 1).any():
        raise AssertionError("error_case_explanations.csv содержит case_group_index < 1.")


def main() -> None:
    assert_static_conditions()
    run_solution_notebooks()
    assert_output_invariants()
    print("Lab 02 smoke-check passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"[FAIL] Ошибка исполнения notebook: {exc}")
        sys.exit(1)
