#!/usr/bin/env python3
"""Smoke-check для ЛР 03.

Скрипт запускает оба solution-ноутбука и проверяет:
- наличие и контракты CSV-артефактов;
- shape-инварианты;
- базовые статические условия вокруг workflow.
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
NOTEBOOKS = [
    BASE_DIR / "solutions/01_train_validation_overfitting_solution.ipynb",
    BASE_DIR / "solutions/02_gridsearch_and_final_choice_solution.ipynb",
]
OUTPUT_DIR = BASE_DIR / "outputs"
MODULE_PATH = BASE_DIR / "lab_utils.py"

SPEC = importlib.util.spec_from_file_location("lab03_utils", MODULE_PATH)
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)

EXPECTED_COLUMNS = {
    "generalization_audit.csv": {
        "dataset",
        "feature_set",
        "model",
        "split",
        "accuracy",
        "f1",
        "roc_auc",
        "fit_time_sec",
    },
    "model_feature_set_decisions.csv": set(lab.MODEL_FEATURE_SET_DECISION_COLUMNS),
    "validation_curve_results.csv": {
        "dataset",
        "feature_set",
        "model",
        "hyperparameter",
        "param_value",
        "split",
        "accuracy",
        "f1",
        "roc_auc",
    },
    "gridsearch_results_top.csv": {
        "dataset",
        "feature_set",
        "model",
        "rank",
        "params_json",
        "mean_cv_f1",
        "std_cv_f1",
        "mean_cv_roc_auc",
        "mean_cv_accuracy",
        "mean_fit_time_sec",
    },
    "baseline_vs_tuned_test_results.csv": {
        "dataset",
        "feature_set",
        "model",
        "variant",
        "accuracy",
        "f1",
        "roc_auc",
        "fit_time_sec",
    },
}

EXPECTED_ROWS = {
    "generalization_audit.csv": 32,
    "model_feature_set_decisions.csv": 4,
    "validation_curve_results.csv": 40,
    "gridsearch_results_top.csv": 20,
    "baseline_vs_tuned_test_results.csv": 4,
}

REQUIRED_NOTE_FILES = [
    BASE_DIR / "study-notes/overfitting-vs-underfitting.md",
    BASE_DIR / "study-notes/train-validation-test-split.md",
    BASE_DIR / "study-notes/gridsearchcv-practice.md",
    BASE_DIR / "study-notes/validation-reuse-vs-nested-cv.md",
]
SECOND_NOTEBOOK_PATHS = [
    BASE_DIR / "notebooks/02_gridsearch_and_final_choice_todo.ipynb",
    BASE_DIR / "solutions/02_gridsearch_and_final_choice_solution.ipynb",
]
NOTEBOOK_STRUCTURE_RULES = {
    BASE_DIR / "notebooks/01_train_validation_overfitting_todo.ipynb": {
        "required_markers": [
            "Как проходить этот ноутбук",
            "Что делаем",
            "Почему это важно",
            "Что уже готово на входе",
            "Что должно получиться",
            "Как интерпретировать результат",
            "Проверь себя",
            "TODO(обязательно)",
        ],
        "min_step_count": 5,
        "min_check_yourself_count": 5,
        "min_todo_count": 5,
        "forbidden_markers": [],
    },
    BASE_DIR / "notebooks/02_gridsearch_and_final_choice_todo.ipynb": {
        "required_markers": [
            "Как проходить этот ноутбук",
            "Что делаем",
            "Почему это важно",
            "Что уже готово на входе",
            "Что должно получиться",
            "Как интерпретировать результат",
            "Проверь себя",
            "TODO(обязательно)",
        ],
        "min_step_count": 5,
        "min_check_yourself_count": 5,
        "min_todo_count": 5,
        "forbidden_markers": [],
    },
    BASE_DIR / "solutions/01_train_validation_overfitting_solution.ipynb": {
        "required_markers": [
            "Как проходить этот ноутбук",
            "Что делаем",
            "Почему это важно",
            "Что уже готово на входе",
            "Что должно получиться",
            "Как интерпретировать результат",
            "Проверь себя",
            "Пример вывода по шагу",
        ],
        "min_step_count": 5,
        "min_check_yourself_count": 5,
        "min_todo_count": 0,
        "forbidden_markers": ["NotImplementedError"],
    },
    BASE_DIR / "solutions/02_gridsearch_and_final_choice_solution.ipynb": {
        "required_markers": [
            "Как проходить этот ноутбук",
            "Что делаем",
            "Почему это важно",
            "Что уже готово на входе",
            "Что должно получиться",
            "Как интерпретировать результат",
            "Проверь себя",
            "Пример вывода по шагу",
        ],
        "min_step_count": 5,
        "min_check_yourself_count": 5,
        "min_todo_count": 0,
        "forbidden_markers": ["NotImplementedError"],
    },
}


def run_solution_notebooks() -> None:
    with tempfile.TemporaryDirectory(prefix="lab03_verify_") as temp_dir:
        for notebook_path in NOTEBOOKS:
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


def assert_static_conditions() -> None:
    lab_utils_text = (BASE_DIR / "lab_utils.py").read_text(encoding="utf-8")
    if "model_results.csv" in lab_utils_text:
        raise AssertionError("lab_utils.py не должен зависеть от model_results.csv.")

    for notebook_path in SECOND_NOTEBOOK_PATHS:
        notebook2_text = notebook_path.read_text(encoding="utf-8")
        notebook_name = notebook_path.relative_to(BASE_DIR)
        if "load_model_feature_set_decisions" not in notebook2_text:
            raise AssertionError(f"{notebook_name} должен читать model_feature_set_decisions.csv.")
        if "get_model_feature_set_decision" not in notebook2_text:
            raise AssertionError(f"{notebook_name} должен получать feature set через helper, а не через raw iloc.")
        if "choose_feature_set_for_model" in notebook2_text:
            raise AssertionError(f"{notebook_name} не должен скрыто пересчитывать feature set.")

    for note_path in REQUIRED_NOTE_FILES:
        if not note_path.exists():
            raise AssertionError(f"Не найден обязательный шаблон заметки: {note_path.name}")

    for notebook_path, rules in NOTEBOOK_STRUCTURE_RULES.items():
        assert_notebook_structure(notebook_path, rules)


def assert_notebook_structure(notebook_path: Path, rules: dict) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    all_text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    markdown_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
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
            f"{notebook_name} должен содержать минимум {rules['min_step_count']} шагов, найдено {step_count}."
        )

    check_yourself_count = all_text.count("Проверь себя")
    if check_yourself_count < rules["min_check_yourself_count"]:
        raise AssertionError(
            f"{notebook_name} должен содержать минимум {rules['min_check_yourself_count']} блоков `Проверь себя`."
        )

    todo_count = all_text.count("TODO(обязательно)")
    if todo_count < rules["min_todo_count"]:
        raise AssertionError(
            f"{notebook_name} должен содержать минимум {rules['min_todo_count']} явных `TODO(обязательно)`."
        )


def load_output(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise AssertionError(f"Не найден обязательный артефакт: {path}")
    df = pd.read_csv(path)
    if set(df.columns) != EXPECTED_COLUMNS[name]:
        raise AssertionError(f"Неверные колонки в {name}: {list(df.columns)}")
    if len(df) != EXPECTED_ROWS[name]:
        raise AssertionError(f"Неверное число строк в {name}: {len(df)}")
    return df


def assert_output_invariants() -> None:
    generalization = load_output("generalization_audit.csv")
    load_output("model_feature_set_decisions.csv")
    decisions = lab.load_model_feature_set_decisions(path=OUTPUT_DIR / "model_feature_set_decisions.csv")
    curves = load_output("validation_curve_results.csv")
    grid = load_output("gridsearch_results_top.csv")
    test_results = load_output("baseline_vs_tuned_test_results.csv")

    if any(size != 5 for size in grid.groupby(["dataset", "model"]).size().tolist()):
        raise AssertionError("gridsearch_results_top.csv должен содержать по 5 строк на каждую пару dataset + model.")

    if any(size != 2 for size in test_results.groupby("dataset").size().tolist()):
        raise AssertionError("baseline_vs_tuned_test_results.csv должен содержать по 2 variants на dataset.")

    if set(curves["feature_set"]) - set(decisions["selected_feature_set"]):
        raise AssertionError("validation_curve_results.csv должен использовать только выбранные feature set.")

    feature_sets = lab.load_feature_sets()
    model_count = len(lab.make_default_models())
    expected_generalization_combinations = sum(
        len(lab.list_feature_set_names(feature_sets, dataset_name)) * model_count
        for dataset_name in sorted(feature_sets)
    )
    observed_generalization_combinations = len(
        generalization[["dataset", "feature_set", "model"]].drop_duplicates()
    )
    if observed_generalization_combinations != expected_generalization_combinations:
        raise AssertionError("generalization_audit.csv должен покрывать все combinations dataset × feature_set × model.")


def main() -> None:
    assert_static_conditions()
    run_solution_notebooks()
    assert_output_invariants()
    print("Lab 03 smoke-check passed.")


if __name__ == "__main__":
    main()
