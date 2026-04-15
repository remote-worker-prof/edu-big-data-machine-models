#!/usr/bin/env python3
"""Проверка обязательных артефактов сдачи по ЛР 01-05.

Использование:
- python scripts/check_submission.py --lab 01
- python scripts/check_submission.py --lab all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
EXPECTED_DATASETS = {"medical", "finance"}

LAB_CONFIGS = {
    "01": {
        "path": ROOT_DIR / "01-feature-importance-and-selection",
        "todo_notebooks": [
            "notebooks/01_filter_methods_todo.ipynb",
            "notebooks/02_wrapper_embedded_todo.ipynb",
            "notebooks/03_model_comparison_todo.ipynb",
        ],
        "csv_outputs": {
            "outputs/feature_ranking_filter_methods.csv": {
                "dataset",
                "method",
                "feature",
                "score",
                "rank",
            },
            "outputs/feature_ranking_wrapper_embedded.csv": {
                "dataset",
                "method",
                "feature",
                "score",
                "rank",
            },
            "outputs/model_results.csv": {
                "dataset",
                "feature_set",
                "model",
                "metric",
                "value",
                "fit_time_sec",
            },
            "outputs/filter_stability_grid.csv": {
                "dataset",
                "variance_threshold",
                "top_n",
                "shortlist_json",
                "overlap_with_baseline",
            },
            "outputs/method_agreement_long.csv": {
                "dataset",
                "method_a",
                "method_b",
                "top_k",
                "overlap_count",
                "jaccard",
            },
            "outputs/selection_stability.csv": {
                "dataset",
                "method",
                "feature",
                "selected_count",
                "total_runs",
                "stability_rate",
            },
            "outputs/threshold_tuning_results.csv": {
                "dataset",
                "model",
                "feature_set",
                "threshold",
                "precision",
                "recall",
                "f1",
            },
            "outputs/cv_stability_results.csv": {
                "dataset",
                "model",
                "feature_set",
                "fold",
                "accuracy",
                "f1",
                "roc_auc",
            },
            "outputs/error_by_segment.csv": {
                "dataset",
                "segment_feature",
                "segment",
                "n",
                "error_rate",
                "false_positive_rate",
                "false_negative_rate",
            },
        },
        "json_outputs": [
            "outputs/shortlist_filter.json",
            "outputs/feature_sets_wrapper_embedded.json",
        ],
    },
    "02": {
        "path": ROOT_DIR / "02-model-interpretability-and-explainability",
        "todo_notebooks": [
            "notebooks/01_global_explanations_todo.ipynb",
            "notebooks/02_local_error_analysis_todo.ipynb",
        ],
        "csv_outputs": {
            "outputs/global_importance_comparison.csv": {
                "dataset",
                "model",
                "feature_set",
                "method",
                "feature",
                "score",
                "rank",
            },
            "outputs/partial_dependence_summary.csv": {
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
            "outputs/error_case_explanations.csv": {
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
        },
        "json_outputs": [],
    },
    "03": {
        "path": ROOT_DIR / "03-overfitting-validation-and-hyperparameter-tuning",
        "todo_notebooks": [
            "notebooks/01_train_validation_overfitting_todo.ipynb",
            "notebooks/02_gridsearch_and_final_choice_todo.ipynb",
        ],
        "csv_outputs": {
            "outputs/generalization_audit.csv": {
                "dataset",
                "feature_set",
                "model",
                "split",
                "accuracy",
                "f1",
                "roc_auc",
                "fit_time_sec",
            },
            "outputs/model_feature_set_decisions.csv": {
                "dataset",
                "model",
                "selected_feature_set",
                "train_f1",
                "validation_f1",
                "f1_gap",
                "abs_f1_gap",
                "tie_break_reason",
            },
            "outputs/validation_curve_results.csv": {
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
            "outputs/gridsearch_results_top.csv": {
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
            "outputs/baseline_vs_tuned_test_results.csv": {
                "dataset",
                "feature_set",
                "model",
                "variant",
                "accuracy",
                "f1",
                "roc_auc",
                "fit_time_sec",
            },
        },
        "json_outputs": [],
    },
    "04": {
        "path": ROOT_DIR / "04-calibration-threshold-and-decision-policy",
        "todo_notebooks": [
            "notebooks/01_calibration_basics_todo.ipynb",
            "notebooks/02_threshold_policy_todo.ipynb",
        ],
        "csv_outputs": {
            "outputs/calibration_audit.csv": {
                "dataset",
                "model",
                "variant",
                "split",
                "brier",
                "log_loss",
                "roc_auc",
                "pr_auc",
                "ece",
            },
            "outputs/threshold_policy_grid.csv": {
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
            },
            "outputs/policy_test_report.csv": {
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
            },
            "outputs/segment_policy_audit.csv": {
                "dataset",
                "segment_feature",
                "segment",
                "n",
                "fp_rate",
                "fn_rate",
                "expected_cost_per_100",
            },
        },
        "json_outputs": [],
    },
    "05": {
        "path": ROOT_DIR / "05-drift-monitoring-and-retraining-policy",
        "todo_notebooks": [
            "notebooks/01_drift_detection_and_monitoring_todo.ipynb",
            "notebooks/02_retraining_policy_todo.ipynb",
        ],
        "csv_outputs": {
            "outputs/drift_detection_audit.csv": {
                "dataset",
                "window_id",
                "scenario",
                "feature",
                "feature_type",
                "detector",
                "statistic",
                "p_value",
                "effect_size",
                "drift_flag",
            },
            "outputs/monitoring_quality_audit.csv": {
                "dataset",
                "window_id",
                "scenario",
                "model_variant",
                "accuracy",
                "f1",
                "roc_auc",
                "pr_auc",
                "brier",
                "ece",
                "expected_cost",
                "delta_f1_vs_reference",
                "delta_cost_vs_reference",
            },
            "outputs/retraining_policy_decisions.csv": {
                "dataset",
                "window_id",
                "scenario",
                "drift_feature_share",
                "delta_f1_vs_reference",
                "delta_cost_vs_reference",
                "policy_action",
                "trigger_reason",
            },
            "outputs/post_retrain_comparison.csv": {
                "dataset",
                "scenario",
                "phase",
                "accuracy",
                "f1",
                "roc_auc",
                "pr_auc",
                "brier",
                "ece",
                "expected_cost",
            },
        },
        "json_outputs": [],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка обязательных артефактов сдачи по ЛР 01-05.")
    parser.add_argument(
        "--lab",
        required=True,
        choices=["01", "02", "03", "04", "05", "all"],
        help="Номер лабораторной (01..05) или all.",
    )
    return parser.parse_args()


def iter_labs(target: str) -> Iterable[str]:
    if target == "all":
        return ["01", "02", "03", "04", "05"]
    return [target]


def check_notebook_todo_markers(notebook_path: Path) -> list[str]:
    errors: list[str] = []
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "NotImplementedError" in source:
            errors.append(f"{notebook_path}: code-cell #{index} содержит NotImplementedError.")
        if "TODO(обязательно)" in source:
            errors.append(f"{notebook_path}: code-cell #{index} содержит TODO(обязательно).")
    return errors


def check_csv_contract(path: Path, expected_columns: set[str]) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        errors.append(f"{path}: отсутствует обязательный CSV-артефакт.")
        return errors

    frame = pd.read_csv(path)
    actual_columns = set(frame.columns)
    if actual_columns != expected_columns:
        errors.append(
            f"{path}: неверные колонки. Ожидались {sorted(expected_columns)}, получены {list(frame.columns)}."
        )
        return errors

    if frame.empty:
        errors.append(f"{path}: CSV пустой, ожидаются данные.")
        return errors

    if "dataset" in frame.columns:
        observed_datasets = set(frame["dataset"].astype(str).unique().tolist())
        if observed_datasets != EXPECTED_DATASETS:
            errors.append(
                f"{path}: ожидались dataset={sorted(EXPECTED_DATASETS)}, получены {sorted(observed_datasets)}."
            )
    return errors


def check_json_contract(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        errors.append(f"{path}: отсутствует обязательный JSON-артефакт.")
        return errors

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"{path}: JSON невалидный ({exc}).")
        return errors

    if not isinstance(data, dict):
        errors.append(f"{path}: ожидается JSON-объект верхнего уровня.")
        return errors

    if path.name == "shortlist_filter.json":
        if set(data.keys()) != EXPECTED_DATASETS:
            errors.append(f"{path}: ожидаются ключи {sorted(EXPECTED_DATASETS)}.")
        for dataset_name, features in data.items():
            if not isinstance(features, list) or not features:
                errors.append(f"{path}: {dataset_name} должен содержать непустой список признаков.")

    if path.name == "feature_sets_wrapper_embedded.json":
        if set(data.keys()) != EXPECTED_DATASETS:
            errors.append(f"{path}: ожидаются ключи {sorted(EXPECTED_DATASETS)}.")
        for dataset_name, feature_sets in data.items():
            if not isinstance(feature_sets, dict) or not feature_sets:
                errors.append(f"{path}: {dataset_name} должен содержать словарь feature set.")
                continue
            for set_name, features in feature_sets.items():
                if not set_name:
                    errors.append(f"{path}: найден пустой ключ feature set.")
                if not isinstance(features, list) or not features:
                    errors.append(f"{path}: {dataset_name}/{set_name} должен содержать непустой список признаков.")
    return errors


def validate_lab(lab_id: str) -> tuple[list[str], list[str]]:
    config = LAB_CONFIGS[lab_id]
    base_path: Path = config["path"]
    errors: list[str] = []
    checks: list[str] = []

    if not base_path.exists():
        errors.append(f"[ЛР {lab_id}] Не найдена директория: {base_path}")
        return checks, errors

    checks.append(f"[ЛР {lab_id}] Проверка TODO/NotImplementedError в student notebooks.")
    for rel_nb in config["todo_notebooks"]:
        nb_path = base_path / rel_nb
        if not nb_path.exists():
            errors.append(f"[ЛР {lab_id}] Не найден notebook: {nb_path}")
            continue
        errors.extend(f"[ЛР {lab_id}] {msg}" for msg in check_notebook_todo_markers(nb_path))

    checks.append(f"[ЛР {lab_id}] Проверка обязательных CSV-артефактов.")
    for rel_path, expected_columns in config["csv_outputs"].items():
        path = base_path / rel_path
        errors.extend(f"[ЛР {lab_id}] {msg}" for msg in check_csv_contract(path, expected_columns))

    if config["json_outputs"]:
        checks.append(f"[ЛР {lab_id}] Проверка обязательных JSON-артефактов.")
    for rel_path in config["json_outputs"]:
        path = base_path / rel_path
        errors.extend(f"[ЛР {lab_id}] {msg}" for msg in check_json_contract(path))

    return checks, errors


def print_fix_hints(lab_id: str) -> None:
    base_path = LAB_CONFIGS[lab_id]["path"]
    print("  Как исправить:")
    print(f"  1) Откройте notebook из {base_path}/notebooks и завершите все TODO-блоки.")
    print("  2) Выполните notebook до export-ячейки и пересохраните outputs.")
    print(f"  3) Повторно запустите: .venv/bin/python scripts/check_submission.py --lab {lab_id}")


def main() -> None:
    args = parse_args()
    target_labs = list(iter_labs(args.lab))

    all_errors: list[str] = []
    for lab_id in target_labs:
        checks, errors = validate_lab(lab_id)
        print(f"\n=== ЛР {lab_id} ===")
        for msg in checks:
            print(f"- {msg}")
        if errors:
            print("Статус: FAIL")
            for err in errors:
                print(f"  - {err}")
            print_fix_hints(lab_id)
            all_errors.extend(errors)
        else:
            print("Статус: PASS")

    if all_errors:
        print(f"\nИтог: FAIL ({len(all_errors)} проблем)")
        sys.exit(1)

    print("\nИтог: PASS (артефакты сдачи выглядят корректно)")


if __name__ == "__main__":
    main()
