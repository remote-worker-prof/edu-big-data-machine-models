#!/usr/bin/env python3
"""Диагностика учебной среды и upstream-контрактов курса.

Назначение:
- быстро показать, почему проект/ноутбуки не стартуют;
- дать точные команды исправления в формате `.venv/bin/python ...`.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

REQUIRED_MODULES = [
    "pandas",
    "numpy",
    "sklearn",
    "scipy",
    "nbconvert",
    "nbformat",
    "ipykernel",
    "jupyter",
]

UPSTREAM_ARTIFACTS = [
    {
        "label": "ЛР02 <- ЛР01 feature sets",
        "path": ROOT_DIR / "01-feature-importance-and-selection/outputs/feature_sets_wrapper_embedded.json",
        "fix": (
            ".venv/bin/python 01-feature-importance-and-selection/scripts/verify_lab01.py\n"
            "или выполните ЛР01 до export-ячейки с `feature_sets_wrapper_embedded.json`."
        ),
    },
    {
        "label": "ЛР02 <- ЛР01 model results",
        "path": ROOT_DIR / "01-feature-importance-and-selection/outputs/model_results.csv",
        "fix": (
            ".venv/bin/python 01-feature-importance-and-selection/scripts/verify_lab01.py\n"
            "или выполните ЛР01 Notebook 3 до export-ячейки с `model_results.csv`."
        ),
    },
    {
        "label": "ЛР03 <- ЛР01 feature sets",
        "path": ROOT_DIR / "01-feature-importance-and-selection/outputs/feature_sets_wrapper_embedded.json",
        "fix": (
            ".venv/bin/python 01-feature-importance-and-selection/scripts/verify_lab01.py\n"
            "или выполните ЛР01 до export-ячейки с `feature_sets_wrapper_embedded.json`."
        ),
    },
    {
        "label": "ЛР04 <- ЛР03 baseline_vs_tuned",
        "path": ROOT_DIR
        / "03-overfitting-validation-and-hyperparameter-tuning/outputs/baseline_vs_tuned_test_results.csv",
        "fix": (
            ".venv/bin/python 03-overfitting-validation-and-hyperparameter-tuning/scripts/verify_lab03.py\n"
            "или выполните ЛР03 до export-ячейки с `baseline_vs_tuned_test_results.csv`."
        ),
    },
]


@dataclass
class CheckResult:
    status: str  # OK | WARN | FAIL
    name: str
    message: str
    fix: str | None = None


def check_repo_context() -> list[CheckResult]:
    results: list[CheckResult] = []
    cwd = Path.cwd().resolve()
    if cwd != ROOT_DIR:
        results.append(
            CheckResult(
                status="WARN",
                name="Рабочая директория",
                message=f"Скрипт запущен из {cwd}, ожидается корень проекта {ROOT_DIR}.",
                fix=f"cd {ROOT_DIR}",
            )
        )
    else:
        results.append(
            CheckResult(
                status="OK",
                name="Рабочая директория",
                message=f"Корень проекта: {ROOT_DIR}",
            )
        )
    return results


def check_python_context() -> list[CheckResult]:
    results: list[CheckResult] = []
    exe_raw = Path(sys.executable)
    exe_resolved = exe_raw.resolve()
    in_project_venv = exe_raw.parent == (ROOT_DIR / ".venv" / "bin")

    if in_project_venv:
        results.append(
            CheckResult(
                status="OK",
                name="Python интерпретатор",
                message=f"Используется локальный интерпретатор: {exe_raw}",
            )
        )
    else:
        results.append(
            CheckResult(
                status="WARN",
                name="Python интерпретатор",
                message=f"Используется внешний интерпретатор: {exe_resolved}",
                fix=f"{ROOT_DIR}/.venv/bin/python scripts/doctor_env.py",
            )
        )
    return results


def check_requirements_files() -> list[CheckResult]:
    results: list[CheckResult] = []
    reqs = sorted(ROOT_DIR.glob("*-*/requirements.txt"))
    if not reqs:
        results.append(
            CheckResult(
                status="FAIL",
                name="requirements.txt",
                message="Не найдены файлы зависимостей лабораторных.",
            )
        )
        return results

    results.append(
        CheckResult(
            status="OK",
            name="requirements.txt",
            message=f"Найдено файлов зависимостей: {len(reqs)}",
        )
    )
    return results


def check_required_modules() -> list[CheckResult]:
    results: list[CheckResult] = []
    missing = [module for module in REQUIRED_MODULES if importlib.util.find_spec(module) is None]
    if missing:
        install_cmds = "\n".join(
            [
                ".venv/bin/python -m pip install --upgrade pip",
                ".venv/bin/python -m pip install -r 01-feature-importance-and-selection/requirements.txt",
                ".venv/bin/python -m pip install -r 02-model-interpretability-and-explainability/requirements.txt",
                ".venv/bin/python -m pip install -r 03-overfitting-validation-and-hyperparameter-tuning/requirements.txt",
                ".venv/bin/python -m pip install -r 04-calibration-threshold-and-decision-policy/requirements.txt",
                ".venv/bin/python -m pip install -r 05-drift-monitoring-and-retraining-policy/requirements.txt",
            ]
        )
        results.append(
            CheckResult(
                status="FAIL",
                name="Python зависимости",
                message=f"Не найдены модули: {', '.join(missing)}.",
                fix=install_cmds,
            )
        )
    else:
        results.append(
            CheckResult(
                status="OK",
                name="Python зависимости",
                message="Все базовые зависимости курса доступны.",
            )
        )
    return results


def check_jupyter_entrypoint() -> list[CheckResult]:
    results: list[CheckResult] = []
    proc = subprocess.run(
        [sys.executable, "-m", "jupyter", "--version"],
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "Не удалось запустить `python -m jupyter --version`."
        results.append(
            CheckResult(
                status="FAIL",
                name="Jupyter",
                message=msg,
                fix=".venv/bin/python -m pip install -r 01-feature-importance-and-selection/requirements.txt",
            )
        )
    else:
        results.append(
            CheckResult(
                status="OK",
                name="Jupyter",
                message="`python -m jupyter` доступен.",
            )
        )
    return results


def check_upstream_artifacts() -> list[CheckResult]:
    results: list[CheckResult] = []
    for item in UPSTREAM_ARTIFACTS:
        path = item["path"]
        if path.exists():
            results.append(
                CheckResult(
                    status="OK",
                    name=item["label"],
                    message=f"Найден: {path.relative_to(ROOT_DIR)}",
                )
            )
        else:
            results.append(
                CheckResult(
                    status="WARN",
                    name=item["label"],
                    message=(
                        f"Не найден upstream-артефакт: {path.relative_to(ROOT_DIR)}. "
                        "Это нормально до завершения предыдущих ЛР, но следующие ЛР не стартуют."
                    ),
                    fix=item["fix"],
                )
            )
    return results


def print_results(results: list[CheckResult]) -> None:
    for item in results:
        print(f"[{item.status}] {item.name}: {item.message}")
        if item.fix:
            print("  Как исправить:")
            for line in item.fix.splitlines():
                print(f"  {line}")


def main() -> None:
    all_results: list[CheckResult] = []
    all_results.extend(check_repo_context())
    all_results.extend(check_python_context())
    all_results.extend(check_requirements_files())
    all_results.extend(check_required_modules())
    all_results.extend(check_jupyter_entrypoint())
    all_results.extend(check_upstream_artifacts())

    print(f"Project root: {ROOT_DIR}")
    print(f"Python: {sys.executable}")
    print_results(all_results)

    fail_count = sum(item.status == "FAIL" for item in all_results)
    warn_count = sum(item.status == "WARN" for item in all_results)
    print(f"\nSummary: FAIL={fail_count}, WARN={warn_count}, TOTAL={len(all_results)}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
