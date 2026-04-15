#!/usr/bin/env python3
"""Единый запускатор QA-проверок по всем лабораторным.

Запускает:
- smoke-check ЛР 01-05;
- проверку стиля комментариев в ноутбуках;
- unit-тесты `lab_utils` для ЛР 03-05.

Возвращает код 0 при успехе и код 1 при любой ошибке.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


CHECKS = [
    {
        "name": "Lab 01 smoke-check",
        "cmd": [sys.executable, "01-feature-importance-and-selection/scripts/verify_lab01.py"],
        "hint": "Проверьте структуру notebook и артефакты ЛР 01, затем перезапустите verify_lab01.py.",
    },
    {
        "name": "Lab 02 smoke-check",
        "cmd": [sys.executable, "02-model-interpretability-and-explainability/scripts/verify_lab02.py"],
        "hint": "Проверьте upstream-артефакты из ЛР 01 и контракты CSV ЛР 02.",
    },
    {
        "name": "Lab 03 smoke-check",
        "cmd": [sys.executable, "03-overfitting-validation-and-hyperparameter-tuning/scripts/verify_lab03.py"],
        "hint": "Проверьте workflow ЛР 03 и корректность outputs/*.csv.",
    },
    {
        "name": "Lab 04 smoke-check",
        "cmd": [sys.executable, "04-calibration-threshold-and-decision-policy/scripts/verify_lab04.py"],
        "hint": "Проверьте контракты калибровки/порогов и теоретический notebook ЛР 04.",
    },
    {
        "name": "Lab 05 smoke-check",
        "cmd": [sys.executable, "05-drift-monitoring-and-retraining-policy/scripts/verify_lab05.py"],
        "hint": "Проверьте policy-контракты и артефакты мониторинга ЛР 05.",
    },
    {
        "name": "Notebook comment-style",
        "cmd": [sys.executable, "scripts/verify_notebook_comment_style.py"],
        "hint": "Синхронизируйте маркеры комментариев и docstring в notebook-коде.",
    },
    {
        "name": "Unit tests (lab_utils)",
        "cmd": [
            sys.executable,
            "-m",
            "unittest",
            "03-overfitting-validation-and-hyperparameter-tuning/tests/test_lab_utils.py",
            "04-calibration-threshold-and-decision-policy/tests/test_lab_utils.py",
            "05-drift-monitoring-and-retraining-policy/tests/test_lab_utils.py",
        ],
        "hint": "Исправьте ошибки в helper-функциях `lab_utils.py` и повторите unit-тесты.",
    },
]


def format_env_hint(output: str) -> str | None:
    lowered = output.lower()
    if "modulenotfounderror" in lowered or "no module named" in lowered:
        return (
            "Похоже, не установлены зависимости. "
            "Запустите `.venv/bin/python -m pip install -r <lab>/requirements.txt` "
            "для всех лабораторных или используйте готовый bootstrap из README."
        )
    return None


def run_check(name: str, cmd: list[str], hint: str) -> tuple[bool, str]:
    result = subprocess.run(
        cmd,
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
    )

    output_parts = []
    if result.stdout:
        output_parts.append(result.stdout.strip())
    if result.stderr:
        output_parts.append(result.stderr.strip())
    output_text = "\n".join(part for part in output_parts if part).strip()

    if result.returncode == 0:
        success_message = output_text or f"{name}: OK"
        return True, success_message

    lines = [output_text] if output_text else [f"{name}: failed with code {result.returncode}"]
    lines.append(f"Рекомендация: {hint}")
    env_hint = format_env_hint(output_text)
    if env_hint:
        lines.append(f"Подсказка по среде: {env_hint}")
    return False, "\n".join(lines)


def main() -> None:
    failures: list[tuple[str, str]] = []

    print(f"Python interpreter: {sys.executable}")
    print(f"Project root: {ROOT_DIR}")

    for index, check in enumerate(CHECKS, start=1):
        name = check["name"]
        print(f"\n[{index}/{len(CHECKS)}] {name}")
        ok, message = run_check(name=name, cmd=check["cmd"], hint=check["hint"])
        print(message)
        if not ok:
            failures.append((name, message))

    if failures:
        print("\nQA summary: FAIL")
        for name, _ in failures:
            print(f"- {name}")
        sys.exit(1)

    print("\nQA summary: PASS")


if __name__ == "__main__":
    main()
