#!/usr/bin/env python3
"""Проверка единого стиля комментариев и docstring в ноутбуках ЛР 01-04."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

NOTEBOOKS = sorted(
    [
        path
        for path in Path('.').glob('*/*/*.ipynb')
        if path.parts and len(path.parts[0]) >= 2 and path.parts[0][:2].isdigit()
    ]
)

REQUIRED_CELL_MARKERS = (
    '# Что делаем:',
    '# Зачем:',
    '# Как читать результат:',
    '# Типичные ошибки:',
)

FORBIDDEN_ANGLO = (
    'walkthrough',
    'skeleton',
    'downstream',
    'strictly',
    'test-check',
    'trade-off',
)

ALLOWED_TECH_TERMS = (
    'feature_set',
    'feature_sets',
    'reliability diagram',
    'fallback policy',
    'policy_name',
    'gridsearchcv',
    'roc_auc',
    'pr_auc',
    'log_loss',
)

MIN_COMMENT_RATIO_TODO = 0.20
MIN_COMMENT_RATIO_SOLUTION_AND_THEORY = 0.18


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def cell_source(cell: dict) -> str:
    return ''.join(cell.get('source', []))


def is_nontrivial_code_cell(source: str) -> bool:
    lines = [line for line in source.splitlines() if line.strip()]
    if not lines:
        return False
    if len(lines) == 1 and lines[0].lstrip().startswith('%'):
        return False
    return True


def normalize_user_text(text: str) -> str:
    # Убираем markdown inline-code и снижаем шум от технических идентификаторов.
    text = re.sub(r'`[^`]*`', ' ', text)
    text = text.lower()
    return text


def assert_required_markers(path: Path, code_cells: list[dict]) -> None:
    for idx, cell in enumerate(code_cells, start=1):
        source = cell_source(cell)
        if not is_nontrivial_code_cell(source):
            continue
        for marker in REQUIRED_CELL_MARKERS:
            if marker not in source:
                raise AssertionError(
                    f'{path}: code-cell #{idx} не содержит обязательный маркер `{marker}`.'
                )


def assert_docstrings(path: Path, code_cells: list[dict]) -> None:
    for idx, cell in enumerate(code_cells, start=1):
        source = cell_source(cell)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node)
                if not doc:
                    raise AssertionError(
                        f'{path}: `{node.name}` в code-cell #{idx} должен содержать docstring.'
                    )
                for required in ('Args:', 'Returns:'):
                    if required not in doc:
                        raise AssertionError(
                            f'{path}: docstring `{node.name}` должен содержать секцию `{required}`.'
                        )


def expected_min_ratio(path: Path) -> float:
    path_str = str(path)
    if '/notebooks/' in path_str:
        return MIN_COMMENT_RATIO_TODO
    return MIN_COMMENT_RATIO_SOLUTION_AND_THEORY


def assert_comment_density(path: Path, code_cells: list[dict]) -> None:
    code_lines = 0
    comment_lines = 0

    for cell in code_cells:
        source = cell_source(cell)
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            code_lines += 1
            if stripped.startswith('#'):
                comment_lines += 1

    ratio = (comment_lines / code_lines) if code_lines else 1.0
    threshold = expected_min_ratio(path)
    if ratio < threshold:
        raise AssertionError(
            f'{path}: недостаточная плотность комментариев ({ratio:.3f} < {threshold:.2f}).'
        )


def assert_language_contract(path: Path, markdown_cells: list[dict], code_cells: list[dict]) -> None:
    markdown_text = '\n'.join(cell_source(cell) for cell in markdown_cells)
    comment_text = '\n'.join(
        line.strip()
        for cell in code_cells
        for line in cell_source(cell).splitlines()
        if line.strip().startswith('#')
    )
    combined = normalize_user_text(markdown_text + '\n' + comment_text)

    for term in FORBIDDEN_ANGLO:
        if term in combined:
            if term in ALLOWED_TECH_TERMS:
                continue
            raise AssertionError(
                f'{path}: найден запрещенный англицизм `{term}` в учебном тексте/комментариях.'
            )


def main() -> None:
    if not NOTEBOOKS:
        raise AssertionError('Не найдено ни одного ноутбука для проверки стиля комментариев.')

    for notebook_path in NOTEBOOKS:
        notebook = load_notebook(notebook_path)
        code_cells = [cell for cell in notebook.get('cells', []) if cell.get('cell_type') == 'code']
        markdown_cells = [cell for cell in notebook.get('cells', []) if cell.get('cell_type') == 'markdown']

        assert_required_markers(notebook_path, code_cells)
        assert_docstrings(notebook_path, code_cells)
        assert_comment_density(notebook_path, code_cells)
        assert_language_contract(notebook_path, markdown_cells, code_cells)

    print(f'Notebook comment-style check passed for {len(NOTEBOOKS)} notebooks.')


if __name__ == '__main__':
    try:
        main()
    except AssertionError as exc:
        print(f'[FAIL] {exc}')
        sys.exit(1)
