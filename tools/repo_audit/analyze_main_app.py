#!/usr/bin/env python3
"""Static audit script for ``Main_App`` modules.

This script analyses Python files under ``src/Main_App`` and produces
CSV, Markdown and optionally XLSX reports describing import structure,
code metrics and reachability from an entry module (by default
``Main_App.PySide6_App.GUI.main_window``).

The script performs static analysis only and does not import any project
modules.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set
import importlib.util
import sys
import sysconfig

logger = logging.getLogger(__name__)


@dataclass
class ModuleInfo:
    """Container for information extracted from a module."""

    module: str
    path: Path
    lines: int
    all_lines: int
    docstring: str
    docstring_first: str
    classes: List[str]
    functions: List[str]
    imports: Set[str]
    imports_count: int
    qt_usage: str
    risk_notes: List[str]


def discover_py_files(root: Path) -> List[Path]:
    """Return all Python files under ``root`` respecting exclusions."""

    def _exclude(p: Path) -> bool:
        excluded = {"__pycache__", "venv", ".venv", "env"}
        return any(part.startswith(".") or part in excluded for part in p.parts)

    files: List[Path] = []
    for file in root.rglob("*.py"):
        if _exclude(file.parent):
            continue
        files.append(file)
    return files


def module_name_from_path(src_root: Path, file: Path) -> str:
    """Create a dotted module name from ``file`` relative to ``src_root``."""

    rel = file.relative_to(src_root).with_suffix("")
    return ".".join(rel.parts)


def _resolve_relative(module: str, level: int, current: str) -> str:
    parts = current.split(".")[:-1]
    if level:
        parts = parts[:-level]
    if module:
        parts.append(module)
    return ".".join(parts)


def parse_file(file: Path, module: str) -> ModuleInfo:
    """Parse ``file`` and return :class:`ModuleInfo`.

    Syntax errors are logged and result in empty AST-derived fields.
    """

    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - unlikely
        logger.error("Failed to read %s: %s", file, exc)
        text = ""
    lines = text.splitlines()
    all_lines = len(lines)
    non_empty = sum(1 for line in lines if line.strip())

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        logger.error("Syntax error in %s: %s", file, exc)
        return ModuleInfo(
            module=module,
            path=file,
            lines=non_empty,
            all_lines=all_lines,
            docstring="",
            docstring_first="",
            classes=[],
            functions=[],
            imports=set(),
            imports_count=0,
            qt_usage="None",
            risk_notes=[],
        )

    doc = ast.get_docstring(tree) or ""
    doc_clean = " ".join(doc.split())
    doc_first = ""
    if doc_clean:
        for sep in [".", "!", "?"]:
            if sep in doc_clean:
                doc_first = doc_clean.split(sep)[0]
                break
        if not doc_first:
            doc_first = doc_clean
    doc_first = doc_first[:160]
    doc = doc_clean[:160]

    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    functions = [
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    imports: Set[str] = set()
    imports_count = 0
    qt_usage = "None"
    risk_notes: List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports_count += 1
            for alias in node.names:
                imports.add(alias.name)
                if alias.name.startswith("PySide6"):
                    qt_usage = "PySide6"
                elif "QtWidgets" in alias.name and qt_usage == "None":
                    qt_usage = "QtWidgets"
        elif isinstance(node, ast.ImportFrom):
            imports_count += 1
            mod = node.module or ""
            abs_mod = _resolve_relative(mod, node.level, module)
            imports.add(abs_mod)
            if abs_mod.startswith("PySide6"):
                qt_usage = "PySide6"
            elif "QtWidgets" in abs_mod and qt_usage == "None":
                qt_usage = "QtWidgets"
            if abs_mod == "PySide6.QtWidgets":
                for alias in node.names:
                    if alias.name == "QAction":
                        risk_notes.append(
                            "QAction imported from QtWidgets; use PySide6.QtGui"
                        )

    return ModuleInfo(
        module=module,
        path=file,
        lines=non_empty,
        all_lines=all_lines,
        docstring=doc,
        docstring_first=doc_first,
        classes=classes,
        functions=functions,
        imports=imports,
        imports_count=imports_count,
        qt_usage=qt_usage,
        risk_notes=risk_notes,
    )


def is_stdlib(module: str) -> bool:
    base = module.split(".")[0]
    if base in sys.builtin_module_names:
        return True
    spec = importlib.util.find_spec(base)
    if spec and spec.origin:
        stdlib = Path(sysconfig.get_path("stdlib"))
        try:
            return Path(spec.origin).is_relative_to(stdlib)
        except AttributeError:  # pragma: no cover - Py<3.9
            return str(spec.origin).startswith(str(stdlib))
    return False


def top_non_std(imports: Iterable[str]) -> List[str]:
    pkgs = {imp.split(".")[0] for imp in imports}
    pkgs = [p for p in pkgs if not is_stdlib(p) and not p.startswith("Main_App")]
    pkgs.sort()
    return pkgs[:3]


def build_graph(
    infos: Dict[str, ModuleInfo]
) -> tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Build import graph and reverse mapping."""

    graph: Dict[str, Set[str]] = {m: set() for m in infos}
    imported_by: Dict[str, Set[str]] = {m: set() for m in infos}
    imports_main_app: Dict[str, Set[str]] = {m: set() for m in infos}

    for mod, info in infos.items():
        for imp in info.imports:
            if imp in infos:
                graph[mod].add(imp)
                imported_by[imp].add(mod)
                imports_main_app[mod].add(imp)
    return graph, imported_by, imports_main_app


def reachable(entry: str, graph: Dict[str, Set[str]]) -> Set[str]:
    seen = set()
    if entry not in graph:
        return seen
    stack = [entry]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph.get(node, []))
    return seen


def guess_role(info: ModuleInfo, imports: Set[str], reachable_flag: bool) -> str:
    if any(i.startswith("mne") for i in imports):
        return "EEG I/O/processing"
    if any(i.startswith("PySide6") for i in imports):
        return "GUI widgets/logic"
    name = info.path.name.lower()
    if "worker" in name or "thread" in name:
        return "Background worker"
    if "Legacy_App" in info.path.parts and reachable_flag:
        return "Legacy runtime dependency"
    return "Utility/adapter"


def summarise(info: ModuleInfo, imports: Set[str], reachable_flag: bool) -> str:
    if info.docstring_first:
        return info.docstring_first
    cls_part = f"{len(info.classes)} classes ({', '.join(info.classes[:3])})"
    fn_part = f"{len(info.functions)} functions ({', '.join(info.functions[:3])})"
    pkgs = top_non_std(imports)
    pkg_part = ", ".join(pkgs) if pkgs else "none"
    role = guess_role(info, imports, reachable_flag)
    summary = (
        f"Defines {cls_part}, {fn_part}, imports ({pkg_part}), primary role: {role}"
    )
    return summary[:160]


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_md(
    path: Path,
    rows: List[Dict[str, object]],
    columns: List[str],
    repo_root: Path,
    timestamp: str,
    entry: str,
) -> None:
    def trunc(s: str, limit: int = 40) -> str:
        s = str(s)
        return s if len(s) <= limit else s[: limit - 3] + "..."

    total_lines = sum(r["all_lines"] for r in rows)
    reachable_cnt = sum(1 for r in rows if r["reachable_from_main_window"])
    unreachable_cnt = len(rows) - reachable_cnt

    top10 = sorted(rows, key=lambda r: r["lines"], reverse=True)[:10]
    qaction_files = [r for r in rows if "QAction" in r["risk_notes"]]
    legacy_reachable = [
        r
        for r in rows
        if r["reachable_from_main_window"]
        and any(m.startswith("Main_App.Legacy_App") for m in r["imports_main_app_modules"].split(";"))
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write("# Main App Audit\n\n")
        f.write(f"- Repo root: {repo_root}\n")
        f.write(f"- Timestamp: {timestamp}\n\n")
        f.write(
            f"Analyzed {len(rows)} Python files with {total_lines} total lines.\n\n"
        )
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for r in rows:
            f.write(
                "| "
                + " | ".join(trunc(r[c]) for c in columns)
                + " |\n"
            )
        f.write("\n## Findings\n\n")
        f.write(
            f"- Reachable files from {entry}: {reachable_cnt}; unreachable: {unreachable_cnt}.\n"
        )
        f.write("- Top 10 largest files (lines, reachable):\n")
        for r in top10:
            f.write(
                f"  - {r['relative_path']} ({r['lines']} lines, reachable={r['reachable_from_main_window']})\n"
            )
        if qaction_files:
            f.write("- Files importing QAction from QtWidgets:\n")
            for r in qaction_files:
                f.write(f"  - {r['relative_path']}\n")
        else:
            f.write("- No files import QAction from QtWidgets.\n")
        f.write(
            f"- {len(legacy_reachable)} reachable files import from Legacy_App.\n"
        )


def write_xlsx_if_available(
    path: Path, rows: List[Dict[str, object]], columns: List[str]
) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.info("openpyxl not available: %s", exc)
        return

    wb = Workbook()
    ws = wb.active
    ws.append(columns)
    for r in rows:
        ws.append([r[c] for c in columns])
    for col in ws.columns:
        max_len = max(len(str(cell.value)) for cell in col) + 2
        ws.column_dimensions[col[0].column_letter].width = max_len
        for cell in col:
            cell.alignment = Alignment(horizontal="center")
    ws.freeze_panes = "A2"
    wb.save(path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", default="src", type=Path)
    parser.add_argument(
        "--entry",
        default="Main_App.PySide6_App.GUI.main_window",
        help="Entry module for reachability",
    )
    parser.add_argument("--out-dir", default="reports", type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    src_root: Path = args.src_root
    main_app_root = src_root / "Main_App"
    files = discover_py_files(main_app_root)
    logger.info("Discovered %d Python files", len(files))

    module_infos: Dict[str, ModuleInfo] = {}
    for file in files:
        module = module_name_from_path(src_root, file)
        info = parse_file(file, module)
        if module in module_infos:
            prev = module_infos[module]
            if len(file.parts) < len(prev.path.parts):
                logger.warning("Duplicate module %s; preferring %s over %s", module, file, prev.path)
                module_infos[module] = info
            else:
                logger.warning("Duplicate module %s; keeping %s", module, prev.path)
        else:
            module_infos[module] = info

    graph, imported_by, imports_main_app = build_graph(module_infos)
    reach = reachable(args.entry, graph)

    rows: List[Dict[str, object]] = []
    repo_root = Path.cwd()
    for mod, info in module_infos.items():
        directly = mod in graph.get(args.entry, set())
        reachable_flag = mod in reach
        imports_main = imports_main_app.get(mod, set())
        summary = summarise(info, info.imports, reachable_flag)
        rows.append(
            {
                "relative_path": str(info.path.relative_to(repo_root)).replace("/", "\\"),
                "module": mod,
                "lines": info.lines,
                "all_lines": info.all_lines,
                "docstring": info.docstring,
                "summary": summary,
                "qt_usage": info.qt_usage,
                "imports_count": info.imports_count,
                "directly_imported_by_main_window": directly,
                "reachable_from_main_window": reachable_flag,
                "imported_by": ";".join(sorted(imported_by.get(mod, set()))),
                "imports_main_app_modules": ";".join(sorted(imports_main)),
                "risk_notes": ";".join(info.risk_notes),
                "used_in_runtime_only": "Unknown",
            }
        )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"main_app_audit_{timestamp}"
    columns = [
        "relative_path",
        "module",
        "lines",
        "all_lines",
        "docstring",
        "summary",
        "qt_usage",
        "imports_count",
        "directly_imported_by_main_window",
        "reachable_from_main_window",
        "imported_by",
        "imports_main_app_modules",
        "risk_notes",
        "used_in_runtime_only",
    ]

    write_csv(base.with_suffix(".csv"), rows, columns)
    write_md(base.with_suffix(".md"), rows, columns, repo_root, timestamp, args.entry)
    write_xlsx_if_available(base.with_suffix(".xlsx"), rows, columns)
    logger.info("Wrote reports to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
