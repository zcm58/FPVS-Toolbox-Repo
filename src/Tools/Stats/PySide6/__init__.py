# src/Tools/Stats/PySide6/__init__.py
"""
PySide6 wrapper package for the FPVS Statistical Analysis Tool.

Public API:
- StatsWindow: Main PySide6 UI for statistical analysis.
- StatsWorker: QRunnable worker with progress/message/error/finished signals.
- scan_folder_simple, ScanError: Excel-folder scanner utilities.

This package does not modify legacy analysis code. It only adapts UI/IO.
"""

from .stats_ui_pyside6 import StatsWindow
from .stats_worker import StatsWorker
from .stats_file_scanner_pyside6 import scan_folder_simple, ScanError

__all__ = [
    "StatsWindow",
    "StatsWorker",
    "scan_folder_simple",
    "ScanError",
]
