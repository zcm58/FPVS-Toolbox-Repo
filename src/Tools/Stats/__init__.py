"""Expose stats modules after the Legacy/PySide6 refactor."""

# Legacy Tkinter based implementation
from .Legacy.stats import StatsAnalysisWindow
from .Legacy.stats_export import export_significance_results_to_excel
from .Legacy import (
    stats_file_scanner,
    stats_ui,
    stats_runners,
    stats_helpers,
    stats_analysis,
    full_snr,
    interpretation_helpers,
    mixed_effects_model,
    repeated_m_anova,
    posthoc_tests,
)

# PySide6 specific helpers
from .PySide6 import stats_file_scanner_pyside6, stats_ui_pyside6

__all__ = [
    "StatsAnalysisWindow",
    "export_significance_results_to_excel",
    "stats_file_scanner",
    "stats_ui",
    "stats_runners",
    "stats_helpers",
    "stats_analysis",
    "full_snr",
    "interpretation_helpers",
    "mixed_effects_model",
    "repeated_m_anova",
    "posthoc_tests",
    "stats_file_scanner_pyside6",
    "stats_ui_pyside6",
]
