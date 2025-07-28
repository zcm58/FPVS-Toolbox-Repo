# src/Tools/Stats/__init__.py

try:
    # The legacy CustomTkinter window requires additional dependencies
    # that may not always be available. Import it lazily so that failure
    # to load the old UI does not break PySide6 components.
    from .stats import StatsAnalysisWindow
except Exception:  # pragma: no cover - optional legacy UI
    StatsAnalysisWindow = None
from .stats_export import export_significance_results_to_excel
from .stats_ui import StatsToolWidget
from .stats_ui_ctk import launch_ctk_stats_tool
from . import stats_file_scanner, stats_ui, stats_runners, stats_helpers

__all__ = [
    "export_significance_results_to_excel",
    "stats_file_scanner",
    "stats_ui",
    "stats_runners",
    "stats_helpers",
    "StatsToolWidget",
    "launch_ctk_stats_tool",
]

if StatsAnalysisWindow is not None:  # pragma: no cover - optional legacy UI
    __all__.insert(0, "StatsAnalysisWindow")
