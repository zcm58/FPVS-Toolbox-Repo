# src/Main_App/__init__.py
"""
Top-level package for the FPVS Toolbox (PySide6-first).

Exports PySide6-safe utilities and re-exports select Legacy symbols via
lazy wrappers so imports like `from Main_App import post_process` keep working
without importing Legacy at module import time.

Also re-exports the PySide6 SettingsManager so existing code can
`from Main_App import SettingsManager`.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QStandardPaths

# -----------------------------
# PySide6-native utilities
# -----------------------------

class _SettingsProxy:
    """Minimal settings facade for boot-time needs."""

    def debug_enabled(self) -> bool:
        env = os.getenv("FPVS_DEBUG", "").strip().lower()
        if env in {"1", "true", "yes", "on"}:
            return True
        if env in {"0", "false", "no", "off"}:
            return False
        try:
            from PySide6.QtCore import QSettings
            qs = QSettings()
            val = qs.value("app/debug_enabled", False, type=bool)
            return bool(val)
        except Exception:
            return False


def get_settings() -> _SettingsProxy:
    return _SettingsProxy()


def _log_dir() -> Path:
    base = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation) or ".")
    p = base / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    if getattr(root, "_fpvs_configured", False):
        root.setLevel(level)
        return
    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    try:
        fh = logging.FileHandler(_log_dir() / "fpvs_toolbox.log", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        pass

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    root._fpvs_configured = True  # type: ignore[attr-defined]


def install_messagebox_logger(debug: bool) -> None:
    if not debug:
        return
    # No-op placeholder. Wire QMessageBox at the UI level if desired.


# Re-export Project API from PySide6 backend (safe, no Legacy import)
from .PySide6_App.Backend import Project  # noqa: E402

# Re-export the PySide6 SettingsManager so code can do `from Main_App import SettingsManager`
from .PySide6_App.Backend.settings_manager import SettingsManager  # noqa: E402

# -----------------------------
# Lazy Legacy wrappers
# -----------------------------

def _lazy_import(module: str, name: Optional[str] = None) -> Any:
    """
    Import `Main_App.Legacy_App.<module>` only when the symbol is first used.
    Example: _lazy_import("post_process", "post_process")
    """
    mod = __import__(f"{__name__}.Legacy_App.{module}", fromlist=[name] if name else [])
    return getattr(mod, name) if name else mod

# Provide only the symbols that current PySide6 code still imports from Main_App.
# Add more wrappers if you see ImportError for other legacy names.

def post_process(*args: Any, **kwargs: Any) -> Any:  # used by Average_Preprocessing
    return _lazy_import("post_process", "post_process")(*args, **kwargs)

def perform_preprocessing(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("eeg_preprocessing", "perform_preprocessing")(*args, **kwargs)

def preprocess_raw(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("app_logic", "preprocess_raw")(*args, **kwargs)

def load_eeg_file(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("load_utils", "load_eeg_file")(*args, **kwargs)

def ProcessingMixin(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("processing_utils", "ProcessingMixin")(*args, **kwargs)

# Optional UI wrappers if anything still imports these:
def SettingsWindow(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    return _lazy_import("settings_window", "SettingsWindow")(*args, **kwargs)

def AppMenuBar(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    return _lazy_import("menu_bar", "AppMenuBar")(*args, **kwargs)

def SetupPanelManager(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    return _lazy_import("ui_setup_panels", "SetupPanelManager")(*args, **kwargs)

def EventMapManager(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    return _lazy_import("ui_event_map_manager", "EventMapManager")(*args, **kwargs)

class EventMapMixin:  # pragma: no cover
    def __init_subclass__(cls, **kw: Any) -> None:
        raise RuntimeError("Import EventMapMixin from Legacy_App.event_map_utils directly.")

class FileSelectionMixin:  # pragma: no cover
    def __init_subclass__(cls, **kw: Any) -> None:
        raise RuntimeError("Import FileSelectionMixin from Legacy_App.file_selection directly.")

class EventDetectionMixin:  # pragma: no cover
    def __init_subclass__(cls, **kw: Any) -> None:
        raise RuntimeError("Import EventDetectionMixin from Legacy_App.event_detection directly.")

class ValidationMixin:  # pragma: no cover
    def __init_subclass__(cls, **kw: Any) -> None:
        raise RuntimeError("Import ValidationMixin from Legacy_App.validation_mixins directly.")

__all__ = [
    # PySide6-first exports
    "get_settings",
    "configure_logging",
    "install_messagebox_logger",
    "Project",
    "SettingsManager",
    # Legacy lazy wrappers that some modules still import
    "post_process",
    "perform_preprocessing",
    "preprocess_raw",
    "load_eeg_file",
    "ProcessingMixin",
    "SettingsWindow",
    "AppMenuBar",
    "SetupPanelManager",
    "EventMapManager",
    "EventMapMixin",
    "FileSelectionMixin",
    "EventDetectionMixin",
    "ValidationMixin",
]
