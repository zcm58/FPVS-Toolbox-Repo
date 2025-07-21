"""Core data structures for advanced averaging logic."""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional

from Main_App.settings_manager import SettingsManager


class AdvancedAnalysisBase:
    """Data container and basic helpers shared by logic mixins."""

    def __init__(self, log_callback: Optional[Callable[[str], None]] = None) -> None:
        self.log_callback = log_callback or (lambda msg: None)
        self.debug_mode = SettingsManager().debug_enabled()
        self.source_eeg_files: List[str] = []
        self.defined_groups: List[Dict[str, Any]] = []
        self.selected_group_index: Optional[int] = None
        self.processing_thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    def log(self, message: str) -> None:
        """Send ``message`` to the provided ``log_callback``."""
        self.log_callback(message)

    def debug(self, message: str) -> None:
        if self.debug_mode:
            self.log(f"[DEBUG] {message}")
