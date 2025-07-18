from __future__ import annotations

import customtkinter as ctk

from .settings_manager import SettingsManager


class TkLoggingMixin:
    """Logging helpers for CustomTkinter widgets."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            self.debug_mode = SettingsManager().debug_enabled()
        except Exception:
            self.debug_mode = False

    def _append_log(self, message: str) -> None:
        if hasattr(self, "log_text") and self.log_text is not None:
            try:
                self.log_text.configure(state="normal")
                self.log_text.insert(ctk.END, message + "\n")
                self.log_text.see(ctk.END)
            finally:
                self.log_text.configure(state="disabled")

    def log(self, message: str) -> None:
        self._append_log(message)

    def debug(self, message: str) -> None:
        if getattr(self, "debug_mode", False):
            self._append_log(f"[DEBUG] {message}")
