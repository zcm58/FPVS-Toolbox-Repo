from __future__ import annotations

import customtkinter as ctk

# log() and debug() schedule updates using the widget's ``after`` method
# when available so they can be safely invoked from background threads.

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
        """Append a message to the log widget in a thread-safe manner."""
        # Use Tk's event queue when available so calls from background threads
        # don't manipulate the widget directly.  When ``after`` is not present
        # (e.g. in unit tests), fall back to a direct call.
        if hasattr(self, "after"):
            self.after(0, self._append_log, message)
        else:
            self._append_log(message)

    def debug(self, message: str) -> None:
        """Append a debug message when debug mode is enabled."""
        if getattr(self, "debug_mode", False):
            if hasattr(self, "after"):
                self.after(0, self._append_log, f"[DEBUG] {message}")
            else:
                self._append_log(f"[DEBUG] {message}")
