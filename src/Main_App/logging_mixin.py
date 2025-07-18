from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QTextCursor
import logging

from .settings_manager import SettingsManager


class _LogProxy(QObject):
    """Internal QObject to emit log signals."""

    log_signal = Signal(str)


class QtLoggingMixin:
    """Thread-safe logging mixin for Qt widgets."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._proxy = _LogProxy()
        self.log_signal = self._proxy.log_signal
        self.log_signal.connect(self._append_log)
        try:
            self.debug_mode = SettingsManager().debug_enabled()
        except Exception:
            self.debug_mode = False

    @Slot(str)
    def _append_log(self, msg: str) -> None:
        if hasattr(self, "log_output"):
            self.log_output.appendPlainText(msg)
            self.log_output.moveCursor(QTextCursor.End)

    def log(self, msg: str) -> None:
        self.log_signal.emit(msg)

    def debug(self, message: str) -> None:
        if getattr(self, "debug_mode", False):
            self.log_signal.emit(f"[DEBUG] {message}")
            logging.getLogger(type(self).__module__).debug(message)


LoggingMixin = QtLoggingMixin
