
"""Thread-safe logging mixin for Qt widgets."""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QTextCursor


class LoggingMixin(QObject):
    log_signal = Signal(str)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.log_signal.connect(self._append_log)

    @Slot(str)
    def _append_log(self, msg: str) -> None:
        self.log_output.appendPlainText(msg)
        self.log_output.moveCursor(QTextCursor.End)

    def log(self, msg: str) -> None:
        self.log_signal.emit(msg)


QtLoggingMixin = LoggingMixin
LoggingMixin = LoggingMixin
