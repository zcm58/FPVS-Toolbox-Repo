
"""Thread-safe logging mixin for Qt widgets."""
from __future__ import annotations

import logging
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QPlainTextEdit
import pandas as pd


logger = logging.getLogger(__name__)


class QtLoggingMixin(QObject):
    """Provide a thread-safe logging mechanism for Qt widgets."""

    log_signal = Signal(str)

    def __init__(self) -> None:  # pragma: no cover - GUI helper
        super().__init__()
        self.log_output: QPlainTextEdit | None = None
        self.log_signal.connect(self._append_log)

    def log(self, message: str, level: int = logging.INFO) -> None:
        """Emit ``message`` to the GUI log widget and :mod:`logging`."""
        ts = pd.Timestamp.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"{ts} [GUI]: {message}"
        logger.log(level, formatted)
        self.log_signal.emit(formatted)

    def debug(self, message: str) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            self.log(f"[DEBUG] {message}", level=logging.DEBUG)

    @Slot(str)
    def _append_log(self, msg: str) -> None:  # pragma: no cover - GUI helper
        """Append ``msg`` to ``self.log_output`` and scroll to bottom."""
        if self.log_output is None:
            return
        self.log_output.appendPlainText(msg)
        bar = self.log_output.verticalScrollBar()
        bar.setValue(bar.maximum())


# Backwards compatibility with previous Tk-based code
LoggingMixin = QtLoggingMixin
