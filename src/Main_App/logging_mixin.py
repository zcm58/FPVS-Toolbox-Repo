"""Logging helpers for both Tkinter and Qt widgets."""
from __future__ import annotations

import logging
import tkinter as tk
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QPlainTextEdit
import pandas as pd

logger = logging.getLogger(__name__)


class LoggingMixin:
    """Adds a timestamped logging function used across the toolbox."""

    def log(self, message: str, level: int = logging.INFO) -> None:
        """Write ``message`` to the GUI log widget and :mod:`logging`."""
        if tk is None:  # During interpreter shutdown ``tk`` may be ``None``
            return
        ts = pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3]
        formatted = f"{ts} [GUI]: {message}\n"

        try:
            if hasattr(self, "log_text") and self.log_text and self.log_text.winfo_exists():
                if level != logging.DEBUG or logger.isEnabledFor(logging.DEBUG):
                    self.log_text.configure(state="normal")
                    self.log_text.insert(tk.END, formatted)
                    self.log_text.see(tk.END)
                    self.log_text.configure(state="disabled")
        except Exception as e:  # pragma: no cover - best effort logging
            logger.exception("Error writing log message to GUI: %s", e)

        if level == logging.DEBUG:
            logger.debug(message)
        elif level == logging.WARNING:
            logger.warning(message)
        elif level == logging.ERROR:
            logger.error(message)
        elif level == logging.CRITICAL:
            logger.critical(message)
        else:
            logger.info(message)

    def debug(self, message: str) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            self.log(f"[DEBUG] {message}", level=logging.DEBUG)


class QtLoggingMixin(QObject):
    """Provide a thread-safe logging mechanism for Qt widgets."""

    log_signal = Signal(str)

    def __init__(self) -> None:  # pragma: no cover - GUI helper
        super().__init__()
        self.log_output: QPlainTextEdit | None = None
        self.log_signal.connect(self._append_log)

    @Slot(str)
    def _append_log(self, msg: str) -> None:  # pragma: no cover - GUI helper
        """Append ``msg`` to ``self.log_output`` and scroll to bottom."""
        if self.log_output is None:
            return
        self.log_output.appendPlainText(msg)
        bar = self.log_output.verticalScrollBar()
        bar.setValue(bar.maximum())
