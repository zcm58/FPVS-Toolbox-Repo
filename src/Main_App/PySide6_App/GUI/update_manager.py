"""Check for application updates via GitHub releases (PySide6-safe, non-blocking)."""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import requests
from packaging import version
from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool, QTimer, QUrl
from PySide6.QtWidgets import QWidget, QMessageBox
from PySide6.QtGui import QDesktopServices

from config import FPVS_TOOLBOX_VERSION, FPVS_TOOLBOX_UPDATE_API
from Main_App.PySide6_App.utils.settings import get_app_settings, mark_update_check

_LOG = logging.getLogger(__name__)
_CHECK_INTERVAL = timedelta(hours=24)


# ---------- public helpers ----------

def cleanup_old_executable() -> None:
    """Remove leftover backup EXE after an Inno update."""
    backup = sys.executable + ".old"
    try:
        if os.path.exists(backup):
            os.remove(backup)
    except Exception:
        pass


def check_for_updates_async(
    app: QWidget,
    silent: bool = True,
    notify_if_no_update: bool = True,
) -> None:
    """Menu action: check in background. Popups only if silent=False."""
    if _skip_due_to_debounce():
        _log(app, "Skipping update check; last run was within 24 hours.")
        return
    job = _CheckJob()
    job.sigs.available.connect(lambda info: _on_available(app, info, silent))
    job.sigs.none.connect(lambda current: _on_none(app, current, silent, notify_if_no_update))
    job.sigs.error.connect(lambda msg: _on_error(app, msg, silent))
    QThreadPool.globalInstance().start(job)


def check_for_updates_on_launch(app: QWidget) -> None:
    """Startup check: never show a dialog if up-to-date or on error."""
    if _skip_due_to_debounce():
        _log(app, "Skipping update check on launch (debounced).")
        return
    job = _CheckJob()
    job.sigs.available.connect(lambda info: _on_available(app, info, False))  # prompt only if update exists
    job.sigs.none.connect(lambda current: _log(app, "No update available."))
    job.sigs.error.connect(lambda msg: _log(app, f"Update check failed: {msg}"))
    QThreadPool.globalInstance().start(job)


# ---------- worker + signals ----------

@dataclass(frozen=True)
class _UpdateInfo:
    latest: str
    url: str


class _UpdateSignals(QObject):
    available = Signal(object)  # _UpdateInfo
    none = Signal(str)          # current version (no update)
    error = Signal(str)         # error message


class _CheckJob(QRunnable):
    """QRunnable that hits the GitHub releases API and compares versions."""
    def __init__(self) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.sigs = _UpdateSignals()

    def run(self) -> None:
        try:
            start = time.perf_counter()
            _LOG.info("Checking for updates...")
            resp = requests.get(FPVS_TOOLBOX_UPDATE_API, timeout=2)
            resp.raise_for_status()
            data = resp.json()
            latest = str(data.get("tag_name") or "").strip()
            url = str(data.get("html_url") or "").strip()
            if not latest:
                raise ValueError("Missing tag_name in release response.")
            mark_update_check(datetime.now(timezone.utc))
            if _is_newer(latest, f"v{FPVS_TOOLBOX_VERSION}"):
                self.sigs.available.emit(_UpdateInfo(latest=latest, url=url))
            else:
                self.sigs.none.emit(FPVS_TOOLBOX_VERSION)
            elapsed = (time.perf_counter() - start) * 1000.0
            _LOG.debug("Update check completed", extra={"op": "update_check", "elapsed_ms": int(elapsed)})
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000.0 if 'start' in locals() else 0.0
            _LOG.warning(
                "Update check failed",
                extra={
                    "op": "update_check",
                    "path": FPVS_TOOLBOX_UPDATE_API,
                    "elapsed_ms": int(elapsed),
                    "exc": repr(e),
                },
            )
            self.sigs.error.emit(str(e))


# ---------- UI-thread handlers ----------

def _log(app: object, msg: str) -> None:
    if hasattr(app, "log"):
        try:
            app.log(msg)  # type: ignore[attr-defined]
            return
        except Exception:
            pass
    _LOG.info(msg)


def _on_available(parent: QWidget, info: _UpdateInfo, silent: bool) -> None:
    _log(parent, f"Update {info.latest} available at {info.url}")
    if silent:
        return
    QTimer.singleShot(0, lambda: _prompt_open_release(parent, info))


def _on_none(parent: QWidget, current: str, silent: bool, notify_if_no_update: bool) -> None:
    _log(parent, "No update available.")
    if not silent and notify_if_no_update:
        QTimer.singleShot(0, lambda: QMessageBox.information(
            parent, "Up to Date", f"You are running the latest version (v{current})."
        ))


def _on_error(parent: QWidget, msg: str, _silent: bool) -> None:
    _log(parent, f"Update check failed: {msg}")


def _prompt_open_release(parent: QWidget, info: _UpdateInfo) -> None:
    m = QMessageBox.question(
        parent,
        "Update Available",
        f"A newer version of the FPVS Toolbox ({info.latest}) is available.\n\nOpen the release page?",
        QMessageBox.Yes | QMessageBox.No,
    )
    if m == QMessageBox.Yes and info.url:
        QDesktopServices.openUrl(QUrl(info.url))

# ---------- util ----------

def _is_newer(latest: str, current: str) -> bool:
    """Return True if latest > current, tolerant of a leading 'v'."""
    return version.parse(latest.lstrip("v")) > version.parse(current.lstrip("v"))


def _skip_due_to_debounce() -> bool:
    settings = get_app_settings()
    raw_value = settings.value("updates/last_checked_utc", "", type=str)
    if not raw_value:
        return False
    sanitized = raw_value.replace("Z", "+00:00") if raw_value.endswith("Z") else raw_value
    try:
        last = datetime.fromisoformat(sanitized)
    except ValueError:
        return False
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return now - last < _CHECK_INTERVAL
