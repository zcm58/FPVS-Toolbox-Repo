"""Centralized settings provider for the FPVS Toolbox."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

from PySide6.QtCore import QCoreApplication, QSettings, QStandardPaths

__all__ = ["get_app_settings", "mark_update_check"]

_SETTINGS: QSettings | None = None
_MIGRATED: bool = False


def _settings_path() -> Path:
    vendor = "FPVS"
    app = "Toolbox"
    base = QStandardPaths.writableLocation(QStandardPaths.GenericDataLocation)
    if not base:
        base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    base_path = Path(base) if base else Path.home()
    target_dir = base_path / vendor / app
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / "settings.ini"


def _migrate_native_settings(target: QSettings) -> None:
    global _MIGRATED
    if _MIGRATED:
        return
    org = QCoreApplication.organizationName() or "FPVS"
    app = QCoreApplication.applicationName() or "Toolbox"
    legacy = QSettings(QSettings.NativeFormat, QSettings.UserScope, org, app)
    keys_to_copy: Iterable[str] = (
        "paths/projectsRoot",
        "loreta/mri_path",
        "recentProjects",
        "updates/last_checked_utc",
    )
    for key in keys_to_copy:
        if target.contains(key):
            continue
        if legacy.contains(key):
            target.setValue(key, legacy.value(key))
    target.sync()
    _MIGRATED = True


def get_app_settings() -> QSettings:
    """Return the singleton :class:`QSettings` configured for portable INI storage."""

    global _SETTINGS
    if _SETTINGS is None:
        settings_path = _settings_path()
        _SETTINGS = QSettings(str(settings_path), QSettings.IniFormat)
        _migrate_native_settings(_SETTINGS)
    return _SETTINGS


def mark_update_check(timestamp: datetime) -> None:
    """Persist the most recent update check timestamp."""

    settings = get_app_settings()
    settings.setValue("updates/last_checked_utc", timestamp.isoformat())
    settings.sync()

