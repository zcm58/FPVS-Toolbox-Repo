from __future__ import annotations

from datetime import datetime, timezone
from unittest import mock

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QWidget

from Main_App.PySide6_App.GUI import update_manager
from Main_App.PySide6_App.utils.paths import bundle_path
from Main_App.PySide6_App.utils.settings import get_app_settings


def test_settings_provider_returns_ini(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    settings = get_app_settings()
    assert isinstance(settings, QSettings)
    assert settings.format() == QSettings.IniFormat


def test_update_check_debounced(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)

    settings = get_app_settings()
    original = settings.value("updates/last_checked_utc", "", type=str)
    now_iso = datetime.now(timezone.utc).isoformat()
    settings.setValue("updates/last_checked_utc", now_iso)
    settings.sync()

    with mock.patch.object(update_manager.requests, "get", side_effect=AssertionError("network call attempted")):
        update_manager.check_for_updates_on_launch(widget)
        qtbot.wait(10)

    settings.setValue("updates/last_checked_utc", original)
    settings.sync()


def test_bundle_path_handles_qss(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    qss_path = bundle_path("..", "..", "..", "qdark_sidebar.qss")
    if not qss_path.exists():
        pytest.skip("qdark_sidebar.qss not present in test environment")
    with open(qss_path, "r", encoding="utf-8") as handle:
        widget.setStyleSheet(handle.read())
