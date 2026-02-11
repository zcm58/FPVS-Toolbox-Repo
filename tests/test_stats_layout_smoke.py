from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QGroupBox, QSplitter, QTabWidget  # noqa: E402

from Tools.Stats.PySide6.stats_ui_pyside6 import StatsWindow  # noqa: E402


@pytest.fixture
def app(qapp):
    """Ensure a QApplication exists for qtbot interactions."""
    return qapp


@pytest.mark.qt
def test_stats_window_layout_smoke(qtbot, tmp_path, app):
    window = StatsWindow(project_dir=str(tmp_path))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    splitter = window.main_horizontal_splitter
    assert isinstance(splitter, QSplitter)
    assert splitter.orientation() == Qt.Horizontal
    assert splitter.count() == 3

    group_boxes = {box.title(): box for box in window.findChildren(QGroupBox)}
    assert group_boxes["Included Conditions"].isVisible()
    assert group_boxes["Summed BCA definition"].isVisible()
    assert group_boxes["Multi-Group Scan Summary"].isVisible()
    assert group_boxes["Analysis Controls"].isVisible()

    assert splitter.widget(0).isAncestorOf(window.conditions_group)
    assert splitter.widget(1).isAncestorOf(window.dv_group)
    assert splitter.widget(2).isAncestorOf(window.output_tabs)

    output_tabs = window.findChild(QTabWidget)
    assert output_tabs is not None
    assert output_tabs.isVisible()
    assert output_tabs.height() > 80

    assert isinstance(window.right_vertical_splitter, QSplitter)
    assert window.right_vertical_splitter.orientation() == Qt.Vertical
