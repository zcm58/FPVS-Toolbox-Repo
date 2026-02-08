import importlib.util
import os

import pytest

try:
    missing_qt = importlib.util.find_spec("PySide6") is None
    missing_pytestqt = importlib.util.find_spec("pytestqt") is None
except ValueError:
    missing_qt = True
    missing_pytestqt = True

if missing_qt or missing_pytestqt:
    pytest.skip("PySide6 or pytest-qt not available", allow_module_level=True)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from Main_App.PySide6_App.GUI.main_window import MainWindow
from Main_App.PySide6_App.GUI.sidebar import SidebarButton


def _collect_sidebar_labels(window: MainWindow) -> list[str]:
    sidebar = window.sidebar
    labels: list[str] = []
    layout = sidebar.layout()
    if layout is None:
        return labels
    for i in range(layout.count()):
        item = layout.itemAt(i)
        widget = item.widget() if item is not None else None
        if isinstance(widget, SidebarButton):
            labels.append(widget.text_lbl.text())
    return labels


def _collect_tools_menu_labels(window: MainWindow) -> list[str]:
    menu_bar = window.menuBar()
    tools_menu = None
    for action in menu_bar.actions():
        if action.text() == "Tools":
            tools_menu = action.menu()
            break
    if tools_menu is None:
        return []
    return [action.text() for action in tools_menu.actions() if action.text()]


def test_ratio_calculator_removed_from_ui(qtbot) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    qtbot.addWidget(window)

    sidebar_labels = _collect_sidebar_labels(window)
    menu_labels = _collect_tools_menu_labels(window)

    assert "Ratio Calculator" not in sidebar_labels
    assert "Ratio Calculator" not in menu_labels
    assert app is not None
