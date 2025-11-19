from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

if importlib.util.find_spec("matplotlib") is None:
    pytest.skip("matplotlib not available", allow_module_level=True)

from Tools.Plot_Generator import plot_generator as plot_module  # noqa: E402

pytestmark = pytest.mark.qt


def _make_project(tmp_path: Path, manifest: dict | None = None):
    project_dir = tmp_path / "proj"
    excel_dir = project_dir / "1 - Excel Data Files"
    cond_dir = excel_dir / "CondA"
    cond_dir.mkdir(parents=True, exist_ok=True)
    out_dir = project_dir / "2 - SNR Plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (project_dir / "project.json").write_text(json.dumps(manifest))
    return project_dir, excel_dir, cond_dir, out_dir


def _create_window(qtbot):
    win = plot_module.PlotGeneratorWindow()
    qtbot.addWidget(win)
    return win


def test_plot_generator_single_group_controls_hidden(qtbot, tmp_path):
    manifest = {"name": "Legacy"}
    project_dir, excel_dir, cond_dir, out_dir = _make_project(tmp_path, manifest)
    (cond_dir / "P01_CondA.xlsx").write_text("dummy")

    win = _create_window(qtbot)
    win.folder_edit.setText(str(excel_dir))
    win.out_edit.setText(str(out_dir))
    win._populate_conditions(str(excel_dir))

    assert not win.group_box.isVisible()
    assert not win.group_overlay_check.isEnabled()


def test_plot_generator_multi_group_params_include_selection(qtbot, tmp_path, monkeypatch):
    manifest = {
        "name": "Multi",
        "groups": {"GroupA": {}, "GroupB": {}},
        "participants": {"P01": {"group": "GroupA"}, "P02": {"group": "GroupB"}},
    }
    project_dir, excel_dir, cond_dir, out_dir = _make_project(tmp_path, manifest)
    (cond_dir / "P01_CondA.xlsx").write_text("dummy")
    (cond_dir / "P02_CondA.xlsx").write_text("dummy")

    win = _create_window(qtbot)
    win.folder_edit.setText(str(excel_dir))
    win.out_edit.setText(str(out_dir))
    win._populate_conditions(str(excel_dir))

    assert win.group_box.isVisible()
    win.group_overlay_check.setChecked(True)
    item = win.group_list.item(0)
    item.setCheckState(Qt.Unchecked)
    selected_groups = win._selected_groups()
    assert selected_groups  # at least one remains checked

    captured: dict[str, tuple] = {}

    def fake_finish(self):
        return None

    def fake_start_next_condition(self):
        captured["params"] = self._gen_params
        self._conditions_queue = []
        self._finish_all()

    win._finish_all = fake_finish.__get__(win, plot_module.PlotGeneratorWindow)
    win._start_next_condition = fake_start_next_condition.__get__(
        win, plot_module.PlotGeneratorWindow
    )
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: None)
    win.condition_combo.setCurrentIndex(1)
    win.gen_btn.setEnabled(True)
    win._generate()

    assert "params" in captured
    _, _, _, _, _, _, group_kwargs = captured["params"]
    assert group_kwargs["enable_group_overlay"] is True
    assert group_kwargs["selected_groups"] == selected_groups


def test_plot_generator_group_overlay_warns_for_unassigned(qtbot, tmp_path):
    manifest = {
        "name": "Multi",
        "groups": {"GroupA": {}, "GroupB": {}},
        "participants": {"P01": {"group": "GroupA"}},
    }
    project_dir, excel_dir, cond_dir, out_dir = _make_project(tmp_path, manifest)
    (cond_dir / "P01_CondA.xlsx").write_text("dummy")
    (cond_dir / "P02_CondA.xlsx").write_text("dummy")

    win = _create_window(qtbot)
    win.folder_edit.setText(str(excel_dir))
    win.out_edit.setText(str(out_dir))
    win._populate_conditions(str(excel_dir))
    assert win.group_box.isVisible()
    win.group_overlay_check.setChecked(True)

    worker = plot_module._Worker(
        folder=str(excel_dir),
        condition="CondA",
        roi_map={"ROI": ["Cz"]},
        selected_roi="ROI",
        title="t",
        xlabel="x",
        ylabel="y",
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        out_dir=str(out_dir),
        subject_groups=win._subject_groups_map,
        selected_groups=win._available_groups,
        enable_group_overlay=True,
        multi_group_mode=True,
    )
    messages: list[str] = []
    worker._emit = lambda msg, *_: messages.append(msg) if msg else None
    worker._unknown_subject_files = {"P02_CondA.xlsx"}
    subject_data = {
        "P01": {"ROI": [1.0, 2.0]},
        "P02": {"ROI": [3.0, 4.0]},
    }
    curves = worker._build_group_curves(subject_data)

    assert any("Warning" in msg for msg in messages)
    assert "GroupA" in curves
    assert worker._unknown_subject_files == set()
