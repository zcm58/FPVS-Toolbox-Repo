from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

import Tools.Stats.PySide6.stats_ui_pyside6 as stats_ui

pytestmark = pytest.mark.qt


def _build_excel_payload(excel_dir: Path, subjects: list[str], condition: str = "CondA"):
    cond_dir = excel_dir / condition
    cond_dir.mkdir(parents=True, exist_ok=True)
    subject_data: dict[str, dict[str, str]] = {}
    for pid in subjects:
        file_path = cond_dir / f"{pid}_{condition}.xlsx"
        file_path.write_text("dummy")
        subject_data.setdefault(pid, {})[condition] = str(file_path)
    return subjects, [condition], subject_data


def _make_stats_window(qtbot, monkeypatch, project_dir: Path) -> stats_ui.StatsWindow:
    monkeypatch.setattr(stats_ui.QTimer, "singleShot", lambda *args, **kwargs: None)
    win = stats_ui.StatsWindow(project_dir=str(project_dir))
    qtbot.addWidget(win)
    return win


def _patch_worker_stubs(monkeypatch):
    def fake_anova(progress_cb, message_cb, **kwargs):
        return {"anova_df_results": pd.DataFrame({"Effect": ["Group"], "p-value": [0.5]})}

    def fake_lmm(progress_cb, message_cb, **kwargs):
        return {
            "mixed_results_df": pd.DataFrame({"Term": ["group"], "p_value": [0.25]}),
            "output_text": "mixed",
        }

    def fake_contrasts(progress_cb, message_cb, **kwargs):
        return {
            "results_df": pd.DataFrame(
                {
                    "group_1": ["A"],
                    "group_2": ["B"],
                    "condition": ["CondA"],
                    "roi": ["ROI"],
                    "difference": [0.0],
                    "p_value": [0.5],
                }
            ),
            "output_text": "contrasts",
        }

    monkeypatch.setattr(stats_ui, "_between_group_anova_calc", fake_anova)
    monkeypatch.setattr(stats_ui, "_lmm_calc", fake_lmm)
    monkeypatch.setattr(stats_ui, "_group_contrasts_calc", fake_contrasts)


def _patch_instant_worker(monkeypatch):
    def immediate_precheck(self, require_anova: bool = False):
        return True

    def immediate_wire(self, worker, finished_slot):
        worker.signals.progress.connect(self._on_worker_progress)
        worker.signals.message.connect(self._on_worker_message)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.finished.connect(finished_slot)
        worker.run()

    monkeypatch.setattr(stats_ui.StatsWindow, "_precheck", immediate_precheck, raising=False)
    monkeypatch.setattr(stats_ui.StatsWindow, "_wire_and_start", immediate_wire, raising=False)


def test_stats_single_group_between_guard(qtbot, tmp_path, monkeypatch):
    project_dir = tmp_path / "proj"
    excel_dir = project_dir / "1 - Excel Data Files"
    excel_dir.mkdir(parents=True)
    manifest = {"name": "Legacy"}
    (project_dir / "project.json").write_text(json.dumps(manifest))
    subjects, conditions, data = _build_excel_payload(excel_dir, ["P01"])

    monkeypatch.setattr(stats_ui, "scan_folder_simple", lambda folder: (subjects, conditions, data))
    warning_calls: list[str] = []
    info_calls: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: warning_calls.append(args[2] if len(args) > 2 else ""),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *args, **kwargs: info_calls.append(args[2] if len(args) > 2 else ""),
    )

    win = _make_stats_window(qtbot, monkeypatch, project_dir)
    win.le_folder.setText(str(excel_dir))
    win._scan_button_clicked()
    assert not warning_calls
    assert not win._known_group_labels()

    qtbot.mouseClick(win.run_between_anova_btn, Qt.LeftButton)
    assert info_calls


def test_stats_between_group_runs_with_stubbed_worker(qtbot, tmp_path, monkeypatch):
    project_dir = tmp_path / "proj"
    excel_dir = project_dir / "1 - Excel Data Files"
    excel_dir.mkdir(parents=True)
    manifest = {
        "name": "Multi",
        "groups": {"GroupA": {}, "GroupB": {}},
        "participants": {"P01": {"group": "GroupA"}, "P02": {"group": "GroupB"}},
    }
    (project_dir / "project.json").write_text(json.dumps(manifest))
    subjects, conditions, data = _build_excel_payload(excel_dir, ["P01", "P02"])

    monkeypatch.setattr(stats_ui, "scan_folder_simple", lambda folder: (subjects, conditions, data))
    _patch_instant_worker(monkeypatch)
    _patch_worker_stubs(monkeypatch)
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)

    win = _make_stats_window(qtbot, monkeypatch, project_dir)
    win.rois = {"ROI": ["Cz"]}
    win.le_folder.setText(str(excel_dir))
    win._scan_button_clicked()

    assert set(win._known_group_labels()) == {"GroupA", "GroupB"}

    qtbot.mouseClick(win.run_between_anova_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: isinstance(win.between_anova_results_data, pd.DataFrame), timeout=1000)

    qtbot.mouseClick(win.run_between_mixed_btn, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: isinstance(win.between_mixed_model_results_data, pd.DataFrame),
        timeout=1000,
    )

    qtbot.mouseClick(win.run_group_contrasts_btn, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: isinstance(win.group_contrasts_results_data, pd.DataFrame),
        timeout=1000,
    )


def test_stats_unknown_subject_warning_once(qtbot, tmp_path, monkeypatch):
    project_dir = tmp_path / "proj"
    excel_dir = project_dir / "1 - Excel Data Files"
    excel_dir.mkdir(parents=True)
    manifest = {
        "name": "Multi",
        "groups": {"GroupA": {}, "GroupB": {}},
        "participants": {"P01": {"group": "GroupA"}},
    }
    (project_dir / "project.json").write_text(json.dumps(manifest))
    subjects, conditions, data = _build_excel_payload(excel_dir, ["P01", "P02"])

    monkeypatch.setattr(stats_ui, "scan_folder_simple", lambda folder: (subjects, conditions, data))
    _patch_instant_worker(monkeypatch)
    _patch_worker_stubs(monkeypatch)
    warning_calls: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: warning_calls.append(args[2] if len(args) > 2 else ""),
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)

    win = _make_stats_window(qtbot, monkeypatch, project_dir)
    win.rois = {"ROI": ["Cz"]}
    win.le_folder.setText(str(excel_dir))
    win._scan_button_clicked()

    assert warning_calls
    assert any("Unrecognized Excel Files" in msg for msg in warning_calls)
    assert win.subject_groups.get("P02") is None

    qtbot.mouseClick(win.run_between_anova_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: isinstance(win.between_anova_results_data, pd.DataFrame), timeout=1000)
