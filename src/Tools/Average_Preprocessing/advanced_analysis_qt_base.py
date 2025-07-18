# advanced_analysis_qt_base.py
"""PySide6 GUI for advanced averaging of preprocessed EEG epochs."""

from __future__ import annotations

import logging
import threading
import os
import glob

from typing import List, Dict, Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QListWidget, QPushButton, QRadioButton, QPlainTextEdit,
    QVBoxLayout, QProgressBar,
)

from Main_App.logging_mixin import QtLoggingMixin

from Main_App.settings_manager import SettingsManager

try:
    from config import (
        PAD_X, PAD_Y, BUTTON_WIDTH,
    )
except Exception:  # pragma: no cover - defaults
    PAD_X = PAD_Y = 5
    BUTTON_WIDTH = 180

logger = logging.getLogger(__name__)


class AdvancedAnalysisWindowBase(QtLoggingMixin, QDialog):
    """Base dialog containing common UI for advanced averaging."""

    def __init__(self, master) -> None:
        super().__init__(parent=None)

        self.master_app = master
        self.debug_mode = SettingsManager().debug_enabled()
        self.setWindowTitle("Advanced Averaging Analysis")
        self.resize(1050, 850)

        self.source_eeg_files: List[str] = []
        self.defined_groups: List[Dict[str, Any]] = []
        self.selected_group_index: Optional[int] = None

        self.processing_thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()
        self._active_threads: List[tuple] = []

        # Do not forward log messages to the main application; keep them local
        # to this window's log output.

        self._build_ui()
        self._populate_default_eeg_files()
        self._center()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QGridLayout(self)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 3)

        left = QFrame()
        left_layout = QVBoxLayout(left)
        layout.addWidget(left, 0, 0)

        left_layout.addWidget(QLabel("Source EEG Files"))
        self.source_files_list = QListWidget()
        self.source_files_list.setSelectionMode(QListWidget.ExtendedSelection)
        left_layout.addWidget(self.source_files_list)

        h = QHBoxLayout()
        btn = QPushButton("Add Files…")
        btn.clicked.connect(self.add_source_files)
        h.addWidget(btn)
        btn = QPushButton("Remove Selected")
        btn.clicked.connect(self.remove_source_files)
        h.addWidget(btn)
        left_layout.addLayout(h)

        left_layout.addWidget(QLabel("Defined Averaging Groups"))
        self.groups_list = QListWidget()
        self.groups_list.itemSelectionChanged.connect(self.on_group_select)
        left_layout.addWidget(self.groups_list)

        h2 = QHBoxLayout()
        btn = QPushButton("Create New Group")
        btn.clicked.connect(self.create_new_group)
        h2.addWidget(btn)
        btn = QPushButton("Rename Group")
        btn.clicked.connect(self.rename_selected_group)
        h2.addWidget(btn)
        btn = QPushButton("Delete Group")
        btn.clicked.connect(self.delete_selected_group)
        h2.addWidget(btn)
        left_layout.addLayout(h2)

        right = QFrame()
        right_layout = QVBoxLayout(right)
        layout.addWidget(right, 0, 1)

        right_layout.addWidget(QLabel("Group Configuration"))
        self.group_config_frame = QFrame()
        self.group_config_layout = QVBoxLayout(self.group_config_frame)
        right_layout.addWidget(self.group_config_frame)

        right_layout.addWidget(QLabel("Condition Mapping for Selected Group"))
        self.condition_mapping_frame = QFrame()
        self.condition_mapping_layout = QVBoxLayout(self.condition_mapping_frame)
        right_layout.addWidget(self.condition_mapping_frame)

        avg_box = QHBoxLayout()
        avg_box.addWidget(QLabel("Averaging Method:"))
        self.pool_rb = QRadioButton("Pool Trials")
        self.pool_rb.setToolTip(
            "All epochs from all files are pooled and averaged simultaneously."\
            " Gives equal weight to every epoch and is typically preferred."
        )
        self.pool_rb.toggled.connect(self._update_current_group_avg_method)
        avg_box.addWidget(self.pool_rb)
        self.avg_rb = QRadioButton("Average of Averages")
        self.avg_rb.setToolTip(
            "Each file is averaged separately before averaging those results. "\
            "Gives equal weight to files rather than epochs."
        )
        self.avg_rb.toggled.connect(self._update_current_group_avg_method)
        avg_box.addWidget(self.avg_rb)
        self.averaging_method_var = self.pool_rb
        right_layout.addLayout(avg_box)

        self.save_group_config_btn = QPushButton("Save Group Configuration")
        self.save_group_config_btn.clicked.connect(self.save_current_group_config)
        self.save_group_config_btn.setEnabled(False)
        right_layout.addWidget(self.save_group_config_btn, alignment=Qt.AlignRight)

        bottom = QFrame()
        bottom_layout = QVBoxLayout(bottom)
        layout.addWidget(bottom, 1, 0, 1, 2)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        bottom_layout.addWidget(self.log_output)

        hb = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        hb.addWidget(self.progress_bar)
        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_log)
        hb.addWidget(self.clear_log_btn)
        bottom_layout.addLayout(hb)

        cb = QHBoxLayout()
        self.start_btn = QPushButton("Start Advanced Processing")
        self.start_btn.clicked.connect(self.start_advanced_processing)
        cb.addWidget(self.start_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        cb.addWidget(self.stop_btn)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self._on_close)
        cb.addWidget(self.close_btn)
        bottom_layout.addLayout(cb)

        self._clear_group_config_display()

    def _populate_default_eeg_files(self) -> None:
        """Load .bdf files from the default input folder if available."""
        directory = ""
        if hasattr(self.master_app, "settings"):
            directory = self.master_app.settings.get("paths", "data_folder", "")
        if (not directory or not os.path.isdir(directory)) and hasattr(self.master_app, "data_paths") and self.master_app.data_paths:
            first = self.master_app.data_paths[0]
            directory = first if os.path.isdir(first) else os.path.dirname(first)
        if directory and os.path.isdir(directory):
            bdf_files = sorted(glob.glob(os.path.join(directory, "*.bdf")))
            added = 0
            for fp in bdf_files:
                if fp not in self.source_eeg_files:
                    self.source_eeg_files.append(fp)
                    added += 1
            if added:
                self.source_eeg_files.sort()
                self._update_source_files_listbox()
                self.log_signal.emit(f"Auto-added {added} BDF file(s) from {directory}.")

    def _center(self) -> None:
        geo = self.frameGeometry()
        center = self.parent().geometry().center() if self.parent() else self.screen().availableGeometry().center()
        geo.moveCenter(center)
        self.move(geo.topLeft())

    def clear_log(self) -> None:
        self.log_output.clear()

    def debug(self, message: str) -> None:
        if self.debug_mode:
            self.log_signal.emit(f"[DEBUG] {message}")


    def _clear_group_config_display(self) -> None:
        for layout in (self.group_config_layout, self.condition_mapping_layout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        self.save_group_config_btn.setEnabled(False)
        self.pool_rb.setChecked(True)

    # Placeholder methods for mixins --------------------
    def add_source_files(self) -> None: ...
    def remove_source_files(self) -> None: ...
    def create_new_group(self) -> None: ...
    def rename_selected_group(self) -> None: ...
    def delete_selected_group(self) -> None: ...
    def on_group_select(self) -> None: ...
    def save_current_group_config(self) -> bool: ...
    def start_advanced_processing(self) -> None: ...
    def stop_processing(self) -> None: ...
    def _update_current_group_avg_method(self) -> None: ...
    def _on_close(self) -> None: ...

    def closeEvent(self, event) -> None:
        # Allow mixins to override with custom close logic
        super().closeEvent(event)

