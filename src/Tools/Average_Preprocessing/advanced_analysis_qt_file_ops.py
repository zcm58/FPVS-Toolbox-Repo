# advanced_analysis_qt_file_ops.py
"""File operations mixin for the PySide6 advanced analysis window."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from PySide6.QtWidgets import QFileDialog

logger = logging.getLogger(__name__)


class AdvancedAnalysisFileOpsMixin:
    def add_source_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Select EEG Files", filter="EEG files (*.bdf)")
        if not files:
            return
        added = 0
        for fp in files:
            if fp not in self.source_eeg_files:
                self.source_eeg_files.append(fp)
                added += 1
        if added:
            self.source_eeg_files.sort()
            self._update_source_files_listbox()
            self.log_signal.emit(
                f"Added {added} source file(s). Total: {len(self.source_eeg_files)}."
            )

    def remove_source_files(self) -> None:
        indices = [self.source_files_list.row(it) for it in self.source_files_list.selectedItems()]
        if not indices:
            self.log_signal.emit("No source files selected to remove.")
            return
        removed_paths = [self.source_eeg_files[i] for i in indices]
        self.source_eeg_files = [f for i, f in enumerate(self.source_eeg_files) if i not in indices]
        self._update_source_files_listbox()
        self.log_signal.emit(f"Removed {len(removed_paths)} file(s) from source list.")
        self._check_groups_for_removed_files(removed_paths)

    def _update_source_files_listbox(self) -> None:
        self.source_files_list.clear()
        for fp in self.source_eeg_files:
            self.source_files_list.addItem(Path(fp).name)

    def _check_groups_for_removed_files(self, removed_paths: List[str]) -> None:
        affected = False
        for group in self.defined_groups:
            before = len(group['file_paths'])
            group['file_paths'] = [p for p in group['file_paths'] if p not in removed_paths]
            if len(group['file_paths']) != before:
                group['config_saved'] = False
                affected = True
        if affected:
            self._update_groups_listbox()
            self._update_start_processing_button_state()
