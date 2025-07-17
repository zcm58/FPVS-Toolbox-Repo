# advanced_analysis_qt_post.py
"""Post/utility mixin for Qt advanced analysis window."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict

from PySide6.QtWidgets import QFileDialog, QMessageBox

logger = logging.getLogger(__name__)


class AdvancedAnalysisPostMixin:
    def _extract_pid_for_group(self, group_data: Dict[str, Any]) -> str:
        file_paths = group_data.get('file_paths', [])
        if not file_paths:
            return "UnknownPID"
        base_name = Path(file_paths[0]).stem
        pid_regex_primary = r"\b(P\d+|S\d+|Sub\d+)\b"
        match = re.search(pid_regex_primary, base_name, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        if '_' in base_name:
            for part in base_name.split('_'):
                if re.fullmatch(r"(P\d+|S\d+|Sub\d+)", part, re.IGNORECASE):
                    return part.upper()
        cleaned_pid = re.sub(r"(_unamb|_ambig|_mid|_run\d*|_sess\d*|_task\w*|_eeg|_raw|_preproc|_ica).*", "", base_name, flags=re.IGNORECASE)
        cleaned_pid = re.sub(r"[^A-Za-z0-9_]", "", cleaned_pid)
        if '_' in cleaned_pid:
            for part in cleaned_pid.split('_'):
                if re.fullmatch(r"(P\d+|S\d+|Sub\d+)", part, re.IGNORECASE):
                    return part.upper()
        cleaned_pid_alphanum_only = re.sub(r"[^A-Za-z0-9]", "", cleaned_pid)
        return cleaned_pid_alphanum_only if cleaned_pid_alphanum_only else base_name

    def save_groups_to_file(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Group Configuration", filter="JSON files (*.json)")
        if not file_path:
            return
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.defined_groups, f, indent=2)
            self.log(f"Saved group configuration to {file_path}.")
        except Exception as e:  # pragma: no cover - display error
            QMessageBox.critical(self, "Error", f"Failed to save configuration:\n{e}")

    def load_groups_from_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Group Configuration", filter="JSON files (*.json)")
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list) or not all(isinstance(g, dict) for g in data):
                raise ValueError("Invalid configuration structure")
            required = {"name", "file_paths", "condition_mappings", "averaging_method", "config_saved"}
            for group in data:
                if not required.issubset(group.keys()):
                    raise ValueError("Configuration missing required keys")
            self.defined_groups = data
            all_paths = {fp for g in self.defined_groups for fp in g.get("file_paths", [])}
            self.source_eeg_files = sorted(all_paths)
            self.selected_group_index = None
            self._update_source_files_listbox()
            self._update_groups_listbox()
            self._clear_group_config_display()
            self._update_start_processing_button_state()
            self.log(f"Loaded {len(self.defined_groups)} group(s) from {file_path}.")
        except Exception as e:  # pragma: no cover - display error
            QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{e}")

    def _on_close(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Qt window close handling
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        if getattr(self, "_active_threads", None):
            if QMessageBox.question(
                self,
                "Confirm Close",
                "Processing is ongoing. Stop and close?",
            ) != QMessageBox.Yes:
                event.ignore()
                return
            self.stop_processing()
            for thread, worker in list(self._active_threads):
                thread.requestInterruption()
                thread.quit()
                thread.wait(10000)
                worker.deleteLater()
                thread.deleteLater()
                self._active_threads.remove((thread, worker))
        super().closeEvent(event)
