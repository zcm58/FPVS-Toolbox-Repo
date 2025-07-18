# advanced_analysis_qt_group_ops.py
"""Group management mixin for Qt advanced analysis window."""

from __future__ import annotations

import logging
from pathlib import Path
import traceback

from PySide6.QtWidgets import (
    QInputDialog, QMessageBox, QLabel
)

logger = logging.getLogger(__name__)


class AdvancedAnalysisGroupOpsMixin:
    def create_new_group(self) -> None:
        """Create a new averaging group based on user input."""
        self.debug("create_new_group called")
        try:
            name, ok = QInputDialog.getText(self, "New Averaging Group", "Group name:")
            self.debug(f"Name dialog returned: {name!r}, accepted={ok}")
            if not ok or not name.strip():
                self.log_signal.emit("Group creation cancelled.")
                self.debug("Creation cancelled due to empty name or dialog rejection")
                return

            if any(g['name'] == name for g in self.defined_groups):
                self.debug(f"Duplicate group name attempted: {name}")
                QMessageBox.critical(self, "Error", f"A group named '{name}' already exists.")
                return

            id_str, ok = QInputDialog.getText(self, "Event IDs", "IDs to average (comma separated):")
            self.debug(f"ID dialog returned: {id_str!r}, accepted={ok}")
            if not ok or not id_str:
                self.log_signal.emit("Group creation cancelled (no IDs).")
                self.debug("Creation cancelled due to no IDs provided")
                return

            try:
                ids_to_average = [int(x.strip()) for x in id_str.split(',') if x.strip()]
                self.debug(f"Parsed IDs to average: {ids_to_average}")
            except ValueError as e:
                self.debug(f"Failed to parse IDs: {id_str!r} ({e})")
                QMessageBox.critical(self, "Error", "Enter only integers separated by commas.")
                return

            if not ids_to_average:
                self.debug("No valid IDs after parsing")
                QMessageBox.critical(self, "Error", "Enter at least one event ID.")
                return

            if not self.source_eeg_files:
                self.debug("Attempted to create group with no EEG files selected")
                QMessageBox.critical(self, "Error", "No EEG files selected. Add files first.")
                return

            mapping_rule = {'output_label': name, 'sources': []}
            self.debug(
                f"Building mapping rule for {len(self.source_eeg_files)} files and IDs {ids_to_average}"
            )
            for fp in self.source_eeg_files:
                for event_id_val in ids_to_average:
                    mapping_rule['sources'].append({
                        'file_path': fp,
                        'original_label': str(event_id_val),
                        'original_id': event_id_val,
                    })
            self.debug(f"Final mapping rule: {mapping_rule}")

            new_group = {
                'name': name,
                'file_paths': list(self.source_eeg_files),
                'condition_mappings': [mapping_rule],
                'averaging_method': 'Pool Trials',
                'config_saved': True,
                'ui_mapping_rules': []
            }
            self.defined_groups.append(new_group)
            self.debug(f"Group appended. Total groups: {len(self.defined_groups)}")
            self._update_groups_listbox()
            self.log_signal.emit(
                f"Created group '{name}' averaging IDs {ids_to_average} in all {len(self.source_eeg_files)} files."
            )
            idx = len(self.defined_groups) - 1
            self.groups_list.setCurrentRow(idx)
            self.on_group_select()
            self._update_start_processing_button_state()
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.exception("Unhandled error creating new group")
            group_name = locals().get("name", "<unknown>")
            self.log_signal.emit(
                f"!!! Error creating group '{group_name}': {exc}\n{traceback.format_exc()}"
            )

    def delete_selected_group(self) -> None:
        self.debug("delete_selected_group called")
        row = self.groups_list.currentRow()
        if row < 0:
            self.log_signal.emit("No group selected to delete.")
            self.debug("Delete requested but no selection in list")
            return
        name = self.defined_groups[row]['name']
        if QMessageBox.question(self, "Confirm Delete", f"Delete group '{name}'?") != QMessageBox.Yes:
            self.debug("User cancelled group delete")
            return
        del self.defined_groups[row]
        self.debug(f"Deleted group index {row} ({name})")
        self._update_groups_listbox()
        self.log_signal.emit(f"Deleted group: {name}")
        self.selected_group_index = None
        self._clear_group_config_display()
        self._update_start_processing_button_state()

    def rename_selected_group(self) -> None:
        self.debug("rename_selected_group called")
        row = self.groups_list.currentRow()
        if row < 0:
            self.log_signal.emit("No group selected to rename.")
            self.debug("Rename requested but no selection")
            return
        current = self.defined_groups[row]['name']
        new_name, ok = QInputDialog.getText(self, "Rename Group", f"Enter a new name for '{current}':")
        self.debug(f"Rename dialog returned: {new_name!r}, accepted={ok}")
        if not ok or not new_name.strip():
            self.debug("Rename cancelled or empty name")
            return
        if any(i != row and g['name'] == new_name for i, g in enumerate(self.defined_groups)):
            self.debug("Attempted to rename to existing group name")
            QMessageBox.critical(self, "Error", f"A group named '{new_name}' already exists.")
            return
        self.defined_groups[row]['name'] = new_name
        self.defined_groups[row]['config_saved'] = False
        self.debug(f"Group {row} renamed to {new_name}")
        self._update_groups_listbox()
        self.log_signal.emit(f"Renamed group to '{new_name}'.")

    def _update_groups_listbox(self) -> None:
        self.debug("_update_groups_listbox called")
        current = self.groups_list.currentRow()
        self.groups_list.clear()
        for g in self.defined_groups:
            status = "" if g.get('config_saved') else " (Unsaved)"
            self.groups_list.addItem(f"{g['name']} ({len(g['file_paths'])} files){status}")
        if current >= 0 and current < self.groups_list.count():
            self.groups_list.setCurrentRow(current)

    def on_group_select(self) -> None:
        self.debug("on_group_select called")
        row = self.groups_list.currentRow()
        if row < 0:
            self.debug("on_group_select with no selection")
            return
        self.selected_group_index = row
        self.debug(f"Selected group index set to {row}")
        self._display_group_configuration()
        self.save_group_config_btn.setEnabled(not self.defined_groups[row].get('config_saved', False))

    def _display_group_configuration(self) -> None:
        self.debug("_display_group_configuration called")
        self._clear_group_config_display()
        idx = self.selected_group_index
        if idx is None or idx >= len(self.defined_groups):
            self.debug("No valid group selected for display")
            return
        g = self.defined_groups[idx]
        self.debug(f"Displaying configuration for group {idx} ({g['name']})")
        self.group_config_layout.addWidget(QLabel(f"Group Name: {g['name']}"))
        files_display = "\n".join(Path(fp).name for fp in g['file_paths']) or "No files in this group."
        self.group_config_layout.addWidget(QLabel("Files in this group:\n" + files_display))
        if g.get('condition_mappings'):
            mapping = g['condition_mappings'][0]
            ids = sorted({src['original_id'] for src in mapping['sources']})
            self.condition_mapping_layout.addWidget(QLabel(f"Averaging Rule: '{mapping['output_label']}'"))
            self.condition_mapping_layout.addWidget(QLabel(f"Averages IDs: {', '.join(map(str, ids))}"))
        self.pool_rb.setChecked(g.get('averaging_method', 'Pool Trials') == 'Pool Trials')

    def _update_current_group_avg_method(self) -> None:
        self.debug("_update_current_group_avg_method called")
        idx = self.selected_group_index
        if idx is None:
            self.debug("No group selected when updating averaging method")
            return
        method = 'Pool Trials' if self.pool_rb.isChecked() else 'Average of Averages'
        g = self.defined_groups[idx]
        if g.get('averaging_method') != method:
            g['averaging_method'] = method
            g['config_saved'] = False
            self.debug(f"Group {idx} averaging method changed to {method}")
            self._update_groups_listbox()
            self.save_group_config_btn.setEnabled(True)
            self._update_start_processing_button_state()

    def save_current_group_config(self) -> bool:
        self.debug("save_current_group_config called")
        idx = self.selected_group_index
        if idx is None:
            self.log_signal.emit("No group selected to save configuration for.")
            self.debug("Save requested with no group selected")
            return False
        g = self.defined_groups[idx]
        g['config_saved'] = True
        self.debug(f"Group {idx} configuration saved")
        self.log_signal.emit(f"Configuration saved for group '{g['name']}'.")
        self._update_groups_listbox()
        self.save_group_config_btn.setEnabled(False)
        self._update_start_processing_button_state()
        QMessageBox.information(self, "Configuration Saved", f"Configuration for group '{g['name']}' has been saved.")
        return True
