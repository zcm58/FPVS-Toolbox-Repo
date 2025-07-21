"""Group management helpers for advanced averaging."""

from __future__ import annotations

from typing import List

from .advanced_analysis_base import AdvancedAnalysisBase


class AdvancedAnalysisGroupOpsMixin(AdvancedAnalysisBase):
    """Mix-in implementing group creation and manipulation logic."""

    def create_new_group(self, name: str, ids_to_average: List[int]) -> None:
        if any(g["name"] == name for g in self.defined_groups):
            raise ValueError(f"A group named '{name}' already exists.")
        if not self.source_eeg_files:
            raise ValueError("No EEG files selected. Add files first.")
        mapping_rule = {"output_label": name, "sources": []}
        for f_path in self.source_eeg_files:
            for event_id_val in ids_to_average:
                mapping_rule["sources"].append(
                    {"file_path": f_path, "original_label": str(event_id_val), "original_id": event_id_val}
                )
        new_grp_data = {
            "name": name,
            "file_paths": list(self.source_eeg_files),
            "condition_mappings": [mapping_rule],
            "averaging_method": "Pool Trials",
            "config_saved": True,
            "ui_mapping_rules": [],
        }
        self.defined_groups.append(new_grp_data)
        self.selected_group_index = len(self.defined_groups) - 1
        self.log(f"Created group '{name}' averaging IDs {ids_to_average} in all {len(self.source_eeg_files)} files.")

    def delete_selected_group(self) -> None:
        if self.selected_group_index is None:
            raise ValueError("No group selected to delete.")
        group_name = self.defined_groups[self.selected_group_index]["name"]
        del self.defined_groups[self.selected_group_index]
        self.selected_group_index = None
        self.log(f"Deleted group: {group_name}")

    def rename_selected_group(self, new_name: str) -> None:
        if self.selected_group_index is None:
            raise ValueError("No group selected to rename.")
        if any(i != self.selected_group_index and g["name"] == new_name for i, g in enumerate(self.defined_groups)):
            raise ValueError(f"A group named '{new_name}' already exists.")
        self.defined_groups[self.selected_group_index]["name"] = new_name
        self.defined_groups[self.selected_group_index]["config_saved"] = False
        self.log(f"Renamed group to '{new_name}'.")

    def _update_current_group_avg_method(self, method: str) -> None:
        idx = self.selected_group_index
        if idx is None:
            return
        group_data = self.defined_groups[idx]
        if group_data.get("averaging_method") != method:
            group_data["averaging_method"] = method
            group_data["config_saved"] = False
            self.log(f"Averaging method for '{group_data['name']}' changed to '{method}'. Needs re-saving.")

