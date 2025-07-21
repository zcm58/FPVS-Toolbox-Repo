"""File handling utilities for advanced averaging logic."""

from __future__ import annotations

from typing import List

from .advanced_analysis_base import AdvancedAnalysisBase


class AdvancedAnalysisFileOpsMixin(AdvancedAnalysisBase):
    """Mix-in providing basic file list management."""

    def add_source_files(self, files: List[str]) -> None:
        """Add ``files`` to the internal source file list."""
        added = 0
        for f_path in files:
            if f_path not in self.source_eeg_files:
                self.source_eeg_files.append(f_path)
                added += 1
        if added:
            self.source_eeg_files.sort()
            self.log(f"Added {added} source file(s). Total: {len(self.source_eeg_files)}.")

    def remove_source_files(self, files: List[str]) -> None:
        """Remove ``files`` from the internal source list."""
        removed = [fp for fp in files if fp in self.source_eeg_files]
        if not removed:
            return
        self.source_eeg_files = [fp for fp in self.source_eeg_files if fp not in removed]
        self._check_groups_for_removed_files(removed)
        self.log(f"Removed {len(removed)} file(s) from source list.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_groups_for_removed_files(self, removed_paths: List[str]) -> None:
        """Prune references to removed files from all defined groups."""
        updated_any = False
        affected_indices: List[int] = []
        for idx, group in enumerate(self.defined_groups):
            original_count = len(group["file_paths"])
            group["file_paths"] = [fp for fp in group["file_paths"] if fp not in removed_paths]
            if len(group["file_paths"]) != original_count:
                self.log(f"Group '{group['name']}' updated: removed files.")
                updated_any = True
                affected_indices.append(idx)
                group["config_saved"] = False
            if "condition_mappings" in group:
                new_rules = []
                for rule in group["condition_mappings"]:
                    original_sources = len(rule["sources"])
                    rule["sources"] = [s for s in rule["sources"] if s["file_path"] not in removed_paths]
                    if original_sources != len(rule["sources"]):
                        self.log(
                            f"  Mapping '{rule['output_label']}' in group '{group['name']}' lost {original_sources - len(rule['sources'])} source(s)."
                        )
                    if rule["sources"]:
                        new_rules.append(rule)
                group["condition_mappings"] = new_rules
                if not new_rules:
                    self.log(f"Group '{group['name']}' has no valid mapping rules left after file removal.")
        if updated_any:
            if self.selected_group_index is not None and self.selected_group_index in affected_indices:
                self.selected_group_index = None


