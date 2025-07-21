"""Utility mixin for PID extraction and config persistence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

from .advanced_analysis_base import AdvancedAnalysisBase


class AdvancedAnalysisPostMixin(AdvancedAnalysisBase):
    """Mix-in providing helpers for saving/loading group configs."""

    def _extract_pid_for_group(self, group_data: Dict[str, Any]) -> str:
        if self.debug_mode:
            self.debug("Extracting PID for group")
        file_paths = group_data.get("file_paths", [])
        if not file_paths:
            return "UnknownPID"
        base_name = Path(file_paths[0]).stem
        pid_regex_primary = r"\b(P\d+|S\d+|Sub\d+)\b"
        match = re.search(pid_regex_primary, base_name, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        if "_" in base_name:
            parts = base_name.split("_")
            for part in parts:
                if re.fullmatch(r"(P\d+|S\d+|Sub\d+)", part, re.IGNORECASE):
                    return part.upper()
        cleaned_pid = re.sub(
            r"(_unamb|_ambig|_mid|_run\d*|_sess\d*|_task\w*|_eeg|_raw|_preproc|_ica).*",
            "",
            base_name,
            flags=re.IGNORECASE,
        )
        cleaned_pid = re.sub(r"[^A-Za-z0-9_]", "", cleaned_pid)
        if "_" in cleaned_pid:
            for part in cleaned_pid.split("_"):
                if re.fullmatch(r"(P\d+|S\d+|Sub\d+)", part, re.IGNORECASE):
                    return part.upper()
        cleaned_pid_alphanum_only = re.sub(r"[^A-Za-z0-9]", "", cleaned_pid)
        result_pid = cleaned_pid_alphanum_only if cleaned_pid_alphanum_only else base_name
        if self.debug_mode:
            self.debug(f"Extracted PID: {result_pid}")
        return result_pid

    def save_groups_to_file(self, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.defined_groups, f, indent=2)
        self.log(f"Saved group configuration to {file_path}.")

    def load_groups_from_file(self, file_path: str) -> None:
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
        self.log(f"Loaded {len(self.defined_groups)} group(s) from {file_path}.")

