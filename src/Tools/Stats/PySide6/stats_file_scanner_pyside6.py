# src/Tools/Stats/PySide6/stats_file_scanner_pyside6.py
from __future__ import annotations

import os
import glob
import re
from typing import List, Dict, Tuple


# Folders to ignore during scanning (case-insensitive)
IGNORED_FOLDERS = {".fif files", "loreta results"}


class ScanError(Exception):
    """Exception raised when scanning fails due to invalid folder or permissions."""
    pass


def _clean_name(name: str) -> str:
    """
    Remove leading ordering tokens like '1 - ' or '02_' and trim.
    Keeps internal spaces and hyphens for readability.
    """
    return re.sub(r'^\d+\s*[-_]*\s*', '', name or '').strip()


def _collect_from_condition_folder(
    condition_folder: str,
    condition_label: str,
    subjects_set: set,
    subject_data: Dict[str, Dict[str, str]],
    pid_pattern: re.Pattern[str],
    id_prefix: str | None = None,
) -> bool:
    """
    Scan one condition folder for .xlsx files and record subject -> condition file.
    Returns True if any files were recorded.
    """
    found_any = False
    for filepath in glob.glob(os.path.join(condition_folder, "*.xlsx")):
        filename = os.path.basename(filepath)
        match = pid_pattern.search(filename)
        if not match:
            continue
        pid_raw = match.group(1).upper()  # e.g., P1, P12
        pid = f"{id_prefix}:{pid_raw}" if id_prefix else pid_raw

        subjects_set.add(pid)
        subject_data.setdefault(pid, {})
        subject_data[pid][condition_label] = filepath
        found_any = True
    return found_any


def scan_folder_simple(parent_folder: str) -> Tuple[List[str], List[str], Dict[str, Dict[str, str]]]:
    """
    Scans the given parent folder for subject Excel files and condition subfolders.

    Supported layouts
    -----------------
    A) Classic (no groups):
        <parent_folder>/
            <Condition A>/*.xlsx
            <Condition B>/*.xlsx
    B) Grouped (two-level):
        <parent_folder>/
            <Group A>/<Condition A>/*.xlsx
            <Group A>/<Condition B>/*.xlsx
            <Group B>/<Condition A>/*.xlsx
            ...

    Subject IDs
    -----------
    - Extracted from filenames by 'P<digits>' (case-insensitive), e.g., P1_BC_F.xlsx.
    - When layout B is detected, IDs are prefixed with the cleaned group name:
      '<Group>:<P#>' so that duplicate P-numbers across groups do not collide.

    Returns:
        subjects: sorted list of subject IDs (strings)
        conditions: sorted list of condition names (cleaned folder names)
        subject_data: mapping {subject_id: {condition_name: full_file_path}}

    Raises:
        ScanError: if the folder is invalid or access is denied.
    """
    if not parent_folder or not os.path.isdir(parent_folder):
        raise ScanError(f"Invalid or missing parent folder: {parent_folder}")

    subjects_set: set[str] = set()
    conditions_set: set[str] = set()
    subject_data: Dict[str, Dict[str, str]] = {}

    # Match subject IDs in filenames (e.g., P1, p23) before .xlsx
    pid_pattern = re.compile(r"(P\d+)(?=\.xlsx$)", re.IGNORECASE)

    try:
        # List first-level entries under parent folder
        first_level = [
            entry for entry in os.listdir(parent_folder)
            if os.path.isdir(os.path.join(parent_folder, entry))
            and entry.lower() not in IGNORED_FOLDERS
        ]

        # Decide if layout is grouped: if any first-level dir contains subdirs
        # that themselves contain .xlsx files, treat first-level as groups.
        grouped_layout = False
        for entry in first_level:
            entry_path = os.path.join(parent_folder, entry)
            # If this level already has .xlsx files directly under it,
            # it's likely a condition folder (classic layout).
            if glob.glob(os.path.join(entry_path, "*.xlsx")):
                grouped_layout = False
                break
            # If it has subfolders that contain .xlsx, it's grouped.
            subdirs = [
                d for d in os.listdir(entry_path)
                if os.path.isdir(os.path.join(entry_path, d))
                and d.lower() not in IGNORED_FOLDERS
            ]
            for sd in subdirs:
                if glob.glob(os.path.join(entry_path, sd, "*.xlsx")):
                    grouped_layout = True
                    break
            if grouped_layout:
                break

        if not grouped_layout:
            # Classic layout: first-level entries are conditions
            for condition_dir in first_level:
                condition_path = os.path.join(parent_folder, condition_dir)
                condition_clean = _clean_name(condition_dir)
                if not condition_clean:
                    continue

                found = _collect_from_condition_folder(
                    condition_path,
                    condition_clean,
                    subjects_set,
                    subject_data,
                    pid_pattern,
                    id_prefix=None,
                )
                if found:
                    conditions_set.add(condition_clean)
        else:
            # Grouped layout: first-level entries are groups; second-level are conditions
            for group_dir in first_level:
                group_path = os.path.join(parent_folder, group_dir)
                group_clean = _clean_name(group_dir)
                if not group_clean:
                    continue

                subdirs = [
                    d for d in os.listdir(group_path)
                    if os.path.isdir(os.path.join(group_path, d))
                    and d.lower() not in IGNORED_FOLDERS
                ]
                for condition_dir in subdirs:
                    condition_path = os.path.join(group_path, condition_dir)
                    condition_clean = _clean_name(condition_dir)
                    if not condition_clean:
                        continue

                    found = _collect_from_condition_folder(
                        condition_path,
                        condition_clean,
                        subjects_set,
                        subject_data,
                        pid_pattern,
                        id_prefix=group_clean,  # disambiguate subjects across groups
                    )
                    if found:
                        conditions_set.add(condition_clean)

    except PermissionError as e:
        raise ScanError(f"Permission denied to access folder: {parent_folder}\n{e}")
    except Exception as e:
        raise ScanError(f"Error scanning folder '{parent_folder}': {e}")

    subjects = sorted(subjects_set)
    conditions = sorted(conditions_set)
    return subjects, conditions, subject_data
