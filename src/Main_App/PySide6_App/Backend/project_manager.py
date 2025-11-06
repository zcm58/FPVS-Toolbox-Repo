""""Project management utilities extracted from main_window.py."""
from __future__ import annotations

import logging
from pathlib import Path
import re
import sys

from PySide6.QtWidgets import QFileDialog, QMessageBox, QInputDialog, QWidget

from .project import Project
from Main_App.PySide6_App.config.projects_root import ensure_projects_root
from Main_App.PySide6_App.utils.op_guard import OpGuard
from Main_App.PySide6_App.utils.settings import get_app_settings

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_open_project_guard = OpGuard()


def _sanitize_windows_name(name: str) -> str:
    """
    Make a safe Windows folder name.
    Removes reserved characters <>:\"/\\|?* and trailing spaces/dots.
    Collapses whitespace to single spaces.
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
    # Normalize whitespace
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    # Strip trailing dots/spaces
    sanitized = sanitized.rstrip(" .")
    return sanitized


def select_projects_root(self) -> None:
    settings = get_app_settings()
    settings.beginGroup("paths")
    saved_root = settings.value("projectsRoot", "", type=str)
    settings.endGroup()

    if saved_root and Path(saved_root).is_dir():
        self.projectsRoot = Path(saved_root)
    else:
        root = QFileDialog.getExistingDirectory(
            self, "Select Projects Root Folder", ""
        )
        if not root:
            QMessageBox.critical(
                self,
                "Projects Root Required",
                "You must select a Projects Root folder to continue.",
            )
            sys.exit(1)
        self.projectsRoot = Path(root)
        settings.beginGroup("paths")
        settings.setValue("projectsRoot", str(self.projectsRoot))
        settings.endGroup()
        settings.sync()


def new_project(self) -> None:
    # Project name
    name, ok = QInputDialog.getText(
        self, "Project Name", "Enter a name for this new project:"
    )
    if not ok or not name.strip():
        return
    proj_name = _sanitize_windows_name(name.strip())
    if not proj_name:
        QMessageBox.warning(self, "Invalid Name", "Please enter a valid project name.")
        return

    # Input folder
    input_folder = QFileDialog.getExistingDirectory(
        self, "Select Input Folder (BDF files)", ""
    )
    if not input_folder:
        return

    # Group count (no hard cap; use a high upper bound)
    group_count, ok = QInputDialog.getInt(
        self,
        "Experimental Groups",
        "How many experimental groups?",
        1,  # default
        1,  # min
        999,  # max (effectively uncapped for practical purposes)
        1,
    )
    if not ok:
        return

    # Group names
    group_names: list[str] = []
    for idx in range(group_count):
        while True:
            gname_raw, gok = QInputDialog.getText(
                self,
                "Group Name",
                f"Enter name for Group {idx + 1}:",
            )
            if not gok:
                return  # cancel entire creation
            gname = _sanitize_windows_name(gname_raw.strip())
            if not gname:
                QMessageBox.warning(self, "Invalid Name", "Group name cannot be empty.")
                continue
            if gname in group_names:
                QMessageBox.warning(self, "Duplicate Name", "Group name must be unique.")
                continue
            group_names.append(gname)
            break

    # Create project folder
    project_dir = self.projectsRoot / proj_name
    project_dir.mkdir(parents=True, exist_ok=True)

    # Create group folders (keep existing per-group scheme creation elsewhere unchanged)
    created = []
    for g in group_names:
        try:
            (project_dir / g).mkdir(parents=True, exist_ok=True)
            created.append(g)
        except (OSError, PermissionError) as exc:
            logger.error(
                "Failed to create group folder",
                exc_info=exc,
                extra={"op": "new_project", "project": proj_name, "group": g, "path": str(project_dir / g)},
            )
            QMessageBox.critical(
                self,
                "Folder Creation Error",
                f"Could not create group folder:\n{project_dir / g}\n{exc}",
            )
            return

    # Initialize and save manifest
    project = Project.load(project_dir)
    project.name = proj_name
    project.input_folder = input_folder
    # Store groups in manifest via existing options dict to avoid schema breaks elsewhere
    try:
        if not isinstance(project.options, dict):
            project.options = {}
    except AttributeError:
        # If Project lacks .options, we still proceed without storing groups
        logger.warning(
            "Project object has no 'options' attribute; groups will not be persisted.",
            extra={"op": "new_project", "project": proj_name},
        )
    else:
        project.options["groups"] = created  # persists in project.json
        project.options["has_groups"] = True

    try:
        project.save()
    except Exception as exc:
        logger.error(
            "Failed to save project.json",
            exc_info=exc,
            extra={"op": "new_project", "project": proj_name, "path": str(project_dir)},
        )
        QMessageBox.critical(
            self,
            "Save Error",
            f"Could not save project at:\n{project_dir}\n{exc}",
        )
        return

    # Load into UI
    self.currentProject = project
    self.loadProject(project)


def open_existing_project(self, parent: QWidget | None = None) -> None:
    if not _open_project_guard.start():
        return

    try:
        # Derive a parent if not provided to keep backward compatibility with older callers
        if parent is None:
            parent = getattr(self, "window", lambda: None)() or getattr(self, "parent", lambda: None)() or None

        root = ensure_projects_root(parent)
        if root is None:
            QMessageBox.information(parent, "Projects Root", "Project root not set.")
            logger.warning(
                "Projects root missing.",
                extra={"op": "open_existing_project", "path": None},
            )
            return

        self.projectsRoot = root

        try:
            entries = list(root.iterdir())
        except (FileNotFoundError, PermissionError, OSError) as exc:
            logger.error(
                "Unable to enumerate projects under %s: %s",
                root,
                exc,
                exc_info=exc,
                extra={"op": "open_existing_project", "path": str(root)},
            )
            QMessageBox.critical(
                parent,
                "Projects Root Unavailable",
                f"Unable to access projects root:\n{root}\n{exc}",
            )
            return

        candidates = [d for d in entries if (d / "project.json").exists()]
        if not candidates:
            QMessageBox.information(
                parent,
                "No Projects Found",
                f"No projects found under {root}.",
            )
            logger.info(
                "No projects discovered under %s.",
                root,
                extra={"op": "open_existing_project", "path": str(root)},
            )
            return

        labels: list[str] = []
        label_to_path: dict[str, Path] = {}
        for candidate in candidates:
            try:
                project = Project.load(candidate)
            except Exception as exc:
                logger.error(
                    "Failed to load project at %s: %s",
                    candidate,
                    exc,
                    exc_info=exc,
                    extra={"op": "open_existing_project", "path": str(candidate)},
                )
                continue
            label = project.name
            labels.append(label)
            label_to_path[label] = candidate

        if not labels:
            QMessageBox.warning(
                parent,
                "Projects Unavailable",
                "No valid projects could be loaded.",
            )
            return

        choice, ok = QInputDialog.getItem(
            parent,
            "Open Existing Project",
            "Select a project:",
            labels,
            0,
            editable=False,
        )
        if not ok or choice not in label_to_path:
            logger.info(
                "Project selection cancelled.",
                extra={"op": "open_existing_project", "path": str(root)},
            )
            return

        project = Project.load(label_to_path[choice])
        self.currentProject = project
        self.loadProject(project)
    finally:
        _open_project_guard.end()


def openProjectPath(self, folder: str) -> None:
    project = Project.load(folder)
    self.currentProject = project
    self.loadProject(project)

    settings = get_app_settings()
    recent = settings.value("recentProjects", [], type=list)
    if folder in recent:
        recent.remove(folder)
    recent.insert(0, folder)
    settings.setValue("recentProjects", recent)
    settings.sync()


def loadProject(self, project: Project) -> None:
    self.currentProject = project
    self.lbl_currentProject.setText(f"Current Project: {project.name}")

    self.settings.set("paths", "data_folder", str(project.input_folder))
    self.settings.save()

    mode = project.options.get("mode", "batch").lower()
    self.rb_single.setChecked(mode == "single")
    self.rb_batch.setChecked(mode == "batch")
    self.cb_loreta.setChecked(bool(project.options.get("run_loreta", False)))

    for row in list(self.event_rows):
        row.setParent(None)
    self.event_rows.clear()

    if project.event_map:
        for label, ident in project.event_map.items():
            self.add_event_row(str(label), str(ident))
    else:
        self.add_event_row()

    self.log(f"Loaded project: {project.name}")


def edit_project_settings(self) -> None:
    if not getattr(self, "currentProject", None):
        QMessageBox.warning(self, "No Project", "Please open or create a project first.")
        return
    folder = QFileDialog.getExistingDirectory(
        self, "Select Input Folder", str(self.currentProject.input_folder)
    )
    if not folder:
        return
    self.currentProject.input_folder = folder
    self.currentProject.save()
    self.loadProject(self.currentProject)
