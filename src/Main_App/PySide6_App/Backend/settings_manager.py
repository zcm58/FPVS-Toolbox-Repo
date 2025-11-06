# src/Main_App/PySide6_App/Backend/settings_manager.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Set

from PySide6.QtCore import QSettings, QCoreApplication, QStandardPaths

# Try relative, then absolute. If both fail, define a local minimal get_app_settings.
try:
    # Three dots → up to Main_App
    from ...settings import get_app_settings  # type: ignore
except Exception:
    try:
        from Main_App.settings import get_app_settings  # type: ignore
    except Exception:
        # Last-resort local shim. Writes to AppDataLocation/settings.ini.
        def get_app_settings() -> QSettings:
            if not QCoreApplication.organizationName():
                QCoreApplication.setOrganizationName("MississippiStateUniversity")
            if not QCoreApplication.applicationName():
                QCoreApplication.setApplicationName("FPVS Toolbox")
            loc = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
            if not loc:
                raise RuntimeError("No AppDataLocation for settings storage.")
            Path(loc).mkdir(parents=True, exist_ok=True)
            ini_path = str(Path(loc) / "settings.ini")
            qs = QSettings(ini_path, QSettings.IniFormat)
            qs.setFallbacksEnabled(False)
            return qs


# ---------- Legacy-defaults parity ----------
DEFAULTS: Dict[str, Dict[str, str]] = {
    "appearance": {"mode": "System", "theme": "blue"},
    "paths": {"data_folder": "", "output_folder": ""},
    "gui": {
        "main_size": "750x920",
        "stats_size": "700x650",
        "resizer_size": "600x600",
        "advanced_size": "500x500",
    },
    "stim": {"channel": "Status"},
    "events": {"labels": "", "ids": ""},
    "analysis": {"base_freq": "6.0", "oddball_freq": "1.2", "bca_upper_limit": "16.8", "alpha": "0.05"},
    "rois": {
        "names": "Frontal Lobe;Central Lobe;Parietal Lobe;Occipital Lobe",
        "electrodes": "F3,F4,Fz;C3,C4,Cz;P3,P4,Pz;O1,O2,Oz",
    },
    "loreta": {
        "mri_path": "",
        "loreta_low_freq": "1.1",
        "loreta_high_freq": "1.3",
        "loreta_threshold": "0.3",
        "oddball_harmonics": "1.2,2.4,3.6,4.8,7.2,8.4,9.6,10.8",
        "loreta_snr": "3.0",
        "auto_oddball_localization": "False",
        "baseline_tmin": "-0.2",
        "baseline_tmax": "0.0",
        "time_window_start_ms": "-1000",
        "time_window_end_ms": "1000",
    },
    "visualization": {
        "threshold": "0.0",
        "surface_opacity": "0.5",
        "time_index_ms": "100",
        "show_brain_mesh": "True",
    },
    "debug": {"enabled": "False"},
    # keep "app/debug_enabled" alias for new code paths
    "app": {"debug_enabled": "False"},
}

CONFIGS_DIR_NAME = "configs"


def _user_base_dir() -> Path:
    """User-writable base directory where settings live (same root as QSettings file)."""
    qs = get_app_settings()
    p = Path(qs.fileName())
    if not str(p):
        home = Path(os.environ.get("APPDATA", Path.home()))
        return home / "FPVS_Toolbox"
    return p.parent


def _configs_dir() -> Path:
    return _user_base_dir() / CONFIGS_DIR_NAME


def _split_key(full: str) -> Tuple[str, str]:
    parts = full.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", full


class SettingsManager:
    """
    PySide6 settings manager that preserves the legacy API:
    - get/set(section, option, value)
    - export/import, named profiles, list/delete/load_named
    - ROI/event helpers
    - debug_enabled()
    - save() compatibility (flushes QSettings)
    Backed by QSettings (INI at AppDataLocation/settings.ini).
    """

    def __init__(self) -> None:
        self._qs: QSettings = get_app_settings()
        self._ensure_defaults()

    # ---------- defaults ----------
    def _ensure_defaults(self) -> None:
        self._qs.sync()
        for section, items in DEFAULTS.items():
            for opt, val in items.items():
                key = f"{section}/{opt}"
                if self._qs.value(key, None) in (None, ""):
                    self._qs.setValue(key, val)
        self._qs.sync()

    # ---------- core get/set ----------
    def get(self, section: str, option: str, fallback: str = "") -> str:
        return self._qs.value(f"{section}/{option}", fallback)

    def set(self, section: str, option: str, value: str) -> None:
        self._qs.setValue(f"{section}/{option}", value)
        self._qs.sync()

    # Legacy compatibility: some callers expect .save()
    def save(self) -> None:
        """No-op compatibility. Ensures settings are flushed to disk."""
        try:
            self._qs.sync()
        except Exception:
            pass

    # Convenience typed accessors
    def get_bool(self, section: str, option: str, fallback: bool = False) -> bool:
        return bool(self._qs.value(f"{section}/{option}", fallback, type=bool))

    def get_float(self, section: str, option: str, fallback: float = 0.0) -> float:
        return float(self._qs.value(f"{section}/{option}", fallback, type=float))

    def get_int(self, section: str, option: str, fallback: int = 0) -> int:
        return int(self._qs.value(f"{section}/{option}", fallback, type=int))

    def remove(self, section: str, option: str) -> None:
        self._qs.remove(f"{section}/{option}")
        self._qs.sync()

    # ---------- export/import ----------
    def export(self, path: str) -> None:
        keys = set(self._qs.allKeys())
        for s, kv in DEFAULTS.items():
            for o in kv:
                keys.add(f"{s}/{o}")

        sections: Dict[str, Dict[str, str]] = {}
        for full in sorted(keys):
            section, option = _split_key(full)
            if not section or not option:
                continue
            val = str(self._qs.value(full, DEFAULTS.get(section, {}).get(option, "")))
            sections.setdefault(section, {})[option] = val

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.suffix.lower() == ".json":
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(sections, f, indent=2)
        else:
            from configparser import ConfigParser
            cfg = ConfigParser()
            for s, kv in sections.items():
                cfg[s] = kv
            with open(dest, "w", encoding="utf-8") as f:
                cfg.write(f)

    def export_named(self, name: str, ext: str = ".ini") -> str:
        out_dir = _configs_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        if not ext.startswith("."):
            ext = "." + ext
        path = out_dir / f"{name}{ext}"
        self.export(str(path))
        return str(path)

    def load_from(self, path: str) -> None:
        src = Path(path)
        if not src.is_file():
            raise FileNotFoundError(path)

        if src.suffix.lower() == ".json":
            with open(src, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("JSON settings must be an object of sections")
            for section, mapping in data.items():
                if not isinstance(mapping, dict):
                    continue
                for opt, val in mapping.items():
                    self._qs.setValue(f"{section}/{opt}", str(val))
        else:
            from configparser import ConfigParser
            cfg = ConfigParser()
            cfg.read(src, encoding="utf-8")
            for section in cfg.sections():
                for opt, val in cfg.items(section):
                    self._qs.setValue(f"{section}/{opt}", val)
        self._qs.sync()

    def reset(self) -> None:
        self._qs.clear()
        self._ensure_defaults()

    # ---------- named profiles ----------
    def list_configs(self) -> List[str]:
        d = _configs_dir()
        if not d.is_dir():
            return []
        names: Set[str] = set()
        for f in d.iterdir():
            if f.suffix.lower() in {".ini", ".json"}:
                names.add(f.stem)
        return sorted(names)

    def delete_named(self, name: str) -> None:
        d = _configs_dir()
        removed = False
        for ext in (".ini", ".json"):
            p = d / f"{name}{ext}"
            if p.exists():
                p.unlink()
                removed = True
        if not removed:
            raise FileNotFoundError(name)

    def load_named(self, name: str) -> None:
        d = _configs_dir()
        for ext in (".ini", ".json"):
            p = d / f"{name}{ext}"
            if p.exists():
                self.load_from(str(p))
                return
        raise FileNotFoundError(name)

    # ---------- helpers: events / rois ----------
    def get_event_pairs(self) -> List[Tuple[str, str]]:
        labels = [s.strip() for s in self.get("events", "labels", "").split(",") if s.strip()]
        ids = [s.strip() for s in self.get("events", "ids", "").split(",") if s.strip()]
        return list(zip(labels, ids))

    def set_event_pairs(self, pairs: List[Tuple[str, str]]) -> None:
        labels = ",".join(p[0] for p in pairs)
        ids = ",".join(p[1] for p in pairs)
        self.set("events", "labels", labels)
        self.set("events", "ids", ids)

    def get_roi_pairs(self) -> List[Tuple[str, List[str]]]:
        names = [n.strip() for n in self.get("rois", "names", "").split(";") if n.strip()]
        groups = [g.strip() for g in self.get("rois", "electrodes", "").split(";") if g.strip()]
        pairs: List[Tuple[str, List[str]]] = []
        for name, group in zip(names, groups):
            electrodes = [e.strip().upper() for e in group.split(",") if e.strip()]
            pairs.append((name, electrodes))
        return pairs

    def set_roi_pairs(self, pairs: List[Tuple[str, List[str]]]) -> None:
        names = ";".join(p[0] for p in pairs)
        electrodes = ";".join(",".join(e.upper() for e in p[1]) for p in pairs)
        self.set("rois", "names", names)
        self.set("rois", "electrodes", electrodes)

    # ---------- debug ----------
    def debug_enabled(self) -> bool:
        raw = self.get("debug", "enabled", self.get("app", "debug_enabled", "False"))
        return str(raw).strip().lower() == "true"
