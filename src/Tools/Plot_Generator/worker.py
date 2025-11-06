"""Worker classes for the plot generator."""
from __future__ import annotations

import os
import re
import math
import json
from pathlib import Path
from typing import Dict, List, Iterable, Sequence, Optional, Tuple, Set

import pandas as pd
import matplotlib
import numpy as np
from Main_App import SettingsManager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PySide6.QtCore import QObject, Signal

from Tools.Stats.Legacy.stats_analysis import ALL_ROIS_OPTION
from Tools.Plot_Generator.snr_utils import calc_snr_matlab

# Global plotting style applied after imports
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "lines.linewidth": 1.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
    }
)

# Accept "P7" and "SCP7" and canonicalize to P{digits}
_PID_RE = re.compile(r"(?i)\b(?:P|SCP)0*(\d{1,4})\b")


def _pid_from_filename(path: Path) -> Optional[str]:
    m = _PID_RE.search(path.name)
    if not m:
        return None
    return f"P{int(m.group(1))}"


def _parse_groups_manifest(project_json: Path) -> Tuple[Dict[str, Set[str]], Optional[str]]:
    """Parse groups from project.json using tolerant rules."""

    def _canon_pid_token(txt: str) -> Optional[str]:
        m = _PID_RE.search(txt or "")
        return f"P{int(m.group(1))}" if m else None

    try:
        with open(project_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        return {}, f"Failed to read project.json: {e}"

    key_variants = {
        "group_members",
        "groups",
        "experimental_groups",
        "GroupMembers",
        "Group_Members",
        "groupNames",
        "group_names",
    }

    def _collect(val) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {}
        if isinstance(val, dict):
            for name, lst in val.items():
                if not isinstance(name, str):
                    continue
                if isinstance(lst, list):
                    s: Set[str] = set()
                    for pid in lst:
                        tok = _canon_pid_token(str(pid))
                        if tok:
                            s.add(tok)
                    out[name] = s
                else:
                    out[name] = set()
        elif isinstance(val, list):
            names_only = [v for v in val if isinstance(v, str)]
            if names_only and len(names_only) == len(val):
                for name in names_only:
                    out[name] = set()
            else:
                for obj in val:
                    if not isinstance(obj, dict):
                        continue
                    name = (
                        str(
                            obj.get("name")
                            or obj.get("label")
                            or obj.get("group")
                            or ""
                        ).strip()
                    )
                    if not name:
                        continue
                    pids = obj.get("pids") or obj.get("members") or obj.get("ids") or []
                    s: Set[str] = set()
                    if isinstance(pids, list):
                        for pid in pids:
                            tok = _canon_pid_token(str(pid))
                            if tok:
                                s.add(tok)
                    out[name] = s
        return out

    def _find_groups(node) -> Optional[Dict[str, Set[str]]]:
        if isinstance(node, dict):
            for k, v in node.items():
                if k in key_variants:
                    gm = _collect(v)
                    if gm:
                        return gm
            for v in node.values():
                gm = _find_groups(v)
                if gm:
                    return gm
        elif isinstance(node, list):
            for v in node:
                gm = _find_groups(v)
                if gm:
                    return gm
        return None

    group_map = _find_groups(cfg) or {}
    if not group_map:
        return {}, "No valid 'group_members' or 'groups' found."
    return {str(name): set(pids) for name, pids in group_map.items()}, None


class _Worker(QObject):
    """Worker to process Excel files and generate plots."""

    progress = Signal(str, int, int)
    error = Signal(str)
    finished = Signal()
    finished_payload = Signal(object)

    def __init__(
        self,
        folder: str,
        condition: str,
        roi_map: Dict[str, List[str]],
        roi_choice: str,
        title: str,
        xlabel: str,
        ylabel: str,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        out_dir: str,
        stem_color: str = "red",
        *,
        condition_b: str | None = None,
        stem_color_b: str = "blue",
        oddballs: Sequence[float] | None = None,
        use_matlab_style: bool = False,
        overlay: bool = False,
        # new flags for group comparison
        compare_groups: bool = False,
        group_a: str | None = None,
        group_b: str | None = None,
    ) -> None:
        super().__init__()
        self.folder = folder
        self.condition = condition
        self.roi_map = roi_map
        self.selected_roi = roi_choice
        self.metric = "SNR"
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.out_dir = Path(out_dir)
        self.stem_color = (stem_color or "red").lower()
        self.stem_color_b = (stem_color_b or "blue").lower()
        self.condition_b = condition_b
        self.overlay = overlay

        self.compare_groups = compare_groups
        self.group_a = (group_a or "").strip()
        self.group_b = (group_b or "").strip()

        # maintain oddballs attribute for compatibility with older versions
        self.oddballs: List[float] = list(oddballs or [])
        self.use_matlab_style = use_matlab_style
        self._stop_requested = False

        # project manifest cache
        self._project_root: Optional[Path] = None
        self._group_members: Dict[str, Set[str]] = {}

    # ---------- lifecycle ----------

    def run(self) -> None:
        try:
            payload = self._run()
            self.finished_payload.emit(payload)
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"Worker failed: {type(e).__name__}: {e}")
            self.finished.emit()

    def stop(self) -> None:
        self._stop_requested = True

    # ---------- logging/progress ----------

    def _emit(self, msg: str, processed: int = 0, total: int = 0) -> None:
        self.progress.emit(msg, processed, total)

    # ---------- project + groups ----------

    def _find_project_root(self) -> Optional[Path]:
        if self._project_root:
            return self._project_root
        p = Path(self.folder).resolve()
        for cand in [p, *p.parents]:
            if (cand / "project.json").is_file():
                self._project_root = cand
                return cand
        return None

    def _load_group_members(self) -> None:
        """Load group membership from project.json using tolerant parser."""
        root = self._find_project_root()
        if not root:
            return
        project_json = root / "project.json"
        group_map, _ = _parse_groups_manifest(project_json)
        self._group_members = group_map

    def _subjects_for_group(self, group_name: str) -> Set[str]:
        if not self._group_members:
            self._load_group_members()
        return self._group_members.get(group_name, set())

    # ---------- discovery ----------

    def _count_excel_files(self, condition: str, include_subjects: Optional[Set[str]] = None) -> int:
        cond_folder = Path(self.folder) / condition
        if not cond_folder.is_dir():
            return 0
        n = 0
        for root, _, files in os.walk(cond_folder):
            for f in files:
                if not f.lower().endswith(".xlsx"):
                    continue
                if include_subjects is not None:
                    pid = _pid_from_filename(Path(root) / f)
                    if not pid or pid not in include_subjects:
                        continue
                n += 1
        return n

    # ---------- IO + aggregation ----------

    def _collect_data(
        self,
        condition: str,
        *,
        offset: int = 0,
        total_override: int | None = None,
        include_subjects: Optional[Set[str]] = None,
    ) -> tuple[List[float], Dict[str, List[float]]]:
        cond_folder = Path(self.folder) / condition
        if not cond_folder.is_dir():
            self._emit(f"Condition folder not found: {cond_folder}")
            return [], {}

        self.out_dir.mkdir(parents=True, exist_ok=True)

        excel_files: List[Path] = []
        for root, _, files in os.walk(cond_folder):
            for f in files:
                if not f.lower().endswith(".xlsx"):
                    continue
                p = Path(root) / f
                if include_subjects is not None:
                    pid = _pid_from_filename(p)
                    if not pid or pid not in include_subjects:
                        continue
                excel_files.append(p)

        if not excel_files:
            self._emit("No Excel files found for condition.")
            return [], {}

        total_files = len(excel_files)
        overall_total = total_override if total_override is not None else total_files
        processed_files = 0
        self._emit(
            f"Found {total_files} Excel files in {cond_folder}",
            offset + processed_files,
            overall_total,
        )

        roi_names = (
            list(self.roi_map.keys())
            if self.selected_roi == ALL_ROIS_OPTION
            else [self.selected_roi]
        )

        roi_data: Dict[str, List[List[float]]] = {rn: [] for rn in roi_names}
        freqs: Iterable[float] | None = None

        for excel_path in excel_files:
            if self._stop_requested:
                self._emit("Generation cancelled by user.")
                return [], {}
            self._emit(
                f"Reading {excel_path.name}",
                offset + processed_files,
                overall_total,
            )
            try:
                xls = pd.ExcelFile(excel_path)
                if "FullSNR" in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name="FullSNR")
                else:
                    df_amp = pd.read_excel(xls, sheet_name="FFT Amplitude (uV)")
                    freq_cols_tmp = [
                        c for c in df_amp.columns if isinstance(c, str) and c.endswith("_Hz")
                    ]
                    snr_vals = df_amp[freq_cols_tmp].apply(
                        calc_snr_matlab, axis=1, result_type="expand"
                    )
                    snr_vals.columns = freq_cols_tmp
                    snr_vals.insert(0, "Electrode", df_amp["Electrode"])
                    df = snr_vals
            except Exception as e:
                self._emit(f"Failed reading {excel_path.name}: {e}")
                processed_files += 1
                self._emit("", offset + processed_files, overall_total)
                continue

            freq_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("_Hz")]
            if not freq_cols:
                self._emit(
                    f"No freq columns in {excel_path.name}",
                    offset + processed_files,
                    overall_total,
                )
                processed_files += 1
                self._emit("", offset + processed_files, overall_total)
                continue

            freq_pairs: List[tuple[float, str]] = []
            for col in freq_cols:
                try:
                    freq = float(col.split("_")[0])
                except ValueError:
                    continue
                freq_pairs.append((freq, col))

            freq_pairs.sort(key=lambda x: x[0])
            ordered_freqs = [f for f, _ in freq_pairs]
            ordered_cols = [c for _, c in freq_pairs]
            if freqs is None:
                freqs = ordered_freqs

            for roi in roi_names:
                chans = [c.upper() for c in self.roi_map.get(roi, [])]
                try:
                    df_roi = df[df["Electrode"].str.upper().isin(chans)]
                except Exception:
                    self._emit(f"Missing 'Electrode' column in {excel_path.name}")
                    continue
                if df_roi.empty:
                    self._emit(f"No electrodes for ROI {roi} in {excel_path.name}")
                    continue

                means = df_roi[ordered_cols].mean().tolist()
                roi_data[roi].append(means)

            processed_files += 1
            self._emit("", offset + processed_files, overall_total)

        if not freqs:
            self._emit(
                "No frequency data found.",
                offset + processed_files,
                overall_total,
            )
            return [], {}

        averaged: Dict[str, List[float]] = {}
        for roi, rows in roi_data.items():
            if not rows:
                self._emit(f"No data collected for ROI {roi}")
                continue
            averaged[roi] = list(pd.DataFrame(rows).mean(axis=0))

        if not averaged:
            self._emit("No ROI data to plot.")
            return [], {}

        return list(freqs), averaged

    # ---------- main driver ----------

    def _run(self) -> None:
        # Mode 1: Condition overlay (A vs B)
        if self.overlay and self.condition_b:
            total_a = self._count_excel_files(self.condition)
            total_b = self._count_excel_files(self.condition_b)
            total = total_a + total_b
            freqs_a, data_a = self._collect_data(
                self.condition,
                offset=0,
                total_override=total,
            )
            freqs_b, data_b = self._collect_data(
                self.condition_b,
                offset=total_a,
                total_override=total,
            )
            if freqs_a and data_a and freqs_b and data_b:
                self._plot_overlay(freqs_a, data_a, data_b, legend_a=self.condition, legend_b=self.condition_b)
            return

        # Mode 2: Group comparison within a single condition
        if self.compare_groups:
            if not self.group_a or not self.group_b:
                self._emit("Group comparison requested but groups not selected.")
                return
            members_a = self._subjects_for_group(self.group_a)
            members_b = self._subjects_for_group(self.group_b)
            if not members_a or not members_b:
                self._emit(
                    f"Missing group membership in project.json for '{self.group_a}' or '{self.group_b}'."
                )
                return

            total_a = self._count_excel_files(self.condition, include_subjects=members_a)
            total_b = self._count_excel_files(self.condition, include_subjects=members_b)
            total = total_a + total_b

            freqs_a, data_a = self._collect_data(
                self.condition,
                offset=0,
                total_override=total,
                include_subjects=members_a,
            )
            freqs_b, data_b = self._collect_data(
                self.condition,
                offset=total_a,
                total_override=total,
                include_subjects=members_b,
            )
            if freqs_a and data_a and freqs_b and data_b:
                self._plot_overlay(
                    freqs_a,
                    data_a,
                    data_b,
                    legend_a=self.group_a,
                    legend_b=self.group_b,
                    group_mode=True,
                )
            return

        # Default: single condition plot
        freqs, averaged = self._collect_data(self.condition)
        if freqs and averaged:
            self._plot(freqs, averaged)

    # ---------- plotting ----------

    def _plot(self, freqs: List[float], roi_data: Dict[str, List[float]]) -> None:
        mgr = SettingsManager()
        harm_str = mgr.get(
            "loreta",
            "oddball_harmonics",
            "1.2,2.4,3.6,4.8,7.2,8.4,9.6,10.8",
        )
        try:
            cfg_odds = [float(h) for h in harm_str.replace(";", ",").split(",") if h.strip()]
        except Exception:
            cfg_odds = []
        odd_freqs = self.oddballs if self.oddballs else cfg_odds

        for roi, amps in roi_data.items():
            if self._stop_requested:
                self._emit("Generation cancelled by user.")
                return
            fig, ax = plt.subplots(figsize=(12, 4))

            stem_vals = amps
            cont = ax.stem(
                freqs,
                stem_vals,
                linefmt=self.stem_color,
                markerfmt=" ",
                basefmt=" ",
                bottom=1.0,
            )
            cont.markerline.set_label(self.metric)
            self._emit(f"Plotted {len(stem_vals)} SNR stems for ROI {roi}", 0, 0)

            if odd_freqs and not self.use_matlab_style:
                freq_array = np.array(freqs)
                for idx, odd in enumerate(odd_freqs):
                    closest = int(np.abs(freq_array - odd).argmin())
                    label = "Oddball Peaks" if idx == 0 else "_nolegend_"
                    ax.scatter(
                        freq_array[closest],
                        amps[closest],
                        facecolor="red",
                        edgecolor="black",
                        zorder=4,
                        label=label,
                    )
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
                self._emit(f"Marked {len(odd_freqs)} oddball points on ROI {roi}", 0, 0)

            tick_start = math.ceil(self.x_min)
            tick_end = math.floor(self.x_max) + 1
            ax.set_xticks(range(tick_start, tick_end))
            ax.set_xlim(self.x_min, self.x_max)
            ax.set_ylim(self.y_min, self.y_max)
            for fx in range(max(1, tick_start), tick_end):
                ax.axvline(fx, color="lightgray", linestyle="--", linewidth=0.5, zorder=0)
            for y in range(math.ceil(self.y_min), math.floor(self.y_max) + 1):
                ax.axhline(y, color="lightgray", linestyle="--", linewidth=0.5, zorder=0)
            if not self.use_matlab_style:
                ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)

            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)
            ax.grid(axis="y", linestyle=":", linewidth=0.8, color="gray")

            combined_title = f"{self.title}: {roi}"
            fig.suptitle(combined_title, fontsize=16, ha="center", va="top")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fname = f"{self.condition}_{roi}_{self.metric}.png"
            fig.savefig(self.out_dir / fname, dpi=300, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            self._emit(f"Saved {fname}")

    def _plot_overlay(
        self,
        freqs: List[float],
        data_a: Dict[str, List[float]],
        data_b: Dict[str, List[float]],
        *,
        legend_a: str,
        legend_b: str,
        group_mode: bool = False,
    ) -> None:
        plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
        mgr = SettingsManager()
        harm_str = mgr.get(
            "loreta",
            "oddball_harmonics",
            "1.2,2.4,3.6,4.8,7.2,8.4,9.6,10.8",
        )
        try:
            cfg_odds = [float(h) for h in harm_str.replace(";", ",").split(",") if h.strip()]
        except Exception:
            cfg_odds = []
        odd_freqs = self.oddballs if self.oddballs else cfg_odds

        for roi in data_a:
            if self._stop_requested:
                self._emit("Generation cancelled by user.")
                return
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(freqs, data_a[roi], label=legend_a, color=self.stem_color)
            ax.plot(freqs, data_b.get(roi, []), label=legend_b, color=self.stem_color_b)

            if odd_freqs:
                freq_array = np.array(freqs)
                for idx, odd in enumerate(odd_freqs):
                    closest = int(np.abs(freq_array - odd).argmin())
                    val_a = data_a[roi][closest]
                    val_b = data_b[roi][closest]
                    label_a = f"{legend_a} Peaks" if idx == 0 else "_nolegend_"
                    label_b = f"{legend_b} Peaks" if idx == 0 else "_nolegend_"
                    ax.scatter(
                        freq_array[closest],
                        val_a,
                        marker="o",
                        facecolor=self.stem_color,
                        edgecolor="black",
                        zorder=4,
                        label=label_a,
                    )
                    ax.scatter(
                        freq_array[closest],
                        val_b,
                        marker="^",
                        facecolor=self.stem_color_b,
                        edgecolor="black",
                        zorder=4,
                        label=label_b,
                    )

            tick_start = math.ceil(self.x_min)
            tick_end = math.floor(self.x_max) + 1
            ax.set_xticks(range(tick_start, tick_end))
            ax.set_xlim(self.x_min, self.x_max)
            ax.set_ylim(self.y_min, self.y_max)
            for fx in range(max(1, tick_start), tick_end):
                ax.axvline(fx, color="lightgray", linestyle="--", linewidth=0.5, zorder=0)
            for y in range(math.ceil(self.y_min), math.floor(self.y_max) + 1):
                ax.axhline(y, color="lightgray", linestyle="--", linewidth=0.5, zorder=0)
            if not self.use_matlab_style:
                ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)

            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)
            base = self.title or (
                f"{self.condition} {legend_a} vs {legend_b}" if group_mode
                else f"{self.condition} vs {self.condition_b}"
            )
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
            ax.grid(axis="y", linestyle=":", linewidth=0.8, color="gray")

            combined_title = f"{base}: {roi}"
            fig.suptitle(combined_title, fontsize=16, ha="center", va="top")
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            if group_mode:
                fname = f"{self.condition}_{legend_a}_vs_{legend_b}_{roi}_{self.metric}.png"
            else:
                fname = f"{self.condition}_vs_{legend_b}_{roi}_{self.metric}.png"
            fig.savefig(self.out_dir / fname, dpi=300, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            self._emit(f"Saved {fname}")
