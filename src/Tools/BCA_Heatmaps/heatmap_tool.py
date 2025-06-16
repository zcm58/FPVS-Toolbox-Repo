import os
import glob
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Dict

import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mne

from config import (
    DEFAULT_ELECTRODE_NAMES_64,
    PAD_X,
    PAD_Y,
    CORNER_RADIUS,
)
from Main_App.settings_manager import SettingsManager


class BCAHeatmapTool(ctk.CTkToplevel):
    """Standalone window to generate scalp heatmaps from BCA data."""

    def __init__(self, master: ctk.CTkBaseClass):
        super().__init__(master)
        self.transient(master)
        self.title("BCA Heatmap Generator")
        self.geometry("700x600")
        self.lift()
        self.attributes("-topmost", True)
        self.after(0, lambda: self.attributes("-topmost", False))
        self.focus_force()

        self.master_app = master
        self.settings = getattr(master, "settings", SettingsManager())

        self.stats_data_folder_var = tk.StringVar(value="")
        self.output_folder_var = tk.StringVar(value="")
        self.detected_info_var = tk.StringVar(value="Select folder containing FPVS results.")
        self.ALL_OPTION = "(All Conditions)"
        self.condition_var = tk.StringVar(value=self.ALL_OPTION)
        self.vmax_var = tk.StringVar(value="")

        self.subjects: List[str] = []
        self.conditions: List[str] = []
        self.subject_data: Dict[str, Dict[str, str]] = {}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    # -------------------------- UI Helpers --------------------------
    def _build_ui(self):
        self.grid_rowconfigure(4, weight=1)
        self.grid_columnconfigure(0, weight=1)

        folder_frame = ctk.CTkFrame(self, corner_radius=CORNER_RADIUS)
        folder_frame.grid(row=0, column=0, sticky="ew", padx=PAD_X, pady=PAD_Y)
        folder_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(folder_frame, text="Browse Results Folder", command=self.browse_folder).grid(
            row=0, column=0, padx=PAD_X, pady=PAD_Y
        )
        ctk.CTkEntry(folder_frame, textvariable=self.stats_data_folder_var, state="readonly").grid(
            row=0, column=1, padx=PAD_X, pady=PAD_Y, sticky="ew"
        )
        ctk.CTkLabel(folder_frame, textvariable=self.detected_info_var, anchor="w").grid(
            row=1, column=0, columnspan=2, sticky="w", padx=PAD_X, pady=PAD_Y
        )

        output_frame = ctk.CTkFrame(self, corner_radius=CORNER_RADIUS)
        output_frame.grid(row=1, column=0, sticky="ew", padx=PAD_X, pady=PAD_Y)
        output_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(output_frame, text="Browse Output Folder", command=self.browse_output_folder).grid(
            row=0, column=0, padx=PAD_X, pady=PAD_Y
        )
        ctk.CTkEntry(output_frame, textvariable=self.output_folder_var, state="readonly").grid(
            row=0, column=1, padx=PAD_X, pady=PAD_Y, sticky="ew"
        )

        options_frame = ctk.CTkFrame(self, corner_radius=CORNER_RADIUS)
        options_frame.grid(row=2, column=0, sticky="nsew", padx=PAD_X, pady=PAD_Y)
        options_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(options_frame, text="Condition:").grid(row=0, column=0, padx=PAD_X, pady=PAD_Y)
        self.cond_menu = ctk.CTkOptionMenu(options_frame, variable=self.condition_var, values=[self.ALL_OPTION])
        self.cond_menu.grid(row=0, column=1, sticky="ew", padx=PAD_X, pady=PAD_Y)

        ctk.CTkLabel(options_frame, text="Color Limit (max μV):").grid(row=1, column=0, padx=PAD_X, pady=PAD_Y)
        ctk.CTkEntry(options_frame, textvariable=self.vmax_var).grid(row=1, column=1, sticky="ew", padx=PAD_X, pady=PAD_Y)

        action_frame = ctk.CTkFrame(self, fg_color="transparent")
        action_frame.grid(row=3, column=0, sticky="ew", padx=PAD_X, pady=PAD_Y)
        ctk.CTkButton(action_frame, text="Generate Heatmaps", command=self.generate_heatmaps).pack(side="left", padx=PAD_X)

        self.status_box = ctk.CTkTextbox(self, height=200, state="disabled")
        self.status_box.grid(row=4, column=0, sticky="nsew", padx=PAD_X, pady=PAD_Y)

    def log(self, message: str):
        self.status_box.configure(state="normal")
        self.status_box.insert("end", message + "\n")
        self.status_box.configure(state="disabled")
        self.status_box.see("end")

    # -------------------------- Folder Scanning --------------------------
    def browse_folder(self):
        current = self.stats_data_folder_var.get()
        initial_dir = current if os.path.isdir(current) else os.path.expanduser("~")
        folder = filedialog.askdirectory(title="Select Parent Folder Containing Condition Subfolders", initialdir=initial_dir)
        if folder:
            self.stats_data_folder_var.set(folder)
            self.scan_folder()
        else:
            self.log("Folder selection cancelled.")

    def browse_output_folder(self):
        current = self.output_folder_var.get()
        initial_dir = current if os.path.isdir(current) else os.path.expanduser("~")
        folder = filedialog.askdirectory(title="Select Output Folder", initialdir=initial_dir)
        if folder:
            self.output_folder_var.set(folder)
        else:
            self.log("Output folder selection cancelled.")

    def scan_folder(self):
        parent_folder = self.stats_data_folder_var.get()
        if not parent_folder or not os.path.isdir(parent_folder):
            self.detected_info_var.set("Invalid parent folder selected.")
            self.update_condition_menu([])
            return

        self.log(f"Scanning parent folder: {parent_folder}")
        subjects_set = set()
        conditions_set = set()
        self.subject_data.clear()

        pid_pattern = re.compile(r"(?:[a-zA-Z]*)?(P\d+).*\.xlsx$", re.IGNORECASE)
        try:
            for item_name in os.listdir(parent_folder):
                item_path = os.path.join(parent_folder, item_name)
                if os.path.isdir(item_path):
                    condition_name_raw = item_name
                    condition_name = re.sub(r'^\d+\s*[-_]*\s*', '', condition_name_raw).strip()
                    if not condition_name:
                        self.log(f"Skipping subfolder '{condition_name_raw}' due to empty name after cleaning.")
                        continue
                    self.log(f"Processing Condition Subfolder: '{condition_name_raw}' as '{condition_name}'")

                    files_in_subfolder = glob.glob(os.path.join(item_path, "*.xlsx"))
                    found = False
                    for f_path in files_in_subfolder:
                        excel_filename = os.path.basename(f_path)
                        pid_match = pid_pattern.search(excel_filename)
                        if pid_match:
                            pid = pid_match.group(1).upper()
                            subjects_set.add(pid)
                            conditions_set.add(condition_name)
                            found = True
                            if pid not in self.subject_data:
                                self.subject_data[pid] = {}
                            self.subject_data[pid][condition_name] = f_path
                            self.log(f"  Found PID: {pid} -> {excel_filename}")
                    if not found:
                        self.log(f"  Warning: No Excel files found for condition '{condition_name}'.")
        except Exception as e:
            self.log(f"Error scanning folder: {e}")
            messagebox.showerror("Scanning Error", str(e))
            self.update_condition_menu([])
            return

        self.subjects = sorted(list(subjects_set))
        self.conditions = sorted(list(conditions_set))

        if not self.conditions or not self.subjects:
            info_text = "Scan complete: No valid condition subfolders or subject Excel files found."
        else:
            info_text = f"Scan complete: Found {len(self.subjects)} subjects and {len(self.conditions)} conditions."
        self.log(info_text)
        self.detected_info_var.set(info_text)
        self.update_condition_menu(self.conditions)

    def update_condition_menu(self, options: List[str]):
        if not options:
            display = [self.ALL_OPTION]
        else:
            display = [self.ALL_OPTION] + sorted(options)
        current = self.condition_var.get()
        if current not in display:
            self.condition_var.set(self.ALL_OPTION)
        self.cond_menu.configure(values=display)

    # -------------------------- Heatmap Generation --------------------------
    def _load_base_freq(self) -> float:
        return float(self.settings.get('analysis', 'base_freq', '6.0'))

    def generate_heatmaps(self):
        parent_folder = self.stats_data_folder_var.get()
        output_folder = self.output_folder_var.get()
        selected = self.condition_var.get()
        if not parent_folder or not os.path.isdir(parent_folder):
            messagebox.showerror("Error", "Please select a valid results folder first.")
            return

        if not output_folder or not os.path.isdir(output_folder):
            messagebox.showerror("Error", "Please select a valid output folder first.")
            return

        if not self.conditions:
            messagebox.showerror("Error", "No conditions detected in selected folder.")
            return

        if selected == self.ALL_OPTION:
            conds = self.conditions
        elif selected in self.conditions:
            conds = [selected]
        else:
            messagebox.showerror("Error", "Please select a valid condition.")
            return

        for cond in conds:
            self._generate_for_condition(parent_folder, output_folder, cond)

        messagebox.showinfo("Complete", "Heatmap generation complete.")

    def _generate_for_condition(self, parent_folder: str, output_folder: str, condition: str):

        file_paths = [self.subject_data[pid][condition] for pid in self.subjects if condition in self.subject_data.get(pid, {})]
        if not file_paths:
            messagebox.showerror("Error", f"No Excel files found for condition '{condition}'.")
            return

        data_arrays = []
        freq_names = None
        for path in file_paths:
            try:
                df = pd.read_excel(path, sheet_name="BCA (uV)", index_col="Electrode")
            except Exception as e:
                self.log(f"Error reading {os.path.basename(path)}: {e}")
                continue
            df = df.reindex(DEFAULT_ELECTRODE_NAMES_64)
            if freq_names is None:
                freq_names = df.columns.tolist()
            data_arrays.append(df.values.astype(float))

        if not data_arrays:
            messagebox.showerror("Error", "No valid BCA data read.")
            return

        avg_data = np.nanmean(np.stack(data_arrays), axis=0)  # channels x freqs

        base_freq = self._load_base_freq()
        output_dir = os.path.join(output_folder, condition, "heatmaps")
        os.makedirs(output_dir, exist_ok=True)

        vmax_in = self.vmax_var.get().strip()
        vmax = None
        if vmax_in:
            try:
                vmax = float(vmax_in)
            except ValueError:
                self.log(f"Invalid color limit: {vmax_in}. Using data max.")
                vmax = None
        if vmax is None:
            vmax = float(np.nanmax(avg_data))
        min_val = float(np.nanmin(avg_data))
        if min_val > 0:
            min_val = 0.0
        cmap = LinearSegmentedColormap.from_list("b2r", ["darkblue", "blue", "red"])

        info = mne.create_info(ch_names=DEFAULT_ELECTRODE_NAMES_64, sfreq=256, ch_types="eeg")
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage)

        for idx, col in enumerate(freq_names):
            try:
                freq_val = float(col.replace("_Hz", ""))
            except ValueError:
                continue
            if abs(freq_val - base_freq) < 1e-6:
                continue
            values = avg_data[:, idx]
            fig, ax = plt.subplots()
            mne.viz.plot_topomap(values, info, axes=ax, cmap=cmap, vmin=min_val, vmax=vmax, show=False)
            ax.set_title(f"{freq_val:.1f} Hz")
            fig.savefig(os.path.join(output_dir, f"{freq_val:.1f}Hz_topomap.png"))
            plt.close(fig)
            self.log(f"Saved {freq_val:.1f} Hz heatmap.")

        self.log(f"Heatmaps saved to: {output_dir}")

