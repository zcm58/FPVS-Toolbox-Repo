import os
import glob
import re
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt

from config import init_fonts, DEFAULT_ELECTRODE_NAMES_64
from Main_App.settings_manager import SettingsManager


def _load_base_freq(master):
    try:
        if hasattr(master, "settings"):
            return float(master.settings.get("analysis", "base_freq", "6.0"))
    except Exception:
        pass
    return float(SettingsManager().get("analysis", "base_freq", "6.0"))


def _get_included_freqs(columns, base_freq):
    freqs = []
    for col in columns:
        if col.endswith("_Hz"):
            try:
                val = float(col[:-3])
                if abs(val / base_freq - round(val / base_freq)) > 1e-6:
                    freqs.append((val, col))
            except ValueError:
                continue
    return freqs


def _compute_avg_bca(file_paths):
    df_sum = None
    count = 0
    for f in file_paths:
        try:
            df = pd.read_excel(f, sheet_name="BCA (uV)", index_col="Electrode")
            df = df.reindex(DEFAULT_ELECTRODE_NAMES_64)
            if df_sum is None:
                df_sum = df
            else:
                df_sum = df_sum.add(df, fill_value=0)
            count += 1
        except Exception:
            continue
    if df_sum is None or count == 0:
        return None
    return df_sum / count


class BCAHeatmapTool(ctk.CTkToplevel):
    def __init__(self, master, default_folder=""):
        super().__init__(master)
        self.transient(master)
        init_fonts()
        self.title("BCA Heatmap Generator")
        self.geometry("800x600")
        self.lift()
        self.attributes('-topmost', True)
        self.after(0, lambda: self.attributes('-topmost', False))
        self.focus_force()

        self.master_app = master

        self.data_folder_var = tk.StringVar(master=self, value=default_folder)
        self.output_folder_var = tk.StringVar(master=self, value=default_folder)
        self.detected_info_var = tk.StringVar(master=self, value="Select folder containing results")
        self.condition_var = tk.StringVar(master=self)
        self.max_val_var = tk.StringVar(master=self)

        self.subject_files = {}
        self.conditions = []
        self.base_freq = _load_base_freq(master)

        self._build_ui()
        if default_folder and os.path.isdir(default_folder):
            self.scan_folder()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # --- UI helpers ---
    def _build_ui(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        main_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(main_frame, text="Results Folder:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ctk.CTkEntry(main_frame, textvariable=self.data_folder_var, state="readonly").grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkButton(main_frame, text="Browse", command=self.browse_data_folder).grid(row=0, column=2, padx=5, pady=5)

        ctk.CTkLabel(main_frame, textvariable=self.detected_info_var).grid(row=1, column=0, columnspan=3, sticky="w", padx=5)

        ctk.CTkLabel(main_frame, text="Condition:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.cond_menu = ctk.CTkOptionMenu(main_frame, variable=self.condition_var, values=self.conditions)
        self.cond_menu.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

        ctk.CTkLabel(main_frame, text="Output Folder:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        ctk.CTkEntry(main_frame, textvariable=self.output_folder_var, state="readonly").grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkButton(main_frame, text="Browse", command=self.browse_output_folder).grid(row=3, column=2, padx=5, pady=5)

        ctk.CTkLabel(main_frame, text="Color Max (red):").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        ctk.CTkEntry(main_frame, textvariable=self.max_val_var).grid(row=4, column=1, sticky="w", padx=5, pady=5)

        ctk.CTkButton(main_frame, text="Generate Heatmaps", command=self.generate_heatmaps).grid(row=5, column=0, columnspan=3, pady=10)

    def log_to_main_app(self, message):
        try:
            if hasattr(self.master_app, "log") and callable(self.master_app.log):
                self.master_app.log(f"[Heatmaps] {message}")
        except Exception:
            pass

    # --- Folder methods ---
    def browse_data_folder(self):
        folder = filedialog.askdirectory(title="Select Parent Folder")
        if folder:
            self.data_folder_var.set(folder)
            self.scan_folder()

    def browse_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder_var.set(folder)

    def scan_folder(self):
        parent = self.data_folder_var.get()
        if not parent or not os.path.isdir(parent):
            self.detected_info_var.set("Invalid folder")
            self.condition_var.set("")
            self.conditions = []
            self.cond_menu.configure(values=self.conditions)
            return
        self.log_to_main_app(f"Scanning {parent}")
        pid_pattern = re.compile(r"(?:[a-zA-Z]*)?(P\d+).*\.xlsx$", re.IGNORECASE)
        self.subject_files.clear()
        conditions = set()
        for name in os.listdir(parent):
            sub = os.path.join(parent, name)
            if os.path.isdir(sub):
                cond = re.sub(r'^\d+\s*[-_]*\s*', '', name).strip()
                files = []
                for f in glob.glob(os.path.join(sub, "*.xlsx")):
                    if pid_pattern.search(os.path.basename(f)):
                        files.append(f)
                if files:
                    conditions.add(cond)
                    self.subject_files[cond] = files
        self.conditions = sorted(conditions)
        self.cond_menu.configure(values=self.conditions)
        if self.conditions:
            self.condition_var.set(self.conditions[0])
            info = f"Found {len(self.conditions)} conditions"
        else:
            self.condition_var.set("")
            info = "No conditions found"
        self.detected_info_var.set(info)

    # --- Heatmap generation ---
    def generate_heatmaps(self):
        cond = self.condition_var.get()
        files = self.subject_files.get(cond)
        if not files:
            messagebox.showerror("Error", "No files for selected condition")
            return
        avg_df = _compute_avg_bca(files)
        if avg_df is None:
            messagebox.showerror("Error", "Failed to load BCA data")
            return
        freqs = _get_included_freqs(avg_df.columns, self.base_freq)
        if not freqs:
            messagebox.showerror("Error", "No valid frequency columns found")
            return
        out_parent = self.output_folder_var.get() or self.data_folder_var.get()
        out_dir = os.path.join(out_parent, f"{cond}_heatmaps")
        os.makedirs(out_dir, exist_ok=True)
        try:
            color_max = float(self.max_val_var.get()) if self.max_val_var.get() else float(avg_df.max().max())
        except ValueError:
            color_max = float(avg_df.max().max())
        montage = mne.channels.make_standard_montage('standard_1020')
        info = mne.create_info(ch_names=DEFAULT_ELECTRODE_NAMES_64, sfreq=250, ch_types='eeg')
        info.set_montage(montage)
        for val, col in freqs:
            data = avg_df[col].reindex(DEFAULT_ELECTRODE_NAMES_64).to_numpy()
            fig, ax = plt.subplots()
            mne.viz.plot_topomap(data, info, axes=ax, show=False, cmap='RdBu_r', vmin=min(data.min(), 0), vmax=color_max)
            ax.set_title(f"{val:.1f} Hz")
            fig.savefig(os.path.join(out_dir, f"{val:.1f}Hz_topomap.png"), bbox_inches='tight')
            plt.close(fig)
        messagebox.showinfo("Done", f"Heatmaps saved to {out_dir}")
        self.log_to_main_app("Heatmap generation complete")

    def on_close(self):
        self.log_to_main_app("Closing Heatmap window")
        self.destroy()
