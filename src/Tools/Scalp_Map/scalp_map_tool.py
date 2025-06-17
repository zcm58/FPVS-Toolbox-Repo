# scalp_map_tool.py
"""Generate average BCA scalp maps from FPVS Excel results."""

import os
import threading
from pathlib import Path

import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import customtkinter as ctk
from tkinter import filedialog, messagebox


class BcaScalpMapWindow(ctk.CTkToplevel):
    """Toplevel window for creating scalp map heatmaps."""

    def __init__(self, master: ctk.CTkBaseClass):
        super().__init__(master)
        self.transient(master)
        self.title("BCA Scalp Maps")
        self.geometry("500x400")
        self.lift()
        self.attributes('-topmost', True)
        self.after(0, lambda: self.attributes('-topmost', False))
        self.focus_force()

        self.input_folder = ""
        self.output_folder = ""

        self._build_ui()

    def _build_ui(self) -> None:
        frame = ctk.CTkFrame(self, corner_radius=8)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkButton(frame, text="Select Input Folder", command=self._select_input).pack(fill="x", pady=5)
        self.in_label = ctk.CTkLabel(frame, text="not selected", anchor="w")
        self.in_label.pack(fill="x")

        ctk.CTkButton(frame, text="Select Output Folder", command=self._select_output).pack(fill="x", pady=5)
        self.out_label = ctk.CTkLabel(frame, text="not selected", anchor="w")
        self.out_label.pack(fill="x")

        self.start_btn = ctk.CTkButton(frame, text="Generate Scalp Maps", command=self._start)
        self.start_btn.pack(pady=10)

        self.log_box = ctk.CTkTextbox(frame, height=150, state="disabled")
        self.log_box.pack(fill="both", expand=True, pady=5)

    # --- UI callbacks ---
    def _log(self, msg: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _select_input(self) -> None:
        folder = filedialog.askdirectory(title="Select Parent Folder")
        if folder:
            self.input_folder = folder
            self.in_label.configure(text=os.path.basename(folder))

    def _select_output(self) -> None:
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.out_label.configure(text=os.path.basename(folder))

    def _start(self) -> None:
        if not self.input_folder or not self.output_folder:
            messagebox.showerror("Error", "Please select both input and output folders.")
            return
        self.start_btn.configure(state="disabled")
        threading.Thread(target=self._run, daemon=True).start()

    # --- Processing ---
    def _run(self) -> None:
        self._log("Processing started...")
        try:
            self._process_all_conditions()
        except Exception as e:  # pragma: no cover - GUI thread
            self._log(f"Error: {e}")
        self._log("Done.")
        self.start_btn.configure(state="normal")

    def _process_all_conditions(self) -> None:
        parent = Path(self.input_folder)
        for item in sorted(parent.iterdir()):
            if not item.is_dir():
                continue
            condition_name = item.name
            self._log(f"Condition: {condition_name}")
            excel_files = list(item.glob("*.xlsx"))
            if not excel_files:
                self._log("  No Excel files found.")
                continue
            dfs = []
            for f in excel_files:
                try:
                    df = pd.read_excel(f, sheet_name="BCA (uV)", index_col="Electrode")
                    dfs.append(df)
                except Exception as exc:
                    self._log(f"  Failed reading {f.name}: {exc}")
            if not dfs:
                self._log("  No readable files for this condition.")
                continue
            mean_df = self._average_dataframes(dfs)
            out_dir = Path(self.output_folder) / condition_name
            out_dir.mkdir(parents=True, exist_ok=True)
            self._plot_condition_maps(mean_df, condition_name, out_dir)
            self._log(f"  Saved maps to {out_dir}")

    @staticmethod
    def _average_dataframes(dfs: list[pd.DataFrame]) -> pd.DataFrame:
        electrodes = sorted({e for df in dfs for e in df.index})
        freqs = sorted({c for df in dfs for c in df.columns})
        arrs = []
        for df in dfs:
            aligned = df.reindex(index=electrodes, columns=freqs)
            arrs.append(aligned.to_numpy(dtype=float))
        data = np.stack(arrs)
        with np.errstate(invalid='ignore'):
            avg = np.nanmean(data, axis=0)
        return pd.DataFrame(avg, index=electrodes, columns=freqs)

    def _plot_condition_maps(self, df: pd.DataFrame, cond: str, out_dir: Path) -> None:
        montage = mne.channels.make_standard_montage('standard_1020')
        info = mne.create_info(ch_names=df.index.tolist(), sfreq=1, ch_types='eeg')
        info.set_montage(montage)
        cmap = LinearSegmentedColormap.from_list('bca', ['darkblue', 'red'])
        vmax = float(np.nanmax(df.values)) if not df.empty else 1.0
        for freq in df.columns:
            data = df[freq].to_numpy(dtype=float)
            fig, ax = plt.subplots()
            im, _ = mne.viz.plot_topomap(data, info, axes=ax, show=False,
                                          vmin=0, vmax=vmax, cmap=cmap)
            ax.set_title(f"{cond} - {freq}")
            fig.colorbar(im, ax=ax, shrink=0.7)
            fname = f"{freq}_scalpmap.png".replace('/', '_')
            plt.savefig(out_dir / fname, bbox_inches='tight')
            plt.close(fig)


