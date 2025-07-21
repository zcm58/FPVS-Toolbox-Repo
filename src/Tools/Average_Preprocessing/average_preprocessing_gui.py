"""CustomTkinter based UI for the advanced averaging tool."""

from __future__ import annotations

import customtkinter as ctk
from tkinter import filedialog
import CTkMessagebox

from .advanced_analysis import AdvancedAnalysis


class AdvancedAnalysisWindow(ctk.CTkToplevel, AdvancedAnalysis):
    """Thin GUI wrapper that exposes the advanced analysis logic."""

    def __init__(self, master: ctk.CTkBaseClass) -> None:
        ctk.CTkToplevel.__init__(self, master)
        AdvancedAnalysis.__init__(self, log_callback=lambda msg: print(msg))
        self.title("Advanced Averaging Analysis")
        ctk.CTkButton(self, text="Add Files", command=self._select_files).pack(pady=10)

    def _select_files(self) -> None:
        files = filedialog.askopenfilenames(title="Select EEG Files", filetypes=[("EEG files", "*.bdf")])
        if files:
            self.add_source_files(list(files))
            CTkMessagebox.CTkMessagebox(title="Files Added", message=f"{len(files)} file(s) added.")

