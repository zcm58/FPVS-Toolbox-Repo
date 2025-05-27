# Tools/Stats/stats_qc_outliers.py
# -*- coding: utf-8 -*-
"""
Provides a CTkFrame class for reviewing and managing participants
flagged for data quality issues (e.g., high channel rejection)
during preprocessing.
"""

import os
import customtkinter as ctk
import tkinter as tk
import pandas as pd
import traceback

# Import constants from the main project config file
try:
    from config import QUALITY_FLAGS_FILENAME
except ImportError:
    print("Warning: Could not import QUALITY_FLAGS_FILENAME from main config.py for stats_qc_outliers. Using default.")
    QUALITY_FLAGS_FILENAME = "Potential_Outlier_Participants.txt" # Fallback


class QualityOutlierReviewFrame(ctk.CTkFrame):
    def __init__(self, master, app_log_func, stats_data_folder_var, **kwargs):
        """
        Frame for displaying and managing participants flagged for data quality.

        Args:
            master: The parent widget (usually a frame in StatsAnalysisWindow).
            app_log_func (callable): Function from the main StatsAnalysisWindow to log messages.
            stats_data_folder_var (tk.StringVar): StringVar from StatsAnalysisWindow holding the
                                                 path to the current data folder.
        """
        super().__init__(master, **kwargs)
        self.app_log_func = app_log_func
        self.stats_data_folder_var = stats_data_folder_var

        # Stores dictionaries like:
        # {'pid': str, 'filename': str, 'bad_channels': int, 'threshold': int,
        #  'exclude_var': tk.BooleanVar}
        self.quality_flagged_participants_info = []

        self.grid_columnconfigure(0, weight=1)  # Allow content to expand

        # --- Header for this section ---
        self.header_label = ctk.CTkLabel(self, text="Review Preprocessing Quality Flags",
                                         font=ctk.CTkFont(weight="bold"))
        self.header_label.grid(row=0, column=0, sticky="w", padx=10, pady=(5, 2))

        self.info_label = ctk.CTkLabel(self,
                                       text="Participants listed below were flagged for a high number of "
                                            "rejected channels during initial processing in the main FPVS app.",
                                       wraplength=self.winfo_width() - 20,  # Adjust wraplength dynamically or set fixed
                                       justify="left")
        self.info_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))
        self.bind("<Configure>", lambda event: self.info_label.configure(wraplength=self.winfo_width() - 20))

        # --- Scrollable Frame for listing flagged participants ---
        self.scrollable_frame = ctk.CTkScrollableFrame(self, label_text="Flagged Participants (High Channel Rejection)")
        self.scrollable_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.scrollable_frame.configure(height=100)  # Default height, adjust as needed
        self.scrollable_frame.grid_columnconfigure(0, weight=1)  # Allow content within to expand if needed

        self.no_flags_label = ctk.CTkLabel(self.scrollable_frame,
                                           text="No quality flags file found or folder not yet scanned.\n"
                                                f"(Looking for '{QUALITY_FLAGS_FILENAME}')")
        self.no_flags_label.pack(pady=10, padx=10, fill="x", expand=True)

        # --- Select/Deselect All Buttons ---
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 5))

        self.select_all_button = ctk.CTkButton(button_frame, text="Mark All as Excluded",
                                               command=lambda: self._toggle_all_exclusions(True))
        self.select_all_button.pack(side="left", padx=(0, 5))

        self.deselect_all_button = ctk.CTkButton(button_frame, text="Mark All to Include",
                                                 command=lambda: self._toggle_all_exclusions(False))
        self.deselect_all_button.pack(side="left", padx=5)

        self._update_button_states()  # Initial state

    def _update_button_states(self):
        """Enables or disables select/deselect all buttons based on content."""
        if self.quality_flagged_participants_info:
            self.select_all_button.configure(state="normal")
            self.deselect_all_button.configure(state="normal")
        else:
            self.select_all_button.configure(state="disabled")
            self.deselect_all_button.configure(state="disabled")

    def _clear_display(self):
        """Clears previously displayed flagged participants."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.quality_flagged_participants_info = []  # Reset the internal list

        # Re-add the default "no flags" label, it will be destroyed if flags are found later
        self.no_flags_label = ctk.CTkLabel(self.scrollable_frame,
                                           text="No quality flags file found or folder not yet scanned.\n"
                                                f"(Looking for '{QUALITY_FLAGS_FILENAME}')")
        self.no_flags_label.pack(pady=10, padx=10, fill="x", expand=True)
        self._update_button_states()

    def load_and_display_flags(self):
        """
        Loads data from the QUALITY_FLAGS_FILENAME, parses it,
        and updates the UI to display the flagged participants with checkboxes.
        """
        self.app_log_func(f"QualityOutlierFrame: Attempting to load '{QUALITY_FLAGS_FILENAME}'...")
        self._clear_display()

        folder_path = self.stats_data_folder_var.get()
        if not folder_path or not os.path.isdir(folder_path):
            self.app_log_func("QualityOutlierFrame: Data folder not set or invalid for loading flags.")
            self.no_flags_label.configure(text="Select a valid data folder first to check for quality flags.")
            return

        quality_file_path = os.path.join(folder_path, QUALITY_FLAGS_FILENAME)

        if os.path.exists(quality_file_path):
            self.app_log_func(f"QualityOutlierFrame: Found quality flags file: {quality_file_path}")
            try:
                df_flags = pd.read_csv(quality_file_path)

                if df_flags.empty:
                    self.app_log_func("QualityOutlierFrame: Quality flags file is empty.")
                    self.no_flags_label.configure(text=f"'{QUALITY_FLAGS_FILENAME}' was found but is empty.")
                    return

                if self.no_flags_label.winfo_exists():  # If flags are found, remove the placeholder
                    self.no_flags_label.destroy()

                for index, row in df_flags.iterrows():
                    pid = str(row.get('PID', 'UnknownPID'))
                    filename = str(row.get('OriginalFilename', 'N/A'))
                    bad_channels = int(row.get('NumBadChannelsIdentified', 0))
                    threshold = int(row.get('UserSetThreshold', 0))

                    exclude_var = tk.BooleanVar(master=self, value=False)  # Default to NOT exclude
                    info = {
                        'pid': pid,
                        'filename': filename,
                        'bad_channels': bad_channels,
                        'threshold': threshold,
                        'exclude_var': exclude_var
                    }
                    self.quality_flagged_participants_info.append(info)

                    entry_text = f"Exclude {pid}? (File: {filename}, Bads: {bad_channels}, Thresh: {threshold})"
                    chk = ctk.CTkCheckBox(self.scrollable_frame, text=entry_text, variable=exclude_var)
                    chk.pack(anchor="w", padx=5, pady=2, fill="x")

                if not self.quality_flagged_participants_info:
                    self.no_flags_label = ctk.CTkLabel(self.scrollable_frame,
                                                       text="No subjects met quality flag criteria in the file.")
                    self.no_flags_label.pack(pady=10, padx=10, fill="x", expand=True)

                self._update_button_states()

            except FileNotFoundError:
                self.app_log_func(
                    f"QualityOutlierFrame: File '{QUALITY_FLAGS_FILENAME}' disappeared (race condition?).")
                self.no_flags_label.configure(text=f"File '{QUALITY_FLAGS_FILENAME}' seems to have disappeared.")
            except pd.errors.EmptyDataError:
                self.app_log_func(f"QualityOutlierFrame: File '{QUALITY_FLAGS_FILENAME}' is empty or not valid CSV.")
                self.no_flags_label.configure(text=f"File '{QUALITY_FLAGS_FILENAME}' is empty or unreadable.")
            except KeyError as ke:
                self.app_log_func(f"QualityOutlierFrame: Missing expected column in '{QUALITY_FLAGS_FILENAME}': {ke}")
                self.no_flags_label.configure(
                    text=f"Flags file ('{QUALITY_FLAGS_FILENAME}') has incorrect columns (Missing: {ke}).")
            except Exception as e:
                self.app_log_func(
                    f"QualityOutlierFrame: Error parsing quality flags file '{quality_file_path}': {e}\n{traceback.format_exc()}")
                self.no_flags_label.configure(text=f"Error reading '{QUALITY_FLAGS_FILENAME}'. See console log.")
        else:
            self.app_log_func(
                f"QualityOutlierFrame: No quality flags file ('{QUALITY_FLAGS_FILENAME}') found in {folder_path}.")
            self.no_flags_label.configure(
                text=f"No quality flags file ('{QUALITY_FLAGS_FILENAME}') found in the selected folder.")

        self._update_button_states()

    def _toggle_all_exclusions(self, exclude_state: bool):
        """Sets the exclusion state for all currently listed quality-flagged PIDs."""
        if not self.quality_flagged_participants_info:
            self.app_log_func("QualityOutlierFrame: No flagged participants to select/deselect.")
            return

        action = "Excluding all" if exclude_state else "Including all (clearing exclusions for)"
        self.app_log_func(f"QualityOutlierFrame: {action} listed quality-flagged participants.")
        for info in self.quality_flagged_participants_info:
            info['exclude_var'].set(exclude_state)

    def get_pids_to_exclude(self) -> set:
        """
        Returns a set of PIDs that the user has checked for exclusion
        based on the quality flags.
        """
        excluded_pids = set()
        for info in self.quality_flagged_participants_info:
            if info['exclude_var'].get():
                excluded_pids.add(info['pid'])

        # Log only if there are actual exclusions to report
        # if excluded_pids:
        #     self.app_log_func(f"QualityOutlierFrame: Returning PIDs for exclusion based on quality flags: {sorted(list(excluded_pids))}")
        return excluded_pids