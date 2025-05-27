# Tools/Stats/stats_qc_outliers.py
# -*- coding: utf-8 -*-
"""
Provides a CTkFrame class for reviewing and managing participants
flagged as potential outliers (e.g., high channel rejection)
during preprocessing.
"""

import os
import customtkinter as ctk
import tkinter as tk
import pandas as pd  # For easily reading the CSV-like text file

# Filename constant - ensure this matches what fpvs_app.py writes
QUALITY_FLAGS_FILENAME = "Potential_Outlier_Participants.txt"


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
        #  'exclude_var': tk.BooleanVar, 'checkbox_widget': ctk.CTkCheckBox (optional ref)}
        self.quality_flagged_participants_info = []

        self.grid_columnconfigure(0, weight=1)

        # --- Header for this section ---
        self.header_label = ctk.CTkLabel(self, text="Review Preprocessing Quality Flags",
                                         font=ctk.CTkFont(weight="bold"))
        self.header_label.grid(row=0, column=0, sticky="w", padx=10, pady=(5, 2))

        self.info_label = ctk.CTkLabel(self,
                                       text="Participants listed below were flagged for a high number of rejected channels during initial processing.",
                                       wraplength=500, justify="left")  # Adjust wraplength as needed
        self.info_label.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 5))

        # --- Scrollable Frame for listing flagged participants ---
        self.scrollable_frame = ctk.CTkScrollableFrame(self, label_text="Flagged Participants")
        self.scrollable_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.scrollable_frame.configure(height=120)  # Default height, can be adjusted
        # No need to configure grid inside if packing, but allow column 0 to expand
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        self.no_flags_label = ctk.CTkLabel(self.scrollable_frame,
                                           text="No quality flags file found or folder not yet scanned.\n"
                                                f"(Looking for '{QUALITY_FLAGS_FILENAME}')")
        self.no_flags_label.pack(pady=10, padx=10, fill="x", expand=True)

        # --- Select/Deselect All Buttons ---
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 5))

        self.select_all_button = ctk.CTkButton(button_frame, text="Exclude All Flagged",
                                               command=self._select_all_for_exclusion)
        self.select_all_button.pack(side="left", padx=(0, 5))

        self.deselect_all_button = ctk.CTkButton(button_frame, text="Include All Flagged",
                                                 command=self._deselect_all_for_exclusion)
        self.deselect_all_button.pack(side="left", padx=5)

        self.select_all_button.configure(state="disabled")
        self.deselect_all_button.configure(state="disabled")

    def _clear_display(self):
        """Clears previously displayed flagged participants."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        # Re-add the default "no flags" label, it will be destroyed if flags are found
        self.no_flags_label = ctk.CTkLabel(self.scrollable_frame,
                                           text="No quality flags file found or folder not yet scanned.\n"
                                                f"(Looking for '{QUALITY_FLAGS_FILENAME}')")
        self.no_flags_label.pack(pady=10, padx=10, fill="x", expand=True)
        self.quality_flagged_participants_info = []  # Reset the internal list
        self.select_all_button.configure(state="disabled")
        self.deselect_all_button.configure(state="disabled")

    def load_and_display_flags(self):
        """
        Loads data from the QUALITY_FLAGS_FILENAME, parses it,
        and updates the UI to display the flagged participants with checkboxes.
        """
        self.app_log_func("QualityOutlierFrame: Checking for quality flags file...")
        self._clear_display()  # Clear previous entries and reset list

        folder_path = self.stats_data_folder_var.get()
        if not folder_path or not os.path.isdir(folder_path):
            # The no_flags_label already indicates folder not scanned or invalid
            self.app_log_func("QualityOutlierFrame: Data folder not set or invalid.")
            return

        quality_file_path = os.path.join(folder_path, QUALITY_FLAGS_FILENAME)

        if os.path.exists(quality_file_path):
            self.app_log_func(f"QualityOutlierFrame: Found quality flags file: {quality_file_path}")
            try:
                # Expected header: PID,OriginalFilename,NumBadChannelsIdentified,UserSetThreshold
                df_flags = pd.read_csv(quality_file_path)

                if df_flags.empty:
                    self.app_log_func("QualityOutlierFrame: Quality flags file is empty.")
                    self.no_flags_label.configure(
                        text="Quality flags file ('Potential_Outlier_Participants.txt') found but is empty.")
                    return

                # If we found flags, destroy the placeholder label
                if self.no_flags_label.winfo_exists():
                    self.no_flags_label.destroy()

                for index, row in df_flags.iterrows():
                    pid = str(row['PID'])
                    filename = str(row['OriginalFilename'])
                    bad_channels = int(row['NumBadChannelsIdentified'])
                    threshold = int(row['UserSetThreshold'])

                    exclude_var = tk.BooleanVar(master=self, value=False)  # Default to NOT exclude
                    info = {
                        'pid': pid,
                        'filename': filename,
                        'bad_channels': bad_channels,
                        'threshold': threshold,
                        'exclude_var': exclude_var
                    }
                    self.quality_flagged_participants_info.append(info)

                    # Create UI for this flagged PID within the scrollable_frame
                    entry_text = f"Exclude {pid}? (File: {filename}, Bads: {bad_channels}, Thresh: {threshold})"
                    chk = ctk.CTkCheckBox(self.scrollable_frame, text=entry_text, variable=exclude_var)
                    # Use pack for items within the scrollable frame for simplicity
                    chk.pack(anchor="w", padx=5, pady=2)

                if self.quality_flagged_participants_info:
                    self.select_all_button.configure(state="normal")
                    self.deselect_all_button.configure(state="normal")
                else:  # Should be caught by df_flags.empty, but as a safeguard
                    self.no_flags_label = ctk.CTkLabel(self.scrollable_frame,
                                                       text="No subjects met quality flag criteria in the file.")
                    self.no_flags_label.pack(pady=10, padx=10, fill="x", expand=True)


            except FileNotFoundError:  # Should be caught by os.path.exists, but for completeness
                self.app_log_func(
                    f"QualityOutlierFrame: File '{QUALITY_FLAGS_FILENAME}' not found though os.path.exists was true (race condition?).")
                self.no_flags_label.configure(text=f"File '{QUALITY_FLAGS_FILENAME}' seems to have disappeared.")
            except pd.errors.EmptyDataError:
                self.app_log_func(f"QualityOutlierFrame: File '{QUALITY_FLAGS_FILENAME}' is empty or not valid CSV.")
                self.no_flags_label.configure(text=f"File '{QUALITY_FLAGS_FILENAME}' is empty or unreadable.")
            except KeyError as ke:
                self.app_log_func(f"QualityOutlierFrame: Missing expected column in '{QUALITY_FLAGS_FILENAME}': {ke}")
                self.no_flags_label.configure(text=f"Flags file has incorrect columns (Missing: {ke}).")
            except Exception as e:
                self.app_log_func(
                    f"QualityOutlierFrame: Error parsing quality flags file '{quality_file_path}': {e}\n{traceback.format_exc()}")
                self.no_flags_label.configure(text=f"Error reading flags file. See console log.")
        else:
            self.app_log_func(
                f"QualityOutlierFrame: No quality flags file ('{QUALITY_FLAGS_FILENAME}') found in {folder_path}.")
            # self.no_flags_label is already set by _clear_display to indicate this.
            self.no_flags_label.configure(text=f"No quality flags file ('{QUALITY_FLAGS_FILENAME}') found.")

    def _select_all_for_exclusion(self, exclude_state=True):
        """Sets the exclusion state for all currently listed quality-flagged PIDs."""
        if not self.quality_flagged_participants_info:
            self.app_log_func("QualityOutlierFrame: No flagged participants to select/deselect.")
            return

        action = "Excluding" if exclude_state else "Including"
        self.app_log_func(f"QualityOutlierFrame: {action} all listed quality-flagged participants.")
        for info in self.quality_flagged_participants_info:
            info['exclude_var'].set(exclude_state)

    def _deselect_all_for_exclusion(self):
        self._select_all_for_exclusion(exclude_state=False)

    def get_pids_to_exclude(self) -> set:
        """
        Returns a set of PIDs that the user has checked for exclusion
        based on the quality flags.
        """
        excluded_pids = set()
        for info in self.quality_flagged_participants_info:
            if info['exclude_var'].get():
                excluded_pids.add(info['pid'])

        if excluded_pids:
            self.app_log_func(
                f"QualityOutlierFrame: Returning PIDs for exclusion based on quality flags: {sorted(list(excluded_pids))}")
        # else:
        # self.app_log_func("QualityOutlierFrame: No PIDs selected for exclusion based on quality flags.")
        return excluded_pids

    # --- Placeholder for statistical outlier methods ---
    # def display_statistical_outliers(self, outliers_data):
    #     # This will be developed in Phase B
    #     pass

    # def get_pids_to_exclude_from_statistical_outliers(self) -> set:
    #     # This will be developed in Phase B
    #     return set()

    # def get_all_excluded_pids_for_analysis(self) -> set:
    #     """Combines exclusions from quality flags and statistical outliers."""
    #     quality_excluded = self.get_pids_to_exclude()
    #     # stat_excluded = self.get_pids_to_exclude_from_statistical_outliers() # For Phase B
    #     # all_excluded = quality_excluded.union(stat_excluded)
    #     # self.app_log_func(f"QualityOutlierFrame: Total PIDs for exclusion: {all_excluded if all_excluded else 'None'}")
    #     # For now, just returns quality excluded as stat outliers are not implemented yet
    #     return quality_excluded