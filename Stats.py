#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stats.py

Provides a Toplevel window for statistical analysis of FPVS results.
Performs:
1. Paired comparisons (t-test or Wilcoxon) between two conditions
   for selected metrics (Z-Score/SNR etc.) averaged over Regions of Interest (ROIs)
   across multiple selected frequencies. Can analyze a single selected ROI or all defined ROIs.
2. Single-condition Significance Check to identify Conditions/ROIs/Frequencies
   where the average metric exceeds a user-specified threshold.

Results are displayed and paired comparison results are exportable to Excel.

Dependencies: customtkinter, pandas, numpy, scipy, (optional: config.py for TARGET_FREQUENCIES)
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import glob
import re
import pandas as pd
import numpy as np
import scipy.stats as stats
import math # For sqrt in effect size
import traceback
import stats_export

# Attempt to import from config, handle if running standalone
try:
    # Assumes config.py is in the same directory or Python path
    # Or that the main app adds it to the Python path
    from config import TARGET_FREQUENCIES
except ImportError:
    print("Warning: config.py not found or TARGET_FREQUENCIES not defined. Using default frequencies.")
    # Define default frequencies if config is missing
    TARGET_FREQUENCIES = np.round(np.arange(1.2, 17.0, 1.2), 1) # Example default


# Define ROIs (e.g., based on Filho et al., 2021 or project specific)
# Ensure channel names match exactly those in the Excel output's 'Electrode' column
ROIS = {
    "Frontal Lobe": ["F3", "F4", "Fz"],
    "Occipital Lobe": ["O1", "O2", "Oz"],
    "Parietal Lobe": ["P3", "P4", "Pz"],
    "Central Lobe": ["C3", "C4", "Cz"]
    # Add more ROIs here if needed, e.g.:
    # "Temporal Lobe": ["T7", "T8", "TP9", "TP10"],
}
ALL_ROIS_OPTION = "(All ROIs)" # Constant for the special option for paired tests


class StatsAnalysisWindow(ctk.CTkToplevel):
    """
    A Toplevel window for performing and displaying paired statistical comparisons
    and single-condition significance checks on FPVS results.
    """

    def run_analysis(self):
        """ Paired comparison analysis """
        self.log_to_main_app("Run Paired Comparison button clicked.")
        self.results_structured = []
        self.significant_findings = {}
        self.results_textbox.configure(state="normal")
        self.results_textbox.delete("1.0", tk.END)
        self.export_button.configure(state="disabled")
        self.export_sig_button.configure(state="disabled")  # <<< CHANGE: Disable sig export too

    def __init__(self, master, default_folder=""):
        """
        Initializes the Stats Analysis Window.

        Args:
            master: The parent Tkinter/CTk window.
            default_folder (str, optional): Default folder path to suggest. Defaults to "".
        """
        if not isinstance(master, (tk.Tk, tk.Toplevel, ctk.CTk, ctk.CTkToplevel)):
            raise TypeError("Master must be a Tkinter root or Toplevel window.")

        super().__init__(master)
        self.title("FPVS Statistical Analysis Tool")
        # Increased default size for better layout
        self.geometry("850x950")
        # Make window modal (prevents interaction with parent window)
        self.grab_set()
        # Focus on this window when opened
        self.lift()
        self.focus_force()

        self.master_app = master  # Reference to the parent app window
        self.default_output_folder = default_folder
        self.subject_data = {}  # Stores {pid: {condition: file_path}}
        self.conditions_list = []  # List of detected condition names
        self.subjects_list = []  # List of detected subject IDs (PIDs)
        self.results_structured = []  # Stores results dicts for paired comparison export
        self.significant_findings = {}  # Stores {cond: {roi: [details]}} for sig check

        # --- Define GUI Variables ---
        self.stats_data_folder_var = tk.StringVar(value=self.default_output_folder)
        self.detected_info_var = tk.StringVar(value="Select the folder that contains all of your FPVS results excel files.")
        self.metric_var = tk.StringVar(value="Z Score")  # Default metric
        self.roi_var = tk.StringVar(value=ALL_ROIS_OPTION)  # Default ROI for paired test
        self.condition_A_var = tk.StringVar()
        self.condition_B_var = tk.StringVar()
        self.freq_checkbox_vars = {}  # Stores {freq_str: tk.BooleanVar}
        self.significance_threshold_var = tk.StringVar(value="1.96")  # Default threshold for sig check

        # Add instance variable for the new button state management
        self.export_sig_button = None  # Initialize significance check export button variable

        # --- Create Widgets ---
        self.create_stats_widgets()

        # --- Initial Scan if Default Folder Exists ---
        if self.default_output_folder and os.path.isdir(self.default_output_folder):
            self.scan_folder()

        # Ensure window destruction cleans up resources if necessary
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        """Handles window closing."""
        self.log_to_main_app("Closing Stats Analysis window.")
        self.grab_release() # Release modal grab
        self.destroy()

    # ==============================================================
    # Widget Creation
    # ==============================================================
    def create_stats_widgets(self):
        """Creates and lays out the widgets for the statistics window."""

        # Register validation command for numeric entry
        validate_num_cmd = (self.register(self._validate_numeric_input), '%P')

        # Configure main frame grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        # Configure grid layout for main_frame content
        main_frame.grid_rowconfigure(2, weight=1)  # Results frame row expands (changed from 3)
        main_frame.grid_columnconfigure(0, weight=1)

        # --- 1. Data Input Frame ---
        input_frame = ctk.CTkFrame(main_frame)
        input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        input_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(input_frame, text="Data Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        folder_entry = ctk.CTkEntry(input_frame, textvariable=self.stats_data_folder_var, state="readonly")
        folder_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        browse_button = ctk.CTkButton(input_frame, text="Select Folder w/ excel files", command=self.browse_folder)
        browse_button.grid(row=0, column=2, padx=5, pady=5)

        detected_label = ctk.CTkLabel(input_frame, textvariable=self.detected_info_var, justify="left", anchor="w")
        detected_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        # --- 2. Analysis Setup Frame ---
        setup_frame = ctk.CTkFrame(main_frame)
        setup_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        # Configure grid columns for setup_frame
        setup_frame.grid_columnconfigure(1, weight=1)
        setup_frame.grid_columnconfigure(3, weight=1)

        # Row 0: Metric & Significance Threshold
        ctk.CTkLabel(setup_frame, text="Metric:").grid(row=0, column=0, padx=(5, 2), pady=5, sticky="w")
        metric_menu = ctk.CTkOptionMenu(setup_frame, variable=self.metric_var,
                                        values=["Z Score", "SNR", "FFT Amplitude (uV)", "BCA (uV)"])
        metric_menu.set("Z Score")
        metric_menu.grid(row=0, column=1, padx=(0, 5), pady=5, sticky="ew")

        ctk.CTkLabel(setup_frame, text="Significance Threshold:").grid(row=0, column=2, padx=(15, 2), pady=5, sticky="w")
        self.sig_threshold_entry = ctk.CTkEntry(
            setup_frame, textvariable=self.significance_threshold_var,
            validate='key', validatecommand=validate_num_cmd, width=80
        )
        self.sig_threshold_entry.grid(row=0, column=3, padx=(0, 5), pady=5, sticky="w")

        # Row 1: ROI for Paired Test
        ctk.CTkLabel(setup_frame, text="ROI (Paired Test):").grid(row=1, column=0, padx=(5, 2), pady=5, sticky="w")
        roi_options = [ALL_ROIS_OPTION] + list(ROIS.keys())
        roi_menu = ctk.CTkOptionMenu(setup_frame, variable=self.roi_var, values=roi_options)
        roi_menu.set(ALL_ROIS_OPTION)
        roi_menu.grid(row=1, column=1, padx=(0, 5), pady=5, sticky="ew")

        # Row 2: Frequencies (Label + Scrollable Frame)
        ctk.CTkLabel(setup_frame, text="Frequencies:").grid(row=2, column=0, padx=(5, 2), pady=5, sticky="nw")
        freq_frame = ctk.CTkScrollableFrame(setup_frame, label_text="", height=120)  # Slightly taller
        freq_frame.grid(row=2, column=1, columnspan=3, padx=(0, 5), pady=5, sticky="nsew")  # Span across remaining cols

        # Populate Frequency Checkboxes
        self.freq_checkbox_vars = {}
        try:
            # Ensure TARGET_FREQUENCIES is iterable (list or numpy array)
            formatted_freqs = [f"{f:.1f}_Hz" for f in TARGET_FREQUENCIES]
        except TypeError:
            self.log_to_main_app("Error: TARGET_FREQUENCIES not defined or not iterable. Using empty list.")
            formatted_freqs = []
            messagebox.showerror("Configuration Error",
                                 "TARGET_FREQUENCIES not defined correctly.\nPlease check config.py or default setup.")

        for i, freq_str in enumerate(formatted_freqs):
            var = tk.BooleanVar(value=(i == 0))  # Select first by default only if list is not empty
            cb = ctk.CTkCheckBox(freq_frame, text=freq_str, variable=var)
            cb.pack(anchor="w", padx=5, pady=1)
            self.freq_checkbox_vars[freq_str] = var

        # Row 3: Frequency Select/Deselect Buttons
        freq_button_frame = ctk.CTkFrame(setup_frame, fg_color="transparent")
        freq_button_frame.grid(row=3, column=1, columnspan=3, sticky="w", padx=(0, 5), pady=(0, 5))  # Align left
        btn_select_all = ctk.CTkButton(freq_button_frame, text="Select All Harmonics", command=self.select_all_freqs,
                                       width=120)
        btn_select_all.pack(side="left", padx=(0, 5))
        btn_deselect_all = ctk.CTkButton(freq_button_frame, text="Deselect All Harmonics", command=self.deselect_all_freqs,
                                         width=120)
        btn_deselect_all.pack(side="left")

        # Row 4: Paired Comparison Condition Selection
        ctk.CTkLabel(setup_frame, text="Condition A (Paired):").grid(row=4, column=0, padx=(5, 2), pady=5, sticky="w")
        self.cond_A_menu = ctk.CTkOptionMenu(setup_frame, variable=self.condition_A_var, values=["(Scan Folder)"],
                                             command=self.update_condition_b_options)
        self.cond_A_menu.grid(row=4, column=1, padx=(0, 5), pady=5, sticky="ew")

        ctk.CTkLabel(setup_frame, text="Condition B (Paired):").grid(row=4, column=2, padx=(15, 2), pady=5, sticky="w")
        self.cond_B_menu = ctk.CTkOptionMenu(setup_frame, variable=self.condition_B_var, values=["(Scan Folder)"])
        self.cond_B_menu.grid(row=4, column=3, padx=(0, 5), pady=5, sticky="ew")

        # --- 3. Run Buttons & Results Frame ---
        results_outer_frame = ctk.CTkFrame(main_frame)  # Contains buttons and results box
        results_outer_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        results_outer_frame.grid_rowconfigure(1, weight=1)  # Textbox row expands
        results_outer_frame.grid_columnconfigure(0, weight=1)  # Textbox col expands

        # Frame for Run Buttons
        run_buttons_frame = ctk.CTkFrame(results_outer_frame, fg_color="transparent")
        run_buttons_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 5))

        run_paired_button = ctk.CTkButton(run_buttons_frame, text="Run Paired Comparison", command=self.run_analysis)
        run_paired_button.pack(side="left", padx=(0, 10), pady=5)

        run_sig_check_button = ctk.CTkButton(run_buttons_frame, text="Check for Significant Responses",
                                             command=self.run_significance_check)
        run_sig_check_button.pack(side="left", padx=(0, 10), pady=5)

        # Paired results export button
        self.export_button = ctk.CTkButton(run_buttons_frame, text="Export Paired Results", state="disabled",
                                           command=self.export_results)
        self.export_button.pack(side="left", padx=(0, 10), pady=5)

        # Significance Check Export Button
        self.export_sig_button = ctk.CTkButton(run_buttons_frame, text="Export Sig. Check Results", state="disabled",
                                               command=self.call_export_significance_results)
        self.export_sig_button.pack(side="left", padx=0, pady=5)

        # Results Textbox (Increased font size)
        results_font = ctk.CTkFont(family="Courier New", size=12)  # Larger font
        self.results_textbox = ctk.CTkTextbox(results_outer_frame, wrap="word", state="disabled", font=results_font)
        self.results_textbox.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")

    # ==============================================================
    # Helper Functions (Including Validation)
    # ==============================================================
    def _validate_numeric_input(self, P):
        """ Allows empty string, '-', or valid float number """
        if P == "" or P == "-":
            return True
        try:
            float(P)
            return True
        except ValueError:
            # self.bell() # Optional: sound feedback
            return False

    def log_to_main_app(self, message):
        """ Safely logs messages to the main app's log method if it exists. """
        try:
            if hasattr(self.master_app, 'log') and callable(self.master_app.log):
                self.master_app.log(f"[Stats] {message}")
            else:
                # Fallback to print if master doesn't have 'log' (e.g., when testing)
                print(f"[Stats] {message}")
        except Exception as e:
            print(f"[Stats Log Error] {e} | Original message: {message}")


    def browse_folder(self):
        """Opens folder selection dialog and triggers scanning."""
        current_folder = self.stats_data_folder_var.get()
        initial_dir = current_folder if os.path.isdir(current_folder) else os.path.expanduser("~")
        folder = filedialog.askdirectory(
            title="Select Parent Folder Containing Condition Subfolders",
            initialdir=initial_dir
        )
        if folder:
            self.stats_data_folder_var.set(folder)
            self.scan_folder()
        else:
            self.log_to_main_app("Folder selection cancelled.")

    def scan_folder(self):
        """
        Scans the selected PARENT folder for SUBFOLDERS (conditions).
        Looks inside each subfolder for participant Excel files matching PID pattern.
        Updates internal lists and GUI dropdowns.
        """
        parent_folder = self.stats_data_folder_var.get()
        if not parent_folder or not os.path.isdir(parent_folder):
            self.detected_info_var.set("Invalid parent folder selected.")
            self.update_condition_menus([]) # Clear menus
            return

        self.log_to_main_app(f"Scanning parent folder for condition subfolders: {parent_folder}")
        subjects = set()
        conditions = set()
        self.subject_data = {} # Reset data
        # PID pattern: Starts with 'P' followed by digits, anywhere in the filename before '.xlsx'
        pid_pattern = re.compile(r"^(P\d+).*\.xlsx$", re.IGNORECASE)

        try:
            for item_name in os.listdir(parent_folder):
                item_path = os.path.join(parent_folder, item_name)
                if os.path.isdir(item_path):
                    # Use subfolder name as potential condition identifier
                    subfolder_name = item_name
                    # Simple cleaning: remove leading numbers/hyphens/spaces
                    condition_name = re.sub(r'^\d+\s*[-_]*\s*', '', subfolder_name).strip()
                    if not condition_name: continue # Skip if cleaning results in empty name

                    self.log_to_main_app(f"  Found potential condition subfolder: '{subfolder_name}' -> Condition Name: '{condition_name}'")
                    conditions.add(condition_name)
                    found_files_in_subfolder = False

                    # Look for Excel files directly within this subfolder
                    subfolder_files = glob.glob(os.path.join(item_path, "*.xlsx"))
                    for f_path in subfolder_files:
                        excel_filename = os.path.basename(f_path)
                        pid_match = pid_pattern.match(excel_filename)

                        if pid_match:
                            pid = pid_match.group(1).upper() # Use uppercase PID consistently
                            subjects.add(pid)
                            found_files_in_subfolder = True
                            if pid not in self.subject_data:
                                self.subject_data[pid] = {}
                            if condition_name in self.subject_data[pid]:
                                self.log_to_main_app(f"    Warn: Duplicate Excel file found for Subject {pid}, Condition '{condition_name}'. Overwriting path with {excel_filename}")
                            self.subject_data[pid][condition_name] = f_path
                        # else: # Optional logging for non-matching files
                            # self.log_to_main_app(f"    Info: File '{excel_filename}' in subfolder '{subfolder_name}' does not match PID pattern. Skipping.")

                    if not found_files_in_subfolder:
                         self.log_to_main_app(f"    Warn: No Excel files matching PID pattern (e.g., P001_*.xlsx) found in subfolder '{subfolder_name}'.")

        except PermissionError as e:
             self.log_to_main_app(f"!!! Permission Error scanning folder: {parent_folder}. Check permissions.")
             messagebox.showerror("Scanning Error", f"Permission denied accessing folder or its contents:\n{parent_folder}\n{e}")
             self.detected_info_var.set("Error: Permission denied during scanning.")
             self.update_condition_menus([])
             return
        except Exception as e:
            self.log_to_main_app(f"!!! Error scanning folder structure: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Scanning Error", f"An unexpected error occurred during scanning:\n{e}")
            self.detected_info_var.set("Error during scanning.")
            self.update_condition_menus([])
            return

        self.subjects_list = sorted(list(subjects))
        self.conditions_list = sorted(list(conditions))

        # Update GUI info label and menus
        if not self.conditions_list or not self.subjects_list:
            info_text = "Scan complete: No valid condition subfolders or subject Excel files (e.g., P001_*.xlsx) found."
            messagebox.showwarning("Scan Results", info_text)
        else:
            info_text = f"Scan complete: Found {len(self.subjects_list)} subjects (PIDs) and {len(self.conditions_list)} conditions (subfolders)."
            self.log_to_main_app(f"Detected Subjects: {', '.join(self.subjects_list)}")
            self.log_to_main_app(f"Detected Conditions: {', '.join(self.conditions_list)}")

        self.detected_info_var.set(info_text)
        self.update_condition_menus(self.conditions_list)

    def update_condition_menus(self, conditions):
        """Updates the condition dropdown menus based on scanned folders."""
        valid_conditions = [c for c in conditions if c] # Exclude potential empty strings

        if not valid_conditions:
            display_list = ["(Scan Folder)"]
            self.condition_A_var.set(display_list[0])
        else:
            display_list = valid_conditions
            # Preserve selection if still valid, otherwise default to first
            if self.condition_A_var.get() not in valid_conditions:
                self.condition_A_var.set(valid_conditions[0])

        self.cond_A_menu.configure(values=display_list)
        # Update Cond B options whenever Cond A options change
        self.update_condition_b_options()

    def update_condition_b_options(self, *args):
        """Updates Condition B options to exclude the selection in Condition A."""
        cond_a = self.condition_A_var.get()
        valid_b_options = [c for c in self.conditions_list if c != cond_a] # Exclude Cond A selection

        # Handle placeholder messages
        if not valid_b_options:
            if cond_a and cond_a != "(Scan Folder)":
                 valid_b_options = ["(No other conditions)"]
            else: # If Cond A is placeholder or list is empty
                 valid_b_options = ["(Select Condition A)"]

        current_b_val = self.condition_B_var.get()
        self.cond_B_menu.configure(values=valid_b_options)

        # Reset B selection if current is invalid or same as A
        if current_b_val not in valid_b_options or current_b_val == cond_a:
            self.condition_B_var.set(valid_b_options[0]) # Default to first valid option

    def select_all_freqs(self):
        """Sets all frequency checkboxes to True."""
        for var in self.freq_checkbox_vars.values():
            var.set(True)

    def deselect_all_freqs(self):
        """Sets all frequency checkboxes to False."""
        for var in self.freq_checkbox_vars.values():
            var.set(False)

    # ==============================================================
    # Data Aggregation
    # ==============================================================
    def aggregate_data(self, metric, roi_name, frequency, cond_a, cond_b=None):
        """
        Aggregates data for the specified parameters across subjects.
        If cond_b is None, aggregates only for cond_a.

        Args:
            metric (str): The metric to extract (e.g., "Z Score").
            roi_name (str): The name of the ROI (must be a key in ROIS dict).
            frequency (str): The frequency column name (e.g., "1.2_Hz").
            cond_a (str): The name of the first condition.
            cond_b (str, optional): The name of the second condition for paired analysis. Defaults to None.

        Returns:
            tuple: (scores_a, scores_b, included_subjects)
                   scores_a (list): List of aggregated scores for cond_a.
                   scores_b (list or None): List of scores for cond_b if paired, else None.
                   included_subjects (list): List of PIDs included in the aggregation.
                   Returns (None, None, None) if aggregation fails for any reason.
        """
        log_conds = f"{cond_a} vs {cond_b}" if cond_b else f"{cond_a} only"
        self.log_to_main_app(f"Aggregating: Metric='{metric}', ROI='{roi_name}', Freq='{frequency}', Conds='{log_conds}'")

        scores_a = []
        scores_b = [] if cond_b else None # Initialize scores_b only if needed
        included_subjects = []
        roi_channels = ROIS.get(roi_name)
        if not roi_channels:
            self.log_to_main_app(f"Error: Invalid ROI name '{roi_name}' provided. Check ROIS definition.")
            return None, None, None

        # Map metric display name to exact sheet name expected in Excel files
        sheet_name_map = {
            "Z Score": "Z Score",
            "SNR": "SNR",
            "FFT Amplitude (uV)": "FFT Amplitude (uV)",
            "BCA (uV)": "BCA (uV)"
            # Add other mappings if needed
        }
        sheet_name = sheet_name_map.get(metric)
        if not sheet_name:
             self.log_to_main_app(f"Error: Invalid metric '{metric}' selected. Cannot map to sheet name.")
             return None, None, None

        if not self.subjects_list:
             self.log_to_main_app("Warning: No subjects found during scan. Cannot aggregate data.")
             return None, None, None

        # Iterate through all scanned subjects
        for pid in self.subjects_list:
            subject_files = self.subject_data.get(pid, {})
            file_a = subject_files.get(cond_a)
            file_b = subject_files.get(cond_b) if cond_b else None

            # Check if files exist for required conditions for this subject
            if not file_a:
                # self.log_to_main_app(f"Debug: Skipping Subject {pid} - Missing file/data for Condition A ('{cond_a}').")
                continue
            if cond_b and not file_b:
                # self.log_to_main_app(f"Debug: Skipping Subject {pid} - Missing file/data for Condition B ('{cond_b}').")
                continue

            # Check file paths actually exist on disk
            if not os.path.exists(file_a):
                self.log_to_main_app(f"Warn: File path not found for Subject {pid}, Cond A: {file_a}. Skipping.")
                continue
            if cond_b and not os.path.exists(file_b):
                self.log_to_main_app(f"Warn: File path not found for Subject {pid}, Cond B: {file_b}. Skipping.")
                continue

            # Try reading data and calculating ROI mean for this subject
            try:
                # --- Process Condition A ---
                df_a = pd.read_excel(file_a, sheet_name=sheet_name, index_col="Electrode")
                # Check if frequency column exists
                if frequency not in df_a.columns:
                    self.log_to_main_app(f"Warn: Freq column '{frequency}' missing in {os.path.basename(file_a)} (Subj {pid}). Skipping for this freq.")
                    continue
                # Extract data for ROI channels, drop any missing channels for this subject
                roi_vals_a = df_a.reindex(roi_channels)[frequency].dropna()

                # --- Process Condition B (only if required) ---
                roi_vals_b = None
                if cond_b:
                    df_b = pd.read_excel(file_b, sheet_name=sheet_name, index_col="Electrode")
                    if frequency not in df_b.columns:
                        self.log_to_main_app(f"Warn: Freq column '{frequency}' missing in {os.path.basename(file_b)} (Subj {pid}). Skipping for this freq.")
                        continue
                    roi_vals_b = df_b.reindex(roi_channels)[frequency].dropna()

                # --- Check data completeness for ROI channels ---
                # Require *all* defined ROI channels to be present after dropping NaNs
                complete_a = len(roi_vals_a) == len(roi_channels)
                # Complete B only matters if we are doing a paired analysis
                complete_b = cond_b is None or (roi_vals_b is not None and len(roi_vals_b) == len(roi_channels))

                if complete_a and complete_b:
                    avg_a = roi_vals_a.mean()
                    scores_a.append(avg_a)
                    if cond_b: # If paired analysis, calculate and append B
                        avg_b = roi_vals_b.mean()
                        scores_b.append(avg_b)
                    included_subjects.append(pid) # Add subject ID to included list
                else:
                    # Log which channels were missing if incomplete
                    missing_a_str = "" if complete_a else f"Cond A missing channels: {set(roi_channels) - set(roi_vals_a.index) or 'N/A'}. "
                    missing_b_str = ""
                    if cond_b and not complete_b: missing_b_str = f"Cond B missing channels: {set(roi_channels) - set(roi_vals_b.index) or 'N/A'}. "
                    self.log_to_main_app(f"Warn: Subj {pid}, Freq {frequency}, ROI {roi_name} - Incomplete channel data. {missing_a_str}{missing_b_str}Excluding.")

            # Handle specific errors during file reading/processing
            except FileNotFoundError:
                self.log_to_main_app(f"Warn: FileNotFoundError for subject {pid}, conds '{cond_a}'/'{cond_b}'. Check paths. Excluding.")
                continue # Skip to next subject
            except KeyError as e:
                 # More specific messages for common KeyErrors
                 if sheet_name in str(e): self.log_to_main_app(f"Warn: Sheet '{sheet_name}' not found in files for {pid}. Check Metric setting and Excel files. Excluding.")
                 elif "Electrode" in str(e): self.log_to_main_app(f"Warn: 'Electrode' index column not found in files for {pid}. Check Excel format. Excluding.")
                 elif frequency in str(e): self.log_to_main_app(f"Warn: Freq column '{frequency}' processing error for {pid}: {e}. Check data. Excluding.")
                 else: self.log_to_main_app(f"Warn: KeyError reading data for subject {pid}: {e}. Excluding.")
                 continue
            except ValueError as e:
                  # Check if error message relates to missing ROI channels during reindex
                 if any(ch in str(e) for ch in roi_channels): self.log_to_main_app(f"Warn: One or more ROI channels ({roi_channels}) not found in index for {pid}. Check ROI definition/Excel file. Excluding.")
                 else: self.log_to_main_app(f"ValueError reading data for subject {pid}: {e}. Excluding.")
                 continue
            except Exception as e:
                # Catch-all for other unexpected errors
                self.log_to_main_app(f"!!! Unexpected Error reading data for subject {pid}: {e}\n{traceback.format_exc()}")
                continue # Skip subject on unexpected error

        # Check if any subjects were successfully included
        if not included_subjects:
             self.log_to_main_app("Aggregation complete: No subjects found with complete data for the selected parameters.")
             return None, None, None # Return None if no subjects included

        self.log_to_main_app(f"Aggregation complete: Successfully aggregated data for {len(included_subjects)} subjects: {', '.join(included_subjects)}")
        return scores_a, scores_b, included_subjects # Return aggregated scores and included IDs

    # ==============================================================
    # Statistical Testing
    # ==============================================================
    def perform_paired_test(self, scores_a, scores_b):
        """
        Performs Shapiro-Wilk test on differences, then Paired t-test or Wilcoxon Signed-Rank test.
        Calculates descriptive statistics and effect size (Cohen's d for t-test).

        Args:
            scores_a (list): List of scores for condition A.
            scores_b (list): List of scores for condition B.

        Returns:
            dict: Dictionary containing statistical results and descriptive stats.
                  Includes an 'error' key if testing fails.
        """
        results = {}
        n_pairs = len(scores_a)
        if n_pairs < 3: # Minimum pairs for meaningful testing (especially normality)
             results['error'] = f"Insufficient pairs (N={n_pairs}, requires >= 3) for reliable statistical testing."
             results['N'] = n_pairs
             return results

        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        differences = scores_a - scores_b

        # --- Descriptive Stats ---
        # Use ddof=1 for sample standard deviation
        mean_a, std_a = np.mean(scores_a), np.std(scores_a, ddof=1)
        mean_b, std_b = np.mean(scores_b), np.std(scores_b, ddof=1)
        median_a, median_b = np.median(scores_a), np.median(scores_b)
        # Interquartile Range (IQR)
        try:
            q75a, q25a = np.percentile(scores_a, [75 ,25])
            iqr_a = q75a - q25a
            q75b, q25b = np.percentile(scores_b, [75 ,25])
            iqr_b = q75b - q25b
        except IndexError: # Handle case N<2 for percentile
            iqr_a, iqr_b = np.nan, np.nan


        results['N'] = n_pairs
        results['Mean_A'], results['SD_A'] = mean_a, std_a
        results['Mean_B'], results['SD_B'] = mean_b, std_b
        results['Median_A'], results['IQR_A'] = median_a, iqr_a
        results['Median_B'], results['IQR_B'] = median_b, iqr_b

        # --- Normality Test (Shapiro-Wilk on differences) ---
        shapiro_stat, shapiro_p = np.nan, np.nan
        is_normal = False # Default to non-normal
        results['Shapiro_Note'] = ""
        try:
            # Handle edge cases for Shapiro test
            if np.all(np.isclose(differences, differences[0])): # All differences are identical
                 shapiro_stat, shapiro_p = np.nan, 1.0 # Cannot test normality, assume possible for t-test? Or force Wilcoxon? Let's assume normal enough.
                 results['Shapiro_Note'] = "Note: Differences are constant."
                 is_normal = True # Allow t-test if differences constant but >0
            elif len(np.unique(differences)) < 3: # Too few unique values
                 shapiro_stat, shapiro_p = np.nan, 0.0 # Force non-normal
                 results['Shapiro_Note'] = "Note: Fewer than 3 unique difference values."
            elif n_pairs < 3: # SciPy shapiro needs N>=3
                 shapiro_stat, shapiro_p = np.nan, 0.0 # Force non-normal
                 results['Shapiro_Note'] = f"Note: N={n_pairs} < 3 for Shapiro test."
            else:
                 shapiro_stat, shapiro_p = stats.shapiro(differences)
                 is_normal = shapiro_p > 0.05 # Standard alpha for normality check

            results['Shapiro_W'] = shapiro_stat
            results['Shapiro_p'] = shapiro_p
            results['Normality_Decision'] = "Normal" if is_normal else "Not Normal"

        except Exception as e:
            results['error'] = f"Shapiro-Wilk test failed: {e}. Cannot proceed with primary test."
            results['Normality_Decision'] = "Test Failed"
            return results # Stop if normality test fails unexpectedly

        # --- Primary Statistical Test & Effect Size ---
        if is_normal:
            # Use Paired t-test
            results['Test_Used'] = "Paired t-test"
            try:
                t_stat, p_val = stats.ttest_rel(scores_a, scores_b, nan_policy='omit') # Omit pairs with NaN if any slip through
                results['Statistic_Type'] = "t"
                results['Statistic_Value'] = t_stat
                results['df'] = n_pairs - 1 # Degrees of freedom for paired t-test
                results['p_value'] = p_val

                # Effect Size: Cohen's d for paired samples
                mean_diff = np.mean(differences)
                std_diff = np.std(differences, ddof=1) # Use sample SD of differences
                # Avoid division by zero or near-zero
                if std_diff is not None and not np.isclose(std_diff, 0):
                     cohen_d = mean_diff / std_diff
                else:
                     cohen_d = np.nan
                     self.log_to_main_app(f"Note: Could not calculate Cohen's d (SD of differences is zero or near-zero). N={n_pairs}")

                results['Effect_Size_Type'] = "Cohen's d (paired)"
                results['Effect_Size_Value'] = cohen_d

            except Exception as e:
                results['error'] = f"Paired t-test failed: {e}"
                self.log_to_main_app(f"Paired t-test error: {e}\n{traceback.format_exc()}")
        else:
            # Use Wilcoxon Signed-Rank Test
            results['Test_Used'] = "Wilcoxon Signed-Rank"
            try:
                # Wilcoxon requires non-zero differences
                non_zero_diffs = differences[~np.isclose(differences, 0)]
                wilcoxon_n = len(non_zero_diffs)
                results['Wilcoxon_Note'] = ""
                if wilcoxon_n < n_pairs:
                     results['Wilcoxon_Note'] = f"Note: {n_pairs - wilcoxon_n} zero difference(s) excluded from ranking."

                # Check if enough non-zero differences for reliable test
                # SciPy default threshold for normal approximation is often ~25, but test runs with fewer.
                # Let's set a practical minimum, e.g., 5 or more non-zero differences.
                min_wilcoxon_n = 5
                if wilcoxon_n >= min_wilcoxon_n :
                    # Use 'wilcox' method for zeros, 'pratt' handles them differently, 'zsplit' includes them in ranking. 'wilcox' is common.
                    # alternative='two-sided' is default
                    wilcox_stat, p_val = stats.wilcoxon(scores_a, scores_b, zero_method='wilcox', nan_policy='omit')
                    results['Statistic_Type'] = "W" # Test statistic W
                    results['Statistic_Value'] = wilcox_stat
                    results['p_value'] = p_val
                    # Effect Size for Wilcoxon: Rank Biserial Correlation is common, but hard to get Z easily.
                    # Report as not calculated for simplicity here.
                    results['Effect_Size_Type'] = "Rank Biserial r"
                    results['Effect_Size_Value'] = "N/A" # Not easily calculated from scipy output

                else:
                    # Not enough non-zero differences
                    results['error'] = f"Wilcoxon test requires >= {min_wilcoxon_n} non-zero differences (found {wilcoxon_n})."
                    results['Statistic_Type'] = "W"; results['Statistic_Value'] = np.nan; results['p_value'] = np.nan
                    results['Effect_Size_Type'] = "N/A"; results['Effect_Size_Value'] = "N/A"

            except Exception as e:
                results['error'] = f"Wilcoxon test failed: {e}"
                self.log_to_main_app(f"Wilcoxon error: {e}\n{traceback.format_exc()}")

        return results


    # ==============================================================
    # Results Formatting
    # ==============================================================

    def format_significance_results(self, findings_dict, metric, threshold):
        """
        Formats the significant findings from the single-condition check
        into a readable string, grouped by Condition, then ROI.

        Args:
            findings_dict (dict): Nested dict {condition: {roi: [finding_list]}}.
            metric (str): The metric being analyzed (e.g., "Z Score").
            threshold (float): The significance threshold used.

        Returns:
            str: Formatted string for display.
        """
        output = f"===== SIGNIFICANCE CHECK RESULTS (Mean {metric} > {threshold:.2f}) =====\n\n"
        any_significant = False
        metric_short_name = metric.split()[0]  # Get "Z", "SNR", "FFT", "BCA"
        metric_key = f'Mean_{metric.replace(" ", "_")}'  # Key used in findings dict

        # Sort conditions alphabetically for consistent output
        sorted_conditions = sorted(findings_dict.keys())

        for condition in sorted_conditions:
            condition_findings = findings_dict[condition]
            # Check if there are any significant ROIs for this condition before adding header
            if not any(condition_findings.values()): continue

            any_significant = True
            output += f"Significant responses found for Condition: '{condition}'\n"
            # Sort ROIs alphabetically within condition
            sorted_rois = sorted(condition_findings.keys())

            for roi_name in sorted_rois:
                roi_findings = condition_findings.get(roi_name, [])
                if not roi_findings: continue  # Skip if no findings for this ROI

                output += f"  * {roi_name}:\n"
                # Sort findings by frequency (convert freq string to float for sorting)
                try:
                    # Attempt numeric sort after removing '_Hz'
                    sorted_findings = sorted(roi_findings, key=lambda x: float(x['Frequency'].replace('_Hz', '')))
                except ValueError:
                    # Fallback to string sort if conversion fails
                    self.log_to_main_app(
                        f"Warning: Could not sort frequencies numerically for ROI {roi_name}, Condition {condition}. Using string sort.")
                    sorted_findings = sorted(roi_findings, key=lambda x: x['Frequency'])

                for finding in sorted_findings:
                    freq = finding['Frequency']
                    # Safely get metric value using .get() with a default
                    mean_val = finding.get(metric_key, np.nan)
                    n_subs = finding.get('N', 'N/A')
                    # Format output line concisely
                    output += f"      - {freq:<8}: Mean {metric_short_name} = {mean_val:.2f} (N={n_subs})\n"  # Left-align freq
            output += "\n"  # Add space between condition blocks

        # Footer / Summary
        if not any_significant:
            output += f"No conditions/ROIs/frequencies found where the Mean {metric} exceeded the {threshold:.2f} threshold."
        else:
            output += "--------------------------------------------------\n"
            output += f"Analysis based on Mean {metric} > {threshold:.2f} threshold across subjects.\n"
            output += f"N = Number of subjects with valid data for the specific Condition/ROI/Frequency.\n"

        return output.strip()

    def format_results_text(self, params, stats_results):
        """
        Formats the statistical results for PAIRED comparisons into a research-friendly string.

        Args:
            params (dict): Dictionary containing analysis parameters (frequency, metric, roi_name, condition_a, condition_b).
            stats_results (dict): Dictionary containing results from perform_paired_test.

        Returns:
            str: Formatted string for display.
        """
        error_msg = stats_results.get('error', '')
        freq = params['frequency']; metric = params['metric']; roi = params['roi_name']
        cond_a = params['condition_a']; cond_b = params['condition_b']; n = stats_results.get('N', 'N/A')

        output = f"--- Paired Analysis: {freq} ---\n" # Header clarifies paired test
        output += f"Metric: {metric}\n"
        output += f"ROI: {roi} (Channels: {','.join(ROIS.get(roi,['N/A']))})\n" # Added channel list
        output += f"Comparison: '{cond_a}' vs '{cond_b}'\n"
        output += f"N (Pairs) = {n}\n\n"

        output += "Descriptive Stats:\n"
        # Format numbers nicely, handle potential NaN
        def format_num(val, precision=3): return f"{val:.{precision}f}" if pd.notna(val) else "N/A"

        if 'Mean_A' in stats_results:
             output += f"  {cond_a:<15}: Mean = {format_num(stats_results['Mean_A'])}, SD = {format_num(stats_results['SD_A'])}\n"
             output += f"  {cond_b:<15}: Mean = {format_num(stats_results['Mean_B'])}, SD = {format_num(stats_results['SD_B'])}\n"
        if 'Median_A' in stats_results:
             output += f"  {cond_a:<15}: Median = {format_num(stats_results['Median_A'])}, IQR = {format_num(stats_results['IQR_A'])}\n"
             output += f"  {cond_b:<15}: Median = {format_num(stats_results['Median_B'])}, IQR = {format_num(stats_results['IQR_B'])}\n"
        output += "\n"

        # Normality Test Info
        if 'Shapiro_p' in stats_results:
            p_shapiro_str = format_num(stats_results['Shapiro_p'])
            w_shapiro_str = format_num(stats_results['Shapiro_W'])
            output += f"Normality of Differences (Shapiro-Wilk): W = {w_shapiro_str}, p = {p_shapiro_str}\n"
            if stats_results.get('Shapiro_Note'): output += f"  {stats_results['Shapiro_Note']}\n"
            output += f"Decision: Differences assessed as {stats_results.get('Normality_Decision', 'N/A')}. Using {stats_results.get('Test_Used', 'N/A')}.\n"
            if stats_results.get('Wilcoxon_Note'): output += f"  {stats_results['Wilcoxon_Note']}\n"
            output += "\n"
        elif 'error' not in stats_results: # Only show if normality test was expected but missing
             output += "Normality test results not available.\n\n"

        # Primary Test Result
        if stats_results.get('Test_Used'):
            output += f"Statistical Test Result ({stats_results['Test_Used']}):\n"
            p_val = stats_results.get('p_value')
            p_str = "N/A"
            if pd.notna(p_val):
                 p_str = f"{p_val:.3f}" if p_val >= 0.001 else "< .001" # Standard p-value formatting

            stat_val = stats_results.get('Statistic_Value')
            stat_str = "N/A"
            if pd.notna(stat_val):
                 stat_type = stats_results.get('Statistic_Type', '')
                 if stats_results['Test_Used'] == "Paired t-test":
                     df_val = stats_results.get('df', 'N/A')
                     stat_str = f"{stat_type}({df_val}) = {stat_val:.2f}" # t(df) = value
                 else: # Wilcoxon
                     stat_str = f"{stat_type} = {stat_val:.1f}" # W = value
            output += f"  Statistic: {stat_str}\n"
            output += f"  p-value:   {p_str}\n\n" # Align p-value

            # Effect Size
            output += "Effect Size:\n"
            eff_size_val = stats_results.get('Effect_Size_Value')
            eff_size_type = stats_results.get('Effect_Size_Type', 'Effect Size')
            # Format effect size value, handle "N/A" string or NaN
            eff_size_str = format_num(eff_size_val, 2) if isinstance(eff_size_val, (int, float)) else str(eff_size_val)
            output += f"  {eff_size_type}: {eff_size_str}\n\n"

            # Interpretation
            if pd.notna(p_val):
                sig_thresh = 0.05 # Standard alpha
                interpretation = "significant" if p_val < sig_thresh else "non-significant"
                output += f"Interpretation (Paired): The difference between '{cond_a}' and '{cond_b}' was statistically {interpretation} (alpha = {sig_thresh}).\n"
            else:
                 output += "Interpretation (Paired): Could not determine significance (statistical test failed or insufficient data).\n"

        # Append error message if a test failed during execution
        if error_msg:
             output += f"\nNote: {error_msg}\n"

        output += "----------------------------------------\n" # Separator
        return output

    def format_significance_results(self, findings_dict, metric, threshold):
        """
        Formats the significant findings from the single-condition check
        into a readable string, grouped by Condition, then ROI.

        Args:
            findings_dict (dict): Nested dict {condition: {roi: [finding_list]}}.
            metric (str): The metric being analyzed (e.g., "Z Score").
            threshold (float): The significance threshold used.

        Returns:
            str: Formatted string for display.
        """
        output = f"===== SIGNIFICANCE CHECK RESULTS (Mean {metric} > {threshold:.2f}) =====\n\n"
        any_significant = False
        metric_short_name = metric.split()[0] # Get "Z", "SNR", "FFT", "BCA"
        metric_key = f'Mean_{metric.replace(" ","_")}' # Key used in findings dict

        # Sort conditions alphabetically for consistent output
        sorted_conditions = sorted(findings_dict.keys())

        for condition in sorted_conditions:
            condition_findings = findings_dict[condition]
            # Check if there are any significant ROIs for this condition
            if not any(condition_findings.values()): continue

            any_significant = True
            output += f"Significant responses found for Condition: '{condition}'\n"
            # Sort ROIs alphabetically within condition
            sorted_rois = sorted(condition_findings.keys())

            for roi_name in sorted_rois:
                roi_findings = condition_findings.get(roi_name, [])
                if not roi_findings: continue # Skip if no findings for this ROI (shouldn't happen with check above, but safe)

                output += f"  * {roi_name}:\n"
                # Sort findings by frequency (convert freq string to float for sorting)
                try:
                     sorted_findings = sorted(roi_findings, key=lambda x: float(x['Frequency'].replace('_Hz','')))
                except ValueError:
                     # Fallback if frequency format is unexpected
                     self.log_to_main_app(f"Warning: Could not sort frequencies numerically for ROI {roi_name}, Condition {condition}. Using string sort.")
                     sorted_findings = sorted(roi_findings, key=lambda x: x['Frequency'])

                for finding in sorted_findings:
                    freq = finding['Frequency']
                    # Handle potential KeyError if metric key isn't found (shouldn't happen)
                    mean_val = finding.get(metric_key, np.nan)
                    n_subs = finding.get('N', 'N/A')
                    # Format output line concisely
                    output += f"      - {freq:<8}: Mean {metric_short_name} = {mean_val:.2f} (N={n_subs})\n"
            output += "\n" # Add space between condition blocks

        # Footer / Summary
        if not any_significant:
            output += f"No conditions/ROIs/frequencies found where the Mean {metric} exceeded the {threshold:.2f} threshold."
        else:
             output += "--------------------------------------------------\n"
             output += f"Analysis based on Mean {metric} > {threshold:.2f} threshold across subjects.\n"
             output += f"N = Number of subjects with valid data for the specific Condition/ROI/Frequency.\n"

        return output.strip()

    # ==============================================================
    # Analysis Execution Functions
    # ==============================================================
    def run_analysis(self):
        """
        Gathers parameters, aggregates data, runs PAIRED tests for selected ROI(s)
        and frequencies, displays results. Enables export for paired results.
        Includes a warning for common artifact frequencies (6Hz, 12Hz).
        """
        self.log_to_main_app("Run Paired Comparison button clicked.")
        self.results_structured = []  # Clear previous paired results for export
        self.significant_findings = {}  # Also clear sig findings structure
        self.results_textbox.configure(state="normal")
        self.results_textbox.delete("1.0", tk.END)
        self.export_button.configure(state="disabled")  # Disable paired export initially
        self.export_sig_button.configure(state="disabled")  # Disable sig export too

        # --- Get Parameters ---
        folder = self.stats_data_folder_var.get()
        metric = self.metric_var.get()
        selected_roi_option = self.roi_var.get()  # ROI selection for paired test
        cond_a = self.condition_A_var.get()
        cond_b = self.condition_B_var.get()
        selected_freqs = [freq for freq, var in self.freq_checkbox_vars.items() if var.get()]

        # --- Validate Parameters ---
        if not folder or not os.path.isdir(folder): messagebox.showerror("Input Error",
                                                                         "Please select a valid data folder."); self.results_textbox.configure(
            state="disabled"); return
        if not selected_freqs: messagebox.showerror("Input Error",
                                                    "Please select at least one frequency."); self.results_textbox.configure(
            state="disabled"); return
        if not selected_roi_option: messagebox.showerror("Input Error",
                                                         "Please select an ROI option for the paired test."); self.results_textbox.configure(
            state="disabled"); return
        # Validate conditions specifically for paired test
        invalid_cond_msgs = ["(Scan Folder)", "(Select Condition A)", "(No other conditions)"]
        if not cond_a or not cond_b or cond_a == cond_b or cond_a in invalid_cond_msgs or cond_b in invalid_cond_msgs:
            messagebox.showerror("Input Error",
                                 "Please select two different valid conditions from the dropdowns for paired comparison.")
            self.results_textbox.configure(state="disabled");
            return

        # <<< --- ADDED WARNING CHECK --- >>>
        warning_freqs_str = ["6.0_Hz", "12.0_Hz"]
        selected_warning_freqs = [f for f in selected_freqs if f in warning_freqs_str]

        if selected_warning_freqs:
            freq_list_str = ", ".join(selected_warning_freqs).replace('_Hz', '') + " Hz"
            warning_message = (
                f"Warning: Frequency {freq_list_str} is selected.\n\n"
                "Responses at these frequencies can be strongly influenced by screen refresh rates "
                "or the fundamental image presentation rate (6 Hz) and may not reflect purely neural activity.\n\n"
                "Do you want to proceed with the Paired Comparison analysis including these frequencies?"
            )
            proceed = messagebox.askyesno("Frequency Warning (Paired Comparison)", warning_message, icon='warning')
            if not proceed:
                self.log_to_main_app(f"Paired Comparison cancelled by user due to frequency warning ({freq_list_str}).")
                self.results_textbox.insert("1.0",
                                            "Analysis cancelled by user due to frequency warning.\nPlease deselect 6.0 Hz and/or 12.0 Hz if desired and run again.")
                self.results_textbox.configure(state="disabled")
                return  # Stop the analysis
            else:
                self.log_to_main_app(f"User chose to proceed despite frequency warning ({freq_list_str}).")
        # <<< --- END WARNING CHECK --- >>>

        # --- Determine which ROIs to analyze (for paired test) ---
        rois_to_analyze = []
        if selected_roi_option == ALL_ROIS_OPTION:
            rois_to_analyze = list(ROIS.keys())
            self.log_to_main_app(f"Paired Analysis: Running for All ROIs: {', '.join(rois_to_analyze)}")
        elif selected_roi_option in ROIS:
            rois_to_analyze = [selected_roi_option]
            self.log_to_main_app(f"Paired Analysis: Running for selected ROI: {selected_roi_option}")
        else:  # Should not happen if dropdown is populated correctly
            messagebox.showerror("Error", f"Invalid ROI selection '{selected_roi_option}'.")
            self.results_textbox.configure(state="disabled");
            return

        # --- Run Paired Analysis Loop ---
        # Add header to results box
        results_header = f"===== PAIRED COMPARISON RESULTS ('{cond_a}' vs '{cond_b}') =====\n"
        results_header += f"Metric: {metric} | ROI(s) Analyzed: {selected_roi_option}\n"  # Use the selection option
        results_header += "=" * (len(results_header.split('\n')[0]) + 4) + "\n\n"
        all_results_text = results_header

        any_analysis_ran = False
        any_test_completed = False
        processing_errors = []  # Collect errors during processing

        # Loop through ROIs and Frequencies
        for roi_name in rois_to_analyze:
            roi_header_added = False
            for freq in selected_freqs:
                params = {'frequency': freq, 'metric': metric, 'roi_name': roi_name, 'condition_a': cond_a,
                          'condition_b': cond_b}
                try:
                    any_analysis_ran = True
                    # Aggregate paired data
                    scores_a, scores_b, included_subjects = self.aggregate_data(metric, roi_name, freq, cond_a, cond_b)

                    # Check if aggregation was successful and returned data
                    if scores_a is not None and scores_b is not None and included_subjects:
                        # Perform the paired statistical test
                        stats_results = self.perform_paired_test(scores_a, scores_b)

                        # Prepare data for export table
                        export_data = params.copy()
                        export_data.update(stats_results)
                        export_data['Included_Subjects'] = ", ".join(included_subjects)
                        self.results_structured.append(export_data)  # Add to list for export

                        # Add ROI header only once if analyzing all ROIs and data exists
                        if selected_roi_option == ALL_ROIS_OPTION and not roi_header_added:
                            all_results_text += f"\n========== ROI: {roi_name} ==========\n"
                            roi_header_added = True

                        # Format and append results text
                        result_text = self.format_results_text(params, stats_results)
                        all_results_text += result_text

                        # Check if the test itself completed without internal error
                        if 'error' not in stats_results:
                            any_test_completed = True
                    else:
                        # Aggregation failed or returned no subjects
                        # Add a note to the output for this combination
                        if selected_roi_option == ALL_ROIS_OPTION and not roi_header_added:
                            all_results_text += f"\n========== ROI: {roi_name} ==========\n"
                            roi_header_added = True
                        all_results_text += f"--- Paired Analysis: {freq} ---\n"
                        all_results_text += "Result: No complete paired data found for analysis.\n"
                        all_results_text += "----------------------------------------\n"
                        # Optionally add an entry to export table indicating failure
                        error_entry = params.copy();
                        error_entry['error'] = "No paired data found"
                        self.results_structured.append(error_entry)

                except Exception as e:
                    # Catch unexpected errors during the loop for one combination
                    err_msg = f"!!! Paired Analysis CRITICAL Error: ROI='{roi_name}', Freq='{freq}'. Error: {e}"
                    self.log_to_main_app(f"{err_msg}\n{traceback.format_exc()}")
                    processing_errors.append(f"ROI: {roi_name}, Freq: {freq} -> {e}")
                    # Add error entry for export
                    error_entry = params.copy();
                    error_entry['error'] = f"Unexpected Processing Error: {e}"
                    self.results_structured.append(error_entry)

        # --- Display Final Results ---
        # Append any critical errors encountered during the loop
        if processing_errors:
            all_results_text += "\n\n===== PROCESSING ERRORS ENCOUNTERED =====\n"
            all_results_text += "\n".join(processing_errors)
            all_results_text += "\n=======================================\n"

        self.results_textbox.insert("1.0", all_results_text.strip())
        self.results_textbox.configure(state="disabled")  # Make read-only

        # Final status logging and export button enabling
        if any_test_completed:
            self.export_button.configure(state="normal")  # Enable paired export
            self.log_to_main_app("Paired comparison analysis complete.")
        elif any_analysis_ran:
            self.log_to_main_app(
                "Paired comparison analysis finished, but some tests may have failed or had insufficient data.")
        else:
            # This case might happen if validation passed but aggregation failed for all combinations
            self.log_to_main_app("Paired comparison analysis did not yield results (check aggregation logs/data).")

    def call_export_significance_results(self):
        """
        Gathers necessary data and calls the external function to export
        Significance Check results stored in self.significant_findings.
        """
        # Check the instance variable populated by run_significance_check
        if not self.significant_findings or not any(
                any(self.significant_findings[cond].values()) for cond in self.significant_findings):
            messagebox.showwarning("No Results", "No significant findings available from the last check to export.")
            self.log_to_main_app("Export Sig Check called, but no significant findings stored.")
            return

        metric = self.metric_var.get()
        try:
            # Read threshold value again for consistency, handle potential error
            threshold = float(self.significance_threshold_var.get())
        except ValueError:
            messagebox.showerror("Error", "Cannot export: Invalid Significance Threshold value in the entry field.")
            return
        parent_folder = self.stats_data_folder_var.get()

        self.log_to_main_app("Calling external function to export significance results...")
        # Call the function from the imported module
        stats_export.export_significance_results_to_excel(
            findings_dict=self.significant_findings,  # Pass the stored findings
            metric=metric,
            threshold=threshold,
            parent_folder=parent_folder,
            log_func=self.log_to_main_app  # Pass the logger method
        )

    def run_significance_check(self):
        """
        Gathers parameters, aggregates data for EACH condition separately,
        checks if mean metric exceeds threshold for all ROIs and selected frequencies,
        and displays only the significant findings in a structured format.
        Enables export for significance check results if any are found.
        Includes a warning for common artifact frequencies (6Hz, 12Hz).
        """
        self.log_to_main_app("Run Significance Check button clicked.")
        structured_significant_findings = {}  # Use local var for building results
        self.results_structured = []  # Clear paired results
        self.results_textbox.configure(state="normal")
        self.results_textbox.delete("1.0", tk.END)
        self.export_button.configure(state="disabled")  # Disable paired export
        self.export_sig_button.configure(state="disabled")  # Disable sig export initially

        # --- Get Parameters ---
        folder = self.stats_data_folder_var.get()
        metric = self.metric_var.get()
        selected_freqs = [freq for freq, var in self.freq_checkbox_vars.items() if var.get()]
        try:
            threshold = float(self.significance_threshold_var.get())
        except ValueError:
            messagebox.showerror("Input Error", "Invalid Significance Threshold. Please enter a number (e.g., 1.96).");
            self.results_textbox.configure(state="disabled");
            return

        # --- Validate Parameters ---
        if not folder or not os.path.isdir(folder): messagebox.showerror("Input Error",
                                                                         "Please select a valid data folder."); self.results_textbox.configure(
            state="disabled"); return
        if not selected_freqs: messagebox.showerror("Input Error",
                                                    "Please select at least one frequency."); self.results_textbox.configure(
            state="disabled"); return
        if not self.conditions_list: messagebox.showerror("Input Error",
                                                          "No conditions found. Please scan a valid folder first."); self.results_textbox.configure(
            state="disabled"); return

        # <<< --- ADDED WARNING CHECK --- >>>
        warning_freqs_str = ["6.0_Hz", "12.0_Hz"]
        selected_warning_freqs = [f for f in selected_freqs if f in warning_freqs_str]

        if selected_warning_freqs:
            freq_list_str = ", ".join(selected_warning_freqs).replace('_Hz', '') + " Hz"
            warning_message = (
                f"Warning: Frequency {freq_list_str} is selected.\n\n"
                "Responses at these frequencies can be strongly influenced by screen refresh rates "
                "or the fundamental image presentation rate (6 Hz) and may not reflect purely neural activity.\n\n"
                "Do you want to proceed with the Significance Check analysis including these frequencies?"
            )
            proceed = messagebox.askyesno("Frequency Warning (Significance Check)", warning_message, icon='warning')
            if not proceed:
                self.log_to_main_app(
                    f"Significance Check cancelled by user due to frequency warning ({freq_list_str}).")
                self.results_textbox.insert("1.0",
                                            "Analysis cancelled by user due to frequency warning.\nPlease deselect 6.0 Hz and/or 12.0 Hz if desired and run again.")
                self.results_textbox.configure(state="disabled")
                return  # Stop the analysis
            else:
                self.log_to_main_app(f"User chose to proceed despite frequency warning ({freq_list_str}).")
        # <<< --- END WARNING CHECK --- >>>

        # --- Run Significance Check Loop ---
        processing_errors = []  # Store any errors encountered

        # Initialize structure for findings
        for condition in self.conditions_list:
            structured_significant_findings[condition] = {}

        # Loop through Conditions -> ROIs -> Frequencies
        self.log_to_main_app(f"Starting Significance Check (Mean {metric} > {threshold:.2f})...")
        for condition in self.conditions_list:
            self.log_to_main_app(f"  Checking Condition: '{condition}'")
            for roi_name in ROIS.keys():  # Check all defined ROIs
                for freq in selected_freqs:
                    try:
                        # Aggregate data for the single condition
                        scores, _, included_subjects = self.aggregate_data(metric, roi_name, freq, condition,
                                                                           None)  # cond_b=None

                        # Check if aggregation successful and data exists
                        if scores is not None and len(scores) > 0:
                            mean_score = np.mean(scores)
                            n_subs = len(scores)

                            # --- Perform the Significance Check ---
                            # Simple check: is the mean score greater than the threshold?
                            is_significant = mean_score > threshold

                            # (Optional: Implement one-sample t-test here if desired for rigor)

                            if is_significant:
                                # Initialize ROI dict if first finding for this ROI in this condition
                                if roi_name not in structured_significant_findings[condition]:
                                    structured_significant_findings[condition][roi_name] = []

                                # Store finding details (using dynamic key for metric)
                                metric_key = f'Mean_{metric.replace(" ", "_")}'
                                finding = {
                                    'Frequency': freq,
                                    metric_key: mean_score,
                                    'N': n_subs,
                                    'Threshold': threshold  # Store for context
                                    # Add t-stat, p-value here if using t-test option
                                }
                                structured_significant_findings[condition][roi_name].append(finding)

                        # else: No data aggregated or N=0 for this combo

                    except Exception as e:
                        # Catch unexpected errors during loop for one combination
                        err_msg = f"!!! Sig Check CRITICAL Error: Cond='{condition}', ROI='{roi_name}', Freq='{freq}'. Error: {e}"
                        self.log_to_main_app(f"{err_msg}\n{traceback.format_exc()}")
                        processing_errors.append(f"Cond: {condition}, ROI: {roi_name}, Freq: {freq} -> {e}")
                        # Continue to next iteration

        # --- Format and Display Results ---
        # Use the dedicated formatting function
        all_results_text = self.format_significance_results(structured_significant_findings, metric, threshold)

        # Append any critical errors encountered during the loop
        if processing_errors:
            all_results_text += "\n\n===== PROCESSING ERRORS ENCOUNTERED =====\n"
            all_results_text += "\n".join(processing_errors)
            all_results_text += "\n=======================================\n"

        # Display in the textbox
        self.results_textbox.insert("1.0", all_results_text)
        self.results_textbox.configure(state="disabled")  # Make read-only

        # --- Enable Export Button if Significant Findings Exist ---
        # Check if the dictionary contains any actual findings across all conditions/ROIs
        any_significant = any(
            any(structured_significant_findings[cond].values()) for cond in structured_significant_findings)
        if any_significant:
            self.export_sig_button.configure(state="normal")
            # Store findings dict in instance variable for the export function to access
            self.significant_findings = structured_significant_findings
            self.log_to_main_app("Significance check complete. Significant findings found.")
        else:
            self.significant_findings = {}  # Ensure it's empty if none found
            self.log_to_main_app("Significance check complete. No findings exceeded the threshold.")


    # ==============================================================
    # Export Function (Only for Paired Results)
    # ==============================================================
    def export_results(self):
        """
        Exports the structured PAIRED comparison results (from self.results_structured)
        to an Excel file using xlsxwriter for formatting.
        """
        if not self.results_structured:
            messagebox.showwarning("No Paired Results",
                                   "No paired comparison results available to export. Please run the 'Run Paired Comparison' analysis first.")
            return

        # Suggest filename based on the parameters of the paired test run
        initial_dir = self.stats_data_folder_var.get() or os.path.expanduser("~")
        # Safely get params from the first result entry, provide defaults
        first_result = self.results_structured[0] if self.results_structured else {}
        cond_a = first_result.get('condition_a', 'CondA').replace(" ", "_").replace(os.sep, "-")
        cond_b = first_result.get('condition_b', 'CondB').replace(" ", "_").replace(os.sep, "-")
        roi_selection = self.roi_var.get()  # Use the ROI setting from the GUI during the paired run
        roi_part = roi_selection if roi_selection != ALL_ROIS_OPTION else "AllROIs"
        metric = first_result.get('metric', 'Metric').replace("-", "").replace(" ", "")
        suggested_filename = f"Stats_Paired_{metric}_{roi_part}_{cond_a}_vs_{cond_b}.xlsx"

        # Ask user for save location
        save_path = filedialog.asksaveasfilename(
            title="Save Paired Statistical Results As",  # Explicit title
            initialdir=initial_dir,
            initialfile=suggested_filename,
            defaultextension=".xlsx",
            filetypes=[("Excel Workbook", "*.xlsx"), ("All Files", "*.*")]
        )
        if not save_path:
            self.log_to_main_app("Paired results export cancelled.");
            return

        try:
            self.log_to_main_app(f"Exporting paired results to: {save_path}")
            df_export = pd.DataFrame(self.results_structured)

            # Check if dataframe is empty after creation (e.g., only error entries existed)
            if df_export.empty:
                messagebox.showwarning("Empty Results", "The results table for paired export is empty.")
                return

            # Define preferred column order for export
            cols_order = [
                'roi_name', 'frequency', 'metric', 'condition_a', 'condition_b', 'N',
                'Mean_A', 'SD_A', 'Median_A', 'IQR_A', 'Mean_B', 'SD_B', 'Median_B', 'IQR_B',
                'Shapiro_W', 'Shapiro_p', 'Normality_Decision', 'Shapiro_Note',
                'Test_Used', 'Statistic_Type', 'Statistic_Value', 'df', 'p_value', 'Wilcoxon_Note',
                'Effect_Size_Type', 'Effect_Size_Value',
                'Included_Subjects', 'error'
            ]
            # Filter to only include columns actually present in the dataframe
            cols_to_export = [col for col in cols_order if col in df_export.columns]
            if not cols_to_export:
                self.log_to_main_app("Export Error: No valid columns found in paired results data.")
                messagebox.showerror("Export Error", "No valid columns found in the paired results data to export.")
                return
            df_export = df_export[cols_to_export]  # Reorder dataframe with existing columns

            # Write to Excel using xlsxwriter engine for formatting
            with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
                df_export.to_excel(writer, sheet_name='Paired Comparison Results', index=False)
                workbook = writer.book
                worksheet = writer.sheets['Paired Comparison Results']

                # Define cell formats (add centering here if desired based on user pref)
                # Example with centering: fmt_num_3dp = workbook.add_format({'num_format': '0.000', 'align': 'center'})
                fmt_num_3dp = workbook.add_format({'num_format': '0.000'})
                fmt_num_2dp = workbook.add_format({'num_format': '0.00'})
                fmt_num_1dp = workbook.add_format({'num_format': '0.0'})
                fmt_wrap = workbook.add_format(
                    {'text_wrap': True, 'valign': 'top'})  # Vertical align top for wrapped text

                # Column types for formatting
                float_cols_3dp = ['Shapiro_W', 'Shapiro_p', 'p_value', 'Mean_A', 'SD_A', 'Mean_B', 'SD_B', 'Median_A',
                                  'IQR_A', 'Median_B', 'IQR_B']
                float_cols_2dp = ['Statistic_Value', 'Effect_Size_Value']
                text_cols = ['Included_Subjects', 'error', 'Wilcoxon_Note', 'Shapiro_Note', 'Normality_Decision',
                             'Test_Used', 'Statistic_Type', 'Effect_Size_Type', 'roi_name', 'frequency', 'metric',
                             'condition_a', 'condition_b']  # Added text columns for width calc

                # Apply formatting and auto-adjust column widths
                for col_idx, col_name in enumerate(df_export.columns):
                    # Calculate width based on data and header length
                    try:
                        max_len_data = df_export[col_name].astype(str).map(len).max()
                        # Handle case where max() returns NaN for empty or all-NaN column
                        if pd.isna(max_len_data): max_len_data = 0
                        # Ensure header length is considered, add padding
                        col_width = max(int(max_len_data), len(col_name)) + 2
                    except Exception:
                        col_width = len(col_name) + 5  # Fallback width

                    # Determine appropriate format
                    col_format = None
                    min_width = 10  # Default minimum width

                    if col_name in float_cols_3dp:
                        col_format = fmt_num_3dp; min_width = 12
                    elif col_name in float_cols_2dp:
                        col_format = fmt_num_2dp; min_width = 12
                    # Special case for Wilcoxon W statistic (usually integer or .0)
                    elif col_name == 'Statistic_Value' and 'Test_Used' in df_export.columns and df_export[
                        'Test_Used'].str.contains('Wilcoxon').any():
                        col_format = fmt_num_1dp;
                        min_width = 10
                    elif col_name in text_cols:
                        col_format = fmt_wrap
                        # Set wider minimum for certain text columns
                        if col_name in ['Included_Subjects', 'error', 'Shapiro_Note', 'Wilcoxon_Note']:
                            min_width = 30
                        else:
                            min_width = 15  # For shorter text fields like Condition names

                    # Ensure final width meets minimum
                    col_width = max(col_width, min_width)

                    # Apply format and width
                    worksheet.set_column(col_idx, col_idx, col_width, col_format)

            self.log_to_main_app("Paired results export successful.")
            messagebox.showinfo("Export Successful", f"Paired comparison results exported to:\n{save_path}")

        except PermissionError as e:
            self.log_to_main_app(
                f"!!! Export Failed: Permission denied writing to {save_path}. Check file/folder permissions.")
            messagebox.showerror("Export Failed",
                                 f"Permission denied writing file:\n{save_path}\nClose the file if it's open, or choose a different location.")
        except Exception as e:
            self.log_to_main_app(f"!!! Export Failed: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Export Failed", f"Could not save Excel file:\n{e}")


# ==============================================================
# Example Usage (if running this file directly for testing)
# ==============================================================
if __name__ == "__main__":
    # This allows testing the StatsAnalysisWindow independently
    # Create a dummy root window (required by CTkToplevel)
    root = ctk.CTk()
    root.title("Main App Window (Test Host)")
    root.geometry("300x150")

    # Define a simple log function for testing if main app isn't available
    def test_log(message):
        timestamp = pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"{timestamp} {message}")

    # Create a dummy master object that mimics having a 'log' method
    # Add other attributes/methods if StatsAnalysisWindow relies on them from master
    class TestMaster:
        # Mimic the log method
        log = test_log
        # Example: add other attributes if needed
        # some_other_attribute = "value"
        pass

    test_master_obj = TestMaster() # Use this if StatsAnalysisWindow needs methods from master

    def launch_stats():
        # Set an optional default path for testing convenience
        test_path = "" # e.g., "C:/Users/YourUser/Documents/FPVS_Processed_Data"
        try:
             # Pass the actual root window as the Tkinter master
             # StatsAnalysisWindow internally uses self.master_app for logging now
             stats_win = StatsAnalysisWindow(master=root, default_folder=test_path)
             # stats_win.grab_set() # Typically disable grab_set during basic testing
        except Exception as e:
             print(f"Error launching Stats window: {e}\n{traceback.format_exc()}")

    # Add a button to the dummy root window to launch the stats tool
    button = ctk.CTkButton(root, text="Open Stats Tool", command=launch_stats)
    button.pack(pady=20, padx=20)

    # Run the dummy main loop
    root.mainloop()