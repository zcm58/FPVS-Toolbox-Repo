# Tools/Stats/stats.py
# -*- coding: utf-8 -*-
"""
Provides a Toplevel window for statistical analysis of FPVS results.
Orchestrates UI (via stats_ui_builder), data handling, analysis,
and outlier review (via stats_qc_outliers).

Handles most of the statistical analysis after processing data via the main FPVS Toolbox window.

"""

import os
import glob
import re
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Optional, List, Dict, Set # For type hinting

# Import new local modules
from .stats_ui_builder import build_stats_interface
from .stats_qc_outliers import QualityOutlierReviewFrame
# QUALITY_FLAGS_FILENAME will be imported from main config by stats_qc_outliers.py

# Import constants from the main project config file
# Adjust the path if your Tools/Stats/ directory is nested differently relative to the main config.py
# Assuming config.py is in the 'src' directory, and 'src' is in sys.path or you handle it with your project structure.
# If config.py is at the same level as fpvs_app.py, and fpvs_app.py imports Tools.Stats,
# then from ..config import ... might be needed if Tools.Stats is a true subpackage.
# Or, if 'src' is your root, 'from config import ...' should work.
try:
    from config import (
        ROIS, ALL_ROIS_OPTION, HARMONIC_CHECK_ALPHA,
        STATS_PLACEHOLDER_SCAN_FOLDER, STATS_PLACEHOLDER_NO_OTHER_CONDITIONS,
        STATS_PLACEHOLDER_SELECT_CONDITION_A,
        # TARGET_FREQUENCIES # No longer needed directly in stats.py for UI
    )
except ImportError as e:
    # Fallbacks if config.py is not found or missing these specific constants
    # This makes stats.py potentially runnable standalone for testing, but ideally config is present
    print(f"Warning: Could not import all constants from main config.py for Stats tool: {e}. Using defaults.")
    ROIS = { "Frontal Lobe": ["F3", "F4", "Fz"], "Occipital Lobe": ["O1", "O2", "Oz"],
             "Parietal Lobe": ["P3", "P4", "Pz"], "Central Lobe": ["C3", "C4", "Cz"] }
    ALL_ROIS_OPTION = "(All ROIs)"
    HARMONIC_CHECK_ALPHA = 0.05
    STATS_PLACEHOLDER_SCAN_FOLDER = "(Scan Folder)"
    STATS_PLACEHOLDER_NO_OTHER_CONDITIONS = "(No other conditions)"
    STATS_PLACEHOLDER_SELECT_CONDITION_A = "(Select Condition A)"
    # TARGET_FREQUENCIES = np.round(np.arange(1.2, 17.0, 1.2), 1)


class StatsAnalysisWindow(ctk.CTkToplevel):
    def __init__(self, master, default_folder=""):
        if not isinstance(master, (tk.Tk, tk.Toplevel, ctk.CTk, ctk.CTkToplevel)):
            raise TypeError("Master must be a Tkinter root or Toplevel window.")
        super().__init__(master)

        self.title("FPVS Statistical Analysis Tool")
        self.geometry("725x900")  # Adjusted for outlier frame
        self.grab_set();
        self.lift();
        self.focus_force()

        self.master_app = master  # For logging via main app's logger if available
        self.default_output_folder = default_folder

        # Core data storage
        self.subject_data: Dict[str, Dict[str, str]] = {}  # {pid: {condition: file_path}}
        self.all_subject_data: Dict[str, Dict[str, Dict[str, float]]] = {}  # {pid: {condition: {roi: sum_bca}}}
        self.conditions_list: List[str] = []
        self.subjects_list: List[str] = []

        # Results storage for export (structured data)
        self.paired_tests_results_data: List[Dict] = []
        self.rm_anova_results_data: Optional[pd.DataFrame] = None
        self.harmonic_check_results_data: List[Dict] = []

        # UI State Variables (to be used by stats_ui_builder)
        self.stats_data_folder_var = tk.StringVar(master=self, value=self.default_output_folder)
        self.detected_info_var = tk.StringVar(master=self, value="Select folder with FPVS Excel results.")
        self.base_freq_var = tk.StringVar(master=self, value="6.0")  # For frequency exclusions
        self.roi_var = tk.StringVar(master=self, value=ALL_ROIS_OPTION)  # For Paired/ANOVA ROI selection
        self.condition_A_var = tk.StringVar(master=self, value=STATS_PLACEHOLDER_SCAN_FOLDER)
        self.condition_B_var = tk.StringVar(master=self, value=STATS_PLACEHOLDER_SCAN_FOLDER)
        # self.freq_checkbox_vars = {} # REMOVED - No longer using manual frequency checkboxes
        self.harmonic_metric_var = tk.StringVar(master=self, value="SNR")  # For Per-Harmonic Check
        self.harmonic_threshold_var = tk.StringVar(master=self, value="1.96")  # For Per-Harmonic Check

        # References to important widgets (will be set by stats_ui_builder)
        self.results_textbox: Optional[ctk.CTkTextbox] = None
        self.export_paired_tests_btn: Optional[ctk.CTkButton] = None
        self.export_rm_anova_btn: Optional[ctk.CTkButton] = None
        self.export_harmonic_check_btn: Optional[ctk.CTkButton] = None
        self.cond_A_menu: Optional[ctk.CTkOptionMenu] = None
        self.cond_B_menu: Optional[ctk.CTkOptionMenu] = None
        self.roi_menu: Optional[ctk.CTkOptionMenu] = None
        self.harmonic_metric_menu: Optional[ctk.CTkOptionMenu] = None
        # self.freq_scrollable_frame = None # REMOVED

        # Outlier Management Frame (will be instantiated by stats_ui_builder)
        self.qc_outlier_frame: Optional[QualityOutlierReviewFrame] = None

        self.create_stats_widgets()  # This will call the external builder

        if self.default_output_folder and os.path.isdir(self.default_output_folder):
            self.scan_folder()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def log_to_main_app(self, message: str):
        try:
            if hasattr(self.master_app, 'log') and callable(self.master_app.log):
                self.master_app.log(f"[StatsTool] {message}")
            else:
                print(f"[StatsTool] {message}")
        except Exception as e:
            print(f"[StatsTool Log Error] {e} | Original message: {message}")

    def on_close(self):
        self.log_to_main_app("Closing Stats Analysis window.")
        self.grab_release()
        self.destroy()

    def _validate_numeric(self, P_value: str) -> bool:
        if P_value in ("", "-"): return True
        try:
            float(P_value); return True
        except ValueError:
            return False

    def create_stats_widgets(self):
        """Delegates UI construction to the stats_ui_builder module."""
        self.log_to_main_app("Building stats UI...")
        build_stats_interface(self)  # Pass self (StatsAnalysisWindow instance)
        self.log_to_main_app("Stats UI build complete.")
        # Any post-build UI adjustments can go here if necessary (e.g., initial population)
        if hasattr(self, 'update_condition_B_options'):  # Ensure menus are updated if needed
            self.update_condition_B_options()

    # --- UI Callbacks & File/Folder Handling ---
    def browse_folder(self):
        current_folder = self.stats_data_folder_var.get()
        initial_dir = current_folder if os.path.isdir(current_folder) else os.path.expanduser("~")
        folder = filedialog.askdirectory(
            title="Select Parent Folder (with Condition Subfolders containing Excel files)", initialdir=initial_dir)
        if folder:
            self.stats_data_folder_var.set(folder)
            self.scan_folder()  # This will now also trigger loading quality flags
        else:
            self.log_to_main_app("Folder selection cancelled.")

    def scan_folder(self):
        parent_folder = self.stats_data_folder_var.get()
        if not parent_folder or not os.path.isdir(parent_folder):
            self.detected_info_var.set("Invalid parent folder selected.")
            self.update_condition_menus([])
            if self.qc_outlier_frame: self.qc_outlier_frame._clear_display()
            return

        self.log_to_main_app(f"Scanning for Excel files in subfolders of: {parent_folder}")
        subjects_set = set()
        conditions_set = set()
        self.subject_data.clear()
        pid_pattern = re.compile(r"(?:[a-zA-Z_]*?)?(P\d+).*\.xlsx$", re.IGNORECASE)

        try:
            for item_name in os.listdir(parent_folder):
                item_path = os.path.join(parent_folder, item_name)
                if os.path.isdir(item_path):
                    condition_name_raw = item_name
                    condition_name = re.sub(r'^\d+\s*[-_]*\s*', '', condition_name_raw).strip()
                    if not condition_name: continue
                    files_in_subfolder = glob.glob(os.path.join(item_path, "*.xlsx"))
                    for f_path in files_in_subfolder:
                        excel_filename = os.path.basename(f_path)
                        pid_match = pid_pattern.search(excel_filename)
                        if pid_match:
                            pid = pid_match.group(1).upper()
                            subjects_set.add(pid)
                            conditions_set.add(condition_name)
                            self.subject_data.setdefault(pid, {})[condition_name] = f_path
        except Exception as e:
            self.log_to_main_app(f"!!! Error scanning folder structure: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Scanning Error", f"An unexpected error occurred: {e}")
            self.update_condition_menus([])
            if self.qc_outlier_frame: self.qc_outlier_frame._clear_display()
            return

        self.subjects_list = sorted(list(subjects_set))
        self.conditions_list = sorted(list(conditions_set))
        info_text = (f"Scan: Found {len(self.subjects_list)} subjects and {len(self.conditions_list)} conditions."
                     if self.subjects_list and self.conditions_list
                     else "Scan: No valid PIDs or conditions found from Excel files.")
        self.log_to_main_app(info_text)
        if self.subjects_list: self.log_to_main_app(f"Detected PIDs: {', '.join(self.subjects_list)}")
        if self.conditions_list: self.log_to_main_app(f"Detected Conditions: {', '.join(self.conditions_list)}")
        self.detected_info_var.set(info_text)

        self.update_condition_menus(self.conditions_list)

        self.all_subject_data.clear()
        self.paired_tests_results_data.clear()
        self.rm_anova_results_data = None
        self.harmonic_check_results_data.clear()

        if self.qc_outlier_frame:  # Check if the frame has been initialized
            self.qc_outlier_frame.load_and_display_flags()
        else:
            self.log_to_main_app("Warning: FPVS Data Outlier Frame not available to load flags.")

    def update_condition_menus(self, conditions_list: List[str]):
        current_a = self.condition_A_var.get()
        display_list = conditions_list if conditions_list else [PLACEHOLDER_SCAN_FOLDER]

        if self.cond_A_menu:
            self.cond_A_menu.configure(values=display_list)
            if current_a not in display_list or (
                    current_a == PLACEHOLDER_SCAN_FOLDER and display_list[0] != PLACEHOLDER_SCAN_FOLDER):
                self.condition_A_var.set(display_list[0])
            elif not display_list or display_list[
                0] == PLACEHOLDER_SCAN_FOLDER:  # if display_list is just ["(Scan Folder)"]
                self.condition_A_var.set(PLACEHOLDER_SCAN_FOLDER)
        self.update_condition_B_options()

    def update_condition_B_options(self, *args):
        cond_a = self.condition_A_var.get()
        valid_b_options = [c for c in self.conditions_list if c and c != cond_a and c != PLACEHOLDER_SCAN_FOLDER]

        if not self.conditions_list or cond_a == STATS_PLACEHOLDER_SCAN_FOLDER:
            display_b_list = [STATS_PLACEHOLDER_SELECT_CONDITION_A]
        elif not valid_b_options:
            display_b_list = [STATS_PLACEHOLDER_NO_OTHER_CONDITIONS]
        else:
            display_b_list = valid_b_options

        current_b_val = self.condition_B_var.get()
        if self.cond_B_menu:
            self.cond_B_menu.configure(values=display_b_list)
            if current_b_val not in display_b_list or current_b_val == cond_a:  # Reset B if current is invalid or same as A
                self.condition_B_var.set(display_b_list[0] if display_b_list else "")

    # --- Frequency Checkbox methods REMOVED ---
    # def populate_frequency_checkboxes(self): ... REMOVED
    # def select_all_freqs(self, select_state=True): ... REMOVED
    # def deselect_all_freqs(self): ... REMOVED

    def _get_excluded_pids(self) -> Set[str]:
        """Gets the set of PIDs to exclude from the QC outlier frame."""
        if self.qc_outlier_frame:
            return self.qc_outlier_frame.get_pids_to_exclude()
        self.log_to_main_app("Warning: qc_outlier_frame not available to get excluded PIDs.")
        return set()

    # --- Data Aggregation (To be potentially refactored later) ---
    def _get_included_freqs(self, all_col_names: List[str]) -> List[float]:
        try:
            base_freq_val = float(self.base_freq_var.get())
            if base_freq_val <= 0: raise ValueError("Base frequency must be positive.")
        except ValueError as e:
            self.log_to_main_app(f"Error: Invalid Base Frequency '{self.base_freq_var.get()}': {e}")
            return []
        numeric_freqs = []
        for col_name in all_col_names:
            if isinstance(col_name, str) and col_name.endswith('_Hz'):
                try:
                    numeric_freqs.append(float(col_name[:-3]))
                except ValueError:
                    self.log_to_main_app(f"Could not parse freq from col: {col_name}")
        if not numeric_freqs: self.log_to_main_app(
            "No numeric frequency columns (e.g. X.Y_Hz) found in data."); return []

        sorted_numeric_freqs = sorted(list(set(numeric_freqs)))
        excluded_freqs = {f_val for f_val in sorted_numeric_freqs if
                          abs(f_val / base_freq_val - round(f_val / base_freq_val)) < 1e-6}
        included_freqs = [f_val for f_val in sorted_numeric_freqs if f_val not in excluded_freqs]
        # self.log_to_main_app(f"Base: {base_freq_val}, All detected freqs: {sorted_numeric_freqs}, Excluded from analysis: {sorted(list(excluded_freqs))}, Included for analysis: {included_freqs}")
        return included_freqs

    def aggregate_bca_sum(self, file_path: str, roi_name: str) -> float:
        try:
            df = pd.read_excel(file_path, sheet_name="BCA (uV)", index_col="Electrode")
            roi_channels = ROIS.get(roi_name)
            if not roi_channels: self.log_to_main_app(f"ROI '{roi_name}' not defined."); return np.nan
            df_roi = df.reindex(roi_channels).dropna(how='all')
            if df_roi.empty: self.log_to_main_app(
                f"No data for ROI '{roi_name}' in {os.path.basename(file_path)}."); return np.nan

            included_freq_values = self._get_included_freqs(df.columns)
            if not included_freq_values: self.log_to_main_app(
                f"No frequencies to include for BCA sum in {os.path.basename(file_path)} for ROI {roi_name}."); return np.nan

            cols_to_sum = [f"{f:.1f}_Hz" for f in included_freq_values if f"{f:.1f}_Hz" in df_roi.columns]
            if not cols_to_sum: self.log_to_main_app(
                f"No matching BCA columns to sum for ROI {roi_name} in {os.path.basename(file_path)}."); return np.nan

            return df_roi[cols_to_sum].sum(axis=1).mean()
        except FileNotFoundError:
            self.log_to_main_app(f"Error: File not found {file_path}"); return np.nan
        except Exception as e:
            self.log_to_main_app(f"Error aggregating BCA for {os.path.basename(file_path)}, ROI {roi_name}: {e}")
            return np.nan

    def prepare_all_subject_summed_bca_data(self, excluded_pids: Optional[Set[str]] = None):
        self.log_to_main_app("Preparing summed BCA data...")
        self.all_subject_data.clear()
        if excluded_pids is None: excluded_pids = set()

        if not self.subjects_list:
            messagebox.showwarning("Data Error", "No subjects found from scan. Cannot prepare summed BCA data.")
            self.log_to_main_app("Summed BCA prep failed: No subjects in subjects_list.")
            return False

        active_subjects = [pid for pid in self.subjects_list if pid not in excluded_pids]
        if not active_subjects:
            messagebox.showwarning("Data Error", "No subjects remaining after exclusions for Summed BCA prep.")
            self.log_to_main_app(f"Summed BCA prep failed: No subjects after excluding {len(excluded_pids)} PIDs.")
            return False

        self.log_to_main_app(
            f"Preparing Summed BCA for {len(active_subjects)} subjects. (Excluded: {len(excluded_pids)} PIDs: {sorted(list(excluded_pids)) if excluded_pids else 'None'})")

        for pid in active_subjects:
            self.all_subject_data[pid] = {}
            for cond_name in self.conditions_list:
                file_path = self.subject_data.get(pid, {}).get(cond_name)
                self.all_subject_data[pid].setdefault(cond_name, {})
                for roi_name in ROIS.keys():
                    sum_val = self.aggregate_bca_sum(file_path, roi_name) if file_path and os.path.exists(
                        file_path) else np.nan
                    self.all_subject_data[pid][cond_name][roi_name] = sum_val
        self.log_to_main_app("Summed BCA data preparation complete.")
        return True

    # --- Analysis Execution Functions (modified for exclusions) ---
    def run_paired_tests(self):
        self.log_to_main_app("Running Paired Tests (Summed BCA)...")
        if not self.results_textbox: self.log_to_main_app("Results textbox not initialized!"); return  # Safeguard
        self.results_textbox.configure(state="normal");
        self.results_textbox.delete("1.0", tk.END)
        if self.export_paired_tests_btn: self.export_paired_tests_btn.configure(state="disabled")
        self.paired_tests_results_data.clear()

        excluded_pids = self._get_excluded_pids()

        cond_a = self.condition_A_var.get()
        cond_b = self.condition_B_var.get()

        if not (cond_a and cond_a != PLACEHOLDER_SCAN_FOLDER and \
                cond_b and cond_b not in [PLACEHOLDER_SCAN_FOLDER, PLACEHOLDER_NO_OTHER_CONDITIONS,
                                          PLACEHOLDER_SELECT_CONDITION_A] and \
                cond_a != cond_b):
            messagebox.showerror("Input Error", "Please select two different and valid conditions for paired tests.")
            self.results_textbox.configure(state="disabled");
            return

        if not self.prepare_all_subject_summed_bca_data(excluded_pids=excluded_pids):
            self.results_textbox.configure(state="disabled");
            return

        if not self.all_subject_data:
            messagebox.showerror("Data Error",
                                 "No summed BCA data available after preparation/exclusions for Paired Tests.")
            self.results_textbox.configure(state="disabled");
            return

        rois_to_analyze_str = self.roi_var.get()
        rois_to_analyze = list(ROIS.keys()) if rois_to_analyze_str == ALL_ROIS_OPTION else (
            [rois_to_analyze_str] if rois_to_analyze_str in ROIS else [])
        if not rois_to_analyze:
            messagebox.showerror("Input Error", f"Invalid ROI selected for Paired Tests: {rois_to_analyze_str}");
            self.results_textbox.configure(state="disabled");
            return

        output_text = f"============================================================\n"
        output_text += f"              Paired t-tests (Summed BCA)\n"
        output_text += f"============================================================\n\n"
        output_text += f"Comparing Condition A: '{cond_a}'\n"
        output_text += f"With Condition B:      '{cond_b}'\n"
        output_text += f"ROIs Analyzed:         {rois_to_analyze_str}\n"
        if excluded_pids:
            output_text += f"Excluded PIDs (from Quality Review): {', '.join(sorted(list(excluded_pids)))}\n"
        output_text += f"(Significance level for p-values: p < {HARMONIC_CHECK_ALPHA})\n\n"  # Using same alpha for consistency

        significant_tests_found_overall = False

        for roi_name in rois_to_analyze:
            output_text += f"--- ROI: {roi_name} ---\n"
            vals_a, vals_b = [], []
            # self.all_subject_data keys are PIDs *not* in excluded_pids due to prepare_all_subject_summed_bca_data
            for pid in self.all_subject_data.keys():
                val_a = self.all_subject_data.get(pid, {}).get(cond_a, {}).get(roi_name, np.nan)
                val_b = self.all_subject_data.get(pid, {}).get(cond_b, {}).get(roi_name, np.nan)
                if not (pd.isna(val_a) or pd.isna(val_b)):
                    vals_a.append(val_a);
                    vals_b.append(val_b)

            n_pairs = len(vals_a)
            if n_pairs < 3:
                output_text += f"  Insufficient paired data (N={n_pairs} after exclusions). Minimum 3 pairs required.\n\n";
                continue
            try:
                t_stat, p_value_raw = stats.ttest_rel(vals_a, vals_b);
                df_val = n_pairs - 1
                mean_a, mean_b = np.mean(vals_a), np.mean(vals_b)
                mean_diff = mean_a - mean_b
                is_significant = p_value_raw < HARMONIC_CHECK_ALPHA
                p_value_str = "< .0001" if p_value_raw < 0.0001 else f"{p_value_raw:.4f}"

                output_text += f"  FINDING: {'SIGNIFICANT DIFFERENCE' if is_significant else 'NO SIGNIFICANT DIFFERENCE'} found.\n"
                output_text += f"    - Average Summed BCA for '{cond_a}': {mean_a:.3f} (approx. uV)\n"
                output_text += f"    - Average Summed BCA for '{cond_b}': {mean_b:.3f} (approx. uV)\n"
                output_text += f"    - Mean Difference ('{cond_a}' - '{cond_b}'): {mean_diff:.3f} (approx. uV)\n"
                if is_significant:
                    significant_tests_found_overall = True
                    if mean_diff > 1e-9:
                        output_text += f"      Interpretation: On average, '{cond_a}' showed significantly HIGHER Summed BCA than '{cond_b}'.\n"
                    elif mean_diff < -1e-9:
                        output_text += f"      Interpretation: On average, '{cond_b}' showed significantly HIGHER Summed BCA than '{cond_a}'.\n"
                    else:
                        output_text += f"      Interpretation: A significant difference found, but average values are very close.\n"
                    self.paired_tests_results_data.append(
                        {'ROI': roi_name, 'Condition_A': cond_a, 'Condition_B': cond_b,
                         'N_Pairs': n_pairs, 'Mean_A': mean_a, 'Mean_B': mean_b, 'Mean_Difference': mean_diff,
                         't_statistic': t_stat, 'df': df_val, 'p_value': p_value_raw})
                else:
                    output_text += f"      Interpretation: The observed difference in averages was not statistically significant.\n"
                output_text += f"    - Statistics: t({df_val}) = {t_stat:.2f}, p-value = {p_value_str}\n"
                output_text += f"    - Number of pairs included in this test: N = {n_pairs}\n\n"
            except Exception as e:
                output_text += f"  Error performing t-test for this ROI: {e}\n\n"; self.log_to_main_app(
                    f"Paired T-test error ROI {roi_name}: {e}")

        if not self.paired_tests_results_data:
            output_text += "------------------------------------------------------------\n"
            output_text += "No statistically significant differences (p < 0.05) were found for any analyzed ROIs with sufficient data.\n"
        else:
            output_text += "------------------------------------------------------------\n"
            output_text += "Tip: For a detailed table of significant findings, please use the 'Export Paired Results' feature.\n"
        output_text += "============================================================\n"
        self.results_textbox.insert("1.0", output_text)
        if self.paired_tests_results_data and self.export_paired_tests_btn: self.export_paired_tests_btn.configure(
            state="normal")
        self.results_textbox.configure(state="disabled")
        self.log_to_main_app("Paired tests (Summed BCA) run complete.")

    def run_rm_anova(self):
        # ... (Keep the version of run_rm_anova I provided last,
        #      which includes the detailed explanation and p-value formatting.
        #      Ensure it also starts with `excluded_pids = self._get_excluded_pids()`
        #      and uses it in `prepare_all_subject_summed_bca_data(excluded_pids=excluded_pids)`.
        #      The df_long preparation should also respect these exclusions implicitly if
        #      self.all_subject_data is already filtered.
        #      Add the Excluded PIDs line to its header.
        #      The N reported should be `df_long['subject'].nunique()`.
        #      This method is long, so I'm not repeating its entire improved text output here,
        #      but the exclusion logic must be integrated at its start.)

        self.log_to_main_app("Running RM-ANOVA (Summed BCA)...")
        if not self.results_textbox: self.log_to_main_app("Results textbox not initialized!"); return
        self.results_textbox.configure(state="normal");
        self.results_textbox.delete("1.0", tk.END)
        if self.export_rm_anova_btn: self.export_rm_anova_btn.configure(state="disabled")
        self.rm_anova_results_data = None

        excluded_pids = self._get_excluded_pids()

        if not self.prepare_all_subject_summed_bca_data(excluded_pids=excluded_pids):
            self.results_textbox.configure(state="disabled");
            return
        if not self.all_subject_data:
            messagebox.showerror("Data Error", "No summed BCA data for RM-ANOVA after prep/exclusions.")
            self.results_textbox.configure(state="disabled");
            return

        long_format_data = []
        for pid, cond_data in self.all_subject_data.items():  # self.all_subject_data now only has included PIDs
            for cond_name, roi_data in cond_data.items():
                for roi_name, value in roi_data.items():
                    if not pd.isna(value):
                        long_format_data.append(
                            {'subject': pid, 'condition': cond_name, 'roi': roi_name, 'value': value})

        if not long_format_data:
            messagebox.showerror("Data Error", "No valid (non-NaN) data entries for RM-ANOVA from included subjects.");
            self.results_textbox.configure(state="disabled");
            return

        df_long = pd.DataFrame(long_format_data)
        num_subjects_for_anova = df_long['subject'].nunique()  # N based on actual data in df_long

        if df_long['condition'].nunique() < 2 or df_long['roi'].nunique() < 1:
            messagebox.showerror("Data Error",
                                 "RM-ANOVA requires at least two conditions and one ROI with data from included subjects.")
            self.results_textbox.configure(state="disabled");
            return

        output_text = f"============================================================\n"
        output_text += f"       Repeated Measures ANOVA (RM-ANOVA) Results\n"
        output_text += f"       Analysis conducted on: Summed BCA Data\n"
        if excluded_pids:
            output_text += f"Excluded PIDs (Quality Flags): {', '.join(sorted(list(excluded_pids)))}\n"
        output_text += f"Number of subjects included in this ANOVA: {num_subjects_for_anova}\n"  # N subjects
        output_text += f"============================================================\n\n"
        output_text += ("This test examines the overall effects of your experimental conditions,\n"
                        "the different brain regions (ROIs), and their interaction.\n\n")

        try:
            self.log_to_main_app(f"Calling run_repeated_measures_anova with DataFrame of shape: {df_long.shape}")
            anova_df_results = run_repeated_measures_anova(data=df_long, dv_col='value',
                                                           within_cols=['condition', 'roi'], subject_col='subject')
            if anova_df_results is not None and not anova_df_results.empty:
                output_text += "------------------------------------------------------------\n"
                output_text += "                 STATISTICAL TABLE (RM-ANOVA)\n"
                output_text += "------------------------------------------------------------\n"
                output_text += anova_df_results.to_string(index=False) + "\n\n"
                self.rm_anova_results_data = anova_df_results
                if self.export_rm_anova_btn: self.export_rm_anova_btn.configure(state="normal")

                output_text += "------------------------------------------------------------\n"
                output_text += "           SIMPLIFIED EXPLANATION OF RESULTS\n"
                output_text += "------------------------------------------------------------\n"
                alpha = 0.05
                output_text += f"(A result is typically considered 'statistically significant' if its p-value ('Pr > F') is less than {alpha:.2f})\n\n"
                for index, row in anova_df_results.iterrows():
                    effect_name_raw = str(row.get('Effect', 'Unknown Effect'))
                    effect_display_name = effect_name_raw.replace(':', ' by ').replace('_', ' ').title()
                    p_value_raw = row.get('Pr > F', row.get('p-unc', np.nan))
                    output_text += f"Effect: {effect_display_name}\n"
                    if pd.isna(p_value_raw): output_text += "  - Significance: N/A (p-value missing).\n\n"; continue
                    is_significant = p_value_raw < alpha
                    p_value_display_str = "< .0001" if p_value_raw < 0.0001 else f"{p_value_raw:.4f}"
                    output_text += f"  - Statistical Finding: {'SIGNIFICANT' if is_significant else 'NOT SIGNIFICANT'} (p-value = {p_value_display_str})\n"

                    explanation = ""
                    if 'condition' in effect_name_raw.lower() and 'roi' in effect_name_raw.lower():  # Interaction
                        if is_significant:
                            explanation = (
                                "  - Interpretation: This is often a very important finding! It means the way brain\n"
                                "                    activity (Summed BCA) changed across your different experimental conditions\n"
                                "                    **significantly depended on which brain region (ROI)** you were observing.\n"
                                "                    The effect of conditions isn't the same for all ROIs.\n"
                                "                    (For example, Condition A might boost activity in the Frontal lobe more than\n"
                                "                     Condition B, but this pattern might be different or even reversed in the\n"
                                "                     Occipital lobe.)\n"
                                "  - Next Steps: Consider creating interaction plots to visualize this. Post-hoc tests\n"
                                "                are often needed to understand where these specific differences lie.\n")
                        else:
                            explanation = (
                                "  - Interpretation: The effect of your experimental conditions on brain activity (Summed BCA)\n"
                                "                    was generally consistent across the different brain regions analyzed.\n"
                                "                    There wasn't a significant 'it depends' relationship found here.\n")
                    elif 'condition' == effect_name_raw.lower():
                        if is_significant:
                            explanation = (
                                "  - Interpretation: Overall, when averaging across all your brain regions (ROIs),\n"
                                "                    your different experimental conditions led to statistically different\n"
                                "                    average levels of brain activity (Summed BCA).\n"
                                "  - Next Steps: If you have more than two conditions, post-hoc tests can help\n"
                                "                identify which specific conditions differ from each other.\n")
                        else:
                            explanation = (
                                "  - Interpretation: When averaging across all brain regions, your different experimental\n"
                                "                    conditions did not produce statistically different overall levels of\n"
                                "                    brain activity (Summed BCA).\n")
                    elif 'roi' == effect_name_raw.lower():
                        if is_significant:
                            explanation = (
                                "  - Interpretation: Different brain regions (ROIs) showed reliably different average levels\n"
                                "                    of brain activity (Summed BCA), regardless of the specific experimental condition.\n"
                                "                    Some regions were consistently more (or less) active than others.\n"
                                "  - Next Steps: If you have more than two ROIs, post-hoc tests can show which specific\n"
                                "                ROIs differ from each other in overall activity.\n")
                        else:
                            explanation = (
                                "  - Interpretation: There wasn't a statistically significant overall difference in brain activity\n"
                                "                    (Summed BCA) between the different brain regions analyzed (when averaged\n"
                                "                    across your experimental conditions).\n")
                    else:
                        explanation = f"  - Interpretation: This effect relates to '{effect_display_name}'.\n"
                    output_text += explanation + "\n"
                output_text += "------------------------------------------------------------\nIMPORTANT NOTE:\n"
                output_text += ("  This explanation simplifies the main statistical patterns. For detailed scientific\n"
                                "  reporting, precise interpretation, and any follow-up analyses (e.g., post-hoc tests\n"
                                "  for significant effects or interactions), please refer to the statistical table above\n"
                                "  and consider consulting with a statistician or researcher familiar with ANOVA.\n")
                output_text += "------------------------------------------------------------\n"
            else:
                output_text += "RM-ANOVA did not return any results or the result was empty.\n"; self.log_to_main_app(
                    "RM-ANOVA returned no results.")
        except Exception as e:
            output_text += f"RM-ANOVA analysis failed: {e}\nCheck logs. Ensure 'statsmodels' is installed.\n"
            self.log_to_main_app(f"!!! RM-ANOVA Error: {e}\n{traceback.format_exc()}")
        self.results_textbox.insert("1.0", output_text)
        self.results_textbox.configure(state="disabled")
        self.log_to_main_app("RM-ANOVA (Summed BCA) run complete.")

    def run_harmonic_check(self):
        # ... (Keep the version of run_harmonic_check I provided last,
        #      which includes the detailed explanation and p-value formatting.
        #      Ensure it also starts with `excluded_pids = self._get_excluded_pids()`
        #      and uses it to filter `active_subjects`.
        #      The N reported should be `num_subjects_in_test`.
        #      Add the Excluded PIDs line to its header.)
        #      This method is long, so I'm not repeating its entire improved text output here,
        #      but the exclusion logic must be integrated at its start.

        self.log_to_main_app("Running Per-Harmonic Significance Check...")
        if not self.results_textbox: self.log_to_main_app("Results textbox not initialized!"); return
        self.results_textbox.configure(state="normal");
        self.results_textbox.delete("1.0", tk.END)
        if self.export_harmonic_check_btn: self.export_harmonic_check_btn.configure(state="disabled")
        self.harmonic_check_results_data.clear()

        excluded_pids = self._get_excluded_pids()
        selected_metric = self.harmonic_metric_var.get()
        actual_sheet_name = self._get_sheet_name_for_metric(selected_metric)

        try:
            mean_value_threshold = float(self.harmonic_threshold_var.get())
        except ValueError:
            messagebox.showerror("Input Error", "Invalid Mean Threshold."); self.results_textbox.configure(
                state="disabled"); return

        active_subjects = [s for s in self.subjects_list if s not in excluded_pids]
        if not active_subjects:
            messagebox.showerror("Data Error", "No subjects remaining after exclusions for Harmonic Check.")
            self.results_textbox.configure(state="disabled");
            return
        if not (self.subject_data and self.conditions_list):
            messagebox.showerror("Data Error", "Subject data or conditions not loaded. Scan folder first.");
            self.results_textbox.configure(state="disabled");
            return

        output_text = f"===== Per-Harmonic Significance Check ({selected_metric}) =====\n"
        output_text += f"A harmonic is flagged as 'Significant' if:\n"
        output_text += f"1. Its average {selected_metric} is reliably different from zero across subjects\n"
        output_text += f"   (1-sample t-test vs 0, p < {HARMONIC_CHECK_ALPHA}).\n"
        output_text += f"2. AND this average {selected_metric} is also > {mean_value_threshold}.\n"
        if excluded_pids:
            output_text += f"Excluded PIDs (Quality Flags): {', '.join(sorted(list(excluded_pids)))}\n"
        output_text += f"(N = subjects included in each specific test listed below)\n\n"

        any_significant_found_overall = False
        loaded_dataframes = {}

        for cond_name in self.conditions_list:
            output_text += f"\n=== Condition: {cond_name} ===\n"
            found_significant_in_this_condition = False
            for roi_name in ROIS.keys():
                sample_file_path = next((self.subject_data[pid].get(cond_name) for pid in active_subjects if
                                         self.subject_data.get(pid, {}).get(cond_name)), None)
                if not sample_file_path: self.log_to_main_app(
                    f"No sample file for {cond_name}/{roi_name} among active subjects to get freq cols."); continue
                try:
                    if sample_file_path not in loaded_dataframes:
                        loaded_dataframes[sample_file_path] = pd.read_excel(sample_file_path,
                                                                            sheet_name=actual_sheet_name,
                                                                            index_col="Electrode")
                    sample_df_cols = loaded_dataframes[sample_file_path].columns
                    included_freq_values = self._get_included_freqs(sample_df_cols)
                except Exception as e:
                    self.log_to_main_app(f"Err reading cols for {cond_name}/{roi_name}: {e}"); continue

                if not included_freq_values:
                    output_text += f"\n  --- ROI: {roi_name} ---\n      No applicable harmonics to check (after exclusions or none found).\n\n";
                    continue

                roi_header_printed = False;
                significant_harmonics_count_for_roi = 0
                for freq_val in included_freq_values:
                    harmonic_col_name = f"{freq_val:.1f}_Hz";
                    subject_harmonic_roi_values = []
                    for pid in active_subjects:
                        file_path = self.subject_data.get(pid, {}).get(cond_name)
                        if not (file_path and os.path.exists(file_path)): continue
                        current_df = loaded_dataframes.get(file_path)
                        if current_df is None:
                            try:
                                current_df = pd.read_excel(file_path, sheet_name=actual_sheet_name,
                                                           index_col="Electrode"); loaded_dataframes[
                                    file_path] = current_df
                            except Exception:
                                continue
                        if current_df.empty or harmonic_col_name not in current_df.columns: continue
                        try:
                            mean_val_subj_roi_harmonic = current_df.reindex(ROIS.get(roi_name))[
                                harmonic_col_name].dropna().mean()
                            if not pd.isna(mean_val_subj_roi_harmonic): subject_harmonic_roi_values.append(
                                mean_val_subj_roi_harmonic)
                        except Exception:
                            continue

                    if len(subject_harmonic_roi_values) >= 3:
                        t_stat, p_value_raw = stats.ttest_1samp(subject_harmonic_roi_values, 0, nan_policy='omit')
                        valid_vals = [v for v in subject_harmonic_roi_values if not pd.isna(v)]
                        n_subj_test = len(valid_vals)
                        if n_subj_test < 3: continue
                        mean_group_val = np.mean(valid_vals);
                        df_val = n_subj_test - 1
                        p_val_str = "< .0001" if p_value_raw < 0.0001 else f"{p_value_raw:.4f}"
                        if p_value_raw < HARMONIC_CHECK_ALPHA and mean_group_val > mean_value_threshold:
                            if not roi_header_printed: output_text += f"\n  --- ROI: {roi_name} ---\n"; roi_header_printed = True
                            found_significant_in_this_condition = True;
                            any_significant_found_overall = True;
                            significant_harmonics_count_for_roi += 1
                            output_text += f"    ----------------------------------------------------\n    Harmonic: {harmonic_col_name} -> SIGNIFICANT RESPONSE\n"
                            output_text += f"        Average {selected_metric}: {mean_group_val:.3f} (N={n_subj_test} subjects)\n"
                            output_text += f"        Statistical Test: t({df_val}) = {t_stat:.2f}, p-value = {p_val_str}\n    ----------------------------------------------------\n"
                            self.harmonic_check_results_data.append(
                                {'Condition': cond_name, 'ROI': roi_name, 'Frequency': harmonic_col_name,
                                 'N_Subjects': n_subj_test, f'Mean_{selected_metric.replace(" ", "_")}': mean_group_val,
                                 'T_Statistic': t_stat, 'P_Value': p_value_raw, 'df': df_val,
                                 'Threshold_Criteria_Mean_Value': mean_value_threshold,
                                 'Threshold_Criteria_Alpha': HARMONIC_CHECK_ALPHA})
                if roi_header_printed:
                    if significant_harmonics_count_for_roi > 1: output_text += f"    Summary for {roi_name}: Found {significant_harmonics_count_for_roi} significant harmonics.\n"
                    output_text += "\n"
                elif included_freq_values:
                    output_text += f"\n  --- ROI: {roi_name} ---\n      No significant harmonics met criteria.\n\n"
            if not found_significant_in_this_condition:
                output_text += f"  No significant harmonics met criteria in this condition.\n\n"
            else:
                output_text += "\n"
        if not any_significant_found_overall:
            output_text += "Overall: No harmonics met criteria across all conditions/ROIs.\n"
        else:
            output_text += "\n--- End of Report ---\nTip: Export for a detailed table.\n"
        self.results_textbox.insert("1.0", output_text)
        if self.harmonic_check_results_data and self.export_harmonic_check_btn: self.export_harmonic_check_btn.configure(
            state="normal")
        self.results_textbox.configure(state="disabled");
        loaded_dataframes.clear()
        self.log_to_main_app("Per-Harmonic Significance Check run complete.")

    def _get_sheet_name_for_metric(self, metric_ui_name: str) -> str:
        """Helper to map UI metric name (from harmonic check dropdown) to actual Excel sheet name."""
        if metric_ui_name == "Z-Score": return "Z Score"  # Handles the space
        return metric_ui_name  # Default for SNR, or others if names match directly

    # --- Export Calls (use keyword arguments) ---
    def export_paired_results(self):
        if not self.paired_tests_results_data: messagebox.showwarning("No Results",
                                                                      "No paired test results data to export."); return
        try:
            stats_export.export_paired_results_to_excel(data=self.paired_tests_results_data,
                                                        parent_folder=self.stats_data_folder_var.get(),
                                                        log_func=self.log_to_main_app)
        except AttributeError:
            messagebox.showerror("Export Error", "'export_paired_results_to_excel' is missing from stats_export.py.")
        except Exception as e:
            messagebox.showerror("Export Error", f"Paired results export failed: {e}\n{traceback.format_exc()}")

    def export_rm_anova_results(self):
        if self.rm_anova_results_data is None or \
                (isinstance(self.rm_anova_results_data, pd.DataFrame) and self.rm_anova_results_data.empty):
            messagebox.showwarning("No Results", "No RM-ANOVA results data to export.");
            return
        try:
            stats_export.export_rm_anova_results_to_excel(anova_table=self.rm_anova_results_data,
                                                          parent_folder=self.stats_data_folder_var.get(),
                                                          log_func=self.log_to_main_app)
        except AttributeError:
            messagebox.showerror("Export Error", "'export_rm_anova_results_to_excel' is missing from stats_export.py.")
        except Exception as e:
            messagebox.showerror("Export Error", f"RM-ANOVA results export failed: {e}\n{traceback.format_exc()}")

    def export_harmonic_check_results(self):
        if not self.harmonic_check_results_data: messagebox.showwarning("No Results",
                                                                        "No harmonic check results data to export."); return
        findings_dict = {}
        metric_from_ui = self.harmonic_metric_var.get()  # e.g., "SNR" or "Z-Score"
        # Ensure key matches what export_significance_results_to_excel expects and what's stored in harmonic_check_results_data
        mean_metric_col_key_in_data = f'Mean_{metric_from_ui.replace(" ", "_")}'

        for item in self.harmonic_check_results_data:
            cond, roi = item['Condition'], item['ROI']
            findings_dict.setdefault(cond, {}).setdefault(roi, []).append({
                'Frequency': item['Frequency'],
                mean_metric_col_key_in_data: item[mean_metric_col_key_in_data],  # Use the dynamic key correctly
                'N': item['N_Subjects'],
                'T_Statistic': item['T_Statistic'],
                'P_Value': item['P_Value'],
                'df': item['df'],
                'Threshold': item['Threshold_Criteria_Mean_Value']  # This is the Mean Value threshold
            })
        if not findings_dict: messagebox.showwarning("No Results",
                                                     "Failed to structure harmonic results for export."); return
        try:
            stats_export.export_significance_results_to_excel(
                findings_dict=findings_dict,
                metric=metric_from_ui,  # Pass the UI name for filename generation
                threshold=float(self.harmonic_threshold_var.get()),  # The user-set mean value threshold
                parent_folder=self.stats_data_folder_var.get(),
                log_func=self.log_to_main_app
            )
        except AttributeError:
            messagebox.showerror("Export Error",
                                 "'export_significance_results_to_excel' is missing from stats_export.py.")
        except Exception as e:
            messagebox.showerror("Export Error", f"Harmonic check export failed: {e}\n{traceback.format_exc()}")

    def call_export_significance_results(self):  # Effectively legacy
        self.log_to_main_app("Note: 'Export Sig. Check Results' button is legacy. Use 'Export Harmonic Results'.")
        pass


# Standalone testing
if __name__ == "__main__":
    try:
        root = ctk.CTk()
        root.title("Stats Tool Test Host")


        # A simple master object with a log method for testing
        class TestMaster:
            def __init__(self): self.name = "TestMasterApp"  # Example attribute

            def log(self, message): print(f"[{pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3]} TestHostLOG] {message}")


        # Mock dependent modules if they are not fully available or for isolated testing
        class MockStatsQCOutliers:
            class QualityOutlierReviewFrame(ctk.CTkFrame):  # Mock the class
                def __init__(self, master, app_log_func, stats_data_folder_var, **kwargs):
                    super().__init__(master, **kwargs)
                    self.app_log_func = app_log_func
                    ctk.CTkLabel(self, text="Mocked QC Outlier Frame").pack()
                    app_log_func("Mocked QualityOutlierReviewFrame initialized.")

                def load_and_display_flags(self): self.app_log_func("Mocked load_and_display_flags called.")

                def get_pids_to_exclude(self): self.app_log_func("Mocked get_pids_to_exclude called."); return set()


        class MockStatsUIBuilder:
            def build_stats_interface(self, app_instance):
                app_instance.log_to_main_app("Mocked build_stats_interface called.")
                # Create a minimal UI or just pass, as real UI builder is complex
                # For testing, ensure essential attributes like app_instance.results_textbox are set if other methods rely on them
                app_instance.results_textbox = ctk.CTkTextbox(app_instance)  # Minimal
                app_instance.results_textbox.pack(fill="both", expand=True)


        import sys


        # Replace actual imports with mocks if they cause issues during standalone run
        # This is tricky with `from .module import Class` type imports within the actual stats.py
        # For simple standalone test, ensure stats_qc_outliers.py and stats_ui_builder.py exist and are importable
        # Or, if they have heavy dependencies not met in test env, mocking them like this becomes necessary.
        # sys.modules['Tools.Stats.stats_qc_outliers'] = MockStatsQCOutliers()
        # sys.modules['Tools.Stats.stats_ui_builder'] = MockStatsUIBuilder()
        # from .stats_qc_outliers import QualityOutlierReviewFrame # This might need adjustment for standalone
        # from .stats_ui_builder import build_stats_interface

        def open_stats_tool():
            StatsAnalysisWindow(master=TestMaster(), default_folder="")


        ctk.CTkButton(root, text="Open Stats Tool (Test)", command=open_stats_tool).pack(pady=20)
        root.mainloop()
    except Exception as e:
        print(f"Error in Stats __main__ test block: {e}\n{traceback.format_exc()}")