# stats_export.py
"""
Contains functions for exporting statistical analysis results from the FPVS Stats Tool
to formatted Excel files using pandas and xlsxwriter.
"""

import os
import pandas as pd
import tkinter as tk # Required for filedialog/messagebox even if GUI isn't built here
from tkinter import filedialog, messagebox
import numpy as np # Required if handling NaN checks explicitly

def export_significance_results_to_excel(findings_dict, metric, threshold, parent_folder, log_func=print):
    """
    Exports the structured Significance Check findings to a formatted Excel file.

    Args:
        findings_dict (dict): Nested dictionary {condition: {roi: [finding_list]}}
                              containing significant results. Each finding in the list
                              is expected to be a dict like:
                              {'Frequency': str, 'Mean_MetricName': float, 'N': int, ...}
        metric (str): The metric analyzed (e.g., "Z Score"). Used for filename and lookup.
        threshold (float): The significance threshold used. Used for filename.
        parent_folder (str): The path to the parent data folder, used for suggesting save location.
        log_func (callable, optional): Function to use for logging messages. Defaults to print.
    """
    log_func("Attempting to export Significance Check results...")

    if not findings_dict or not any(any(findings_dict[cond].values()) for cond in findings_dict):
        log_func("No significant findings available to export.")
        messagebox.showwarning("No Results", "No significant findings available to export for the Significance Check.")
        return

    # --- 1. Flatten the data structure for DataFrame ---
    flat_results = []
    metric_col_name = f'Mean_{metric.replace(" ","_")}' # Construct the key for the metric value

    for condition, roi_data in findings_dict.items():
        for roi_name, findings_list in roi_data.items():
            for finding in findings_list:
                row = {
                    'Condition': condition,
                    'ROI': roi_name,
                    'Frequency': finding.get('Frequency', 'N/A'),
                    metric_col_name: finding.get(metric_col_name, np.nan),
                    'N': finding.get('N', 'N/A'),
                    'Threshold_Used': finding.get('Threshold', threshold) # Get threshold from finding or default
                    # Add other relevant columns if stored (e.g., t-stat, p-value from 1-samp test)
                    # 'T_Stat_vs_Threshold': finding.get('T_Stat', np.nan),
                    # 'P_Value_vs_Threshold': finding.get('P_Value', np.nan)
                }
                flat_results.append(row)

    if not flat_results:
        log_func("Data structure was provided, but yielded no results to flatten for export.")
        messagebox.showwarning("No Results", "Could not prepare any results for export.")
        return

    df_export = pd.DataFrame(flat_results)

    # --- 2. Define preferred column order ---
    # Adjust if adding t-test results etc.
    preferred_cols = ['Condition', 'ROI', 'Frequency', metric_col_name, 'N', 'Threshold_Used']
    # Filter to only columns that actually exist in the generated DataFrame
    cols_to_export = [col for col in preferred_cols if col in df_export.columns]
    if not cols_to_export:
         log_func("Error: Could not determine columns to export.")
         messagebox.showerror("Export Error", "Could not determine valid columns for export.")
         return
    df_export = df_export[cols_to_export] # Reorder DF

    # --- 3. Get Save Path from User ---
    initial_dir = parent_folder if os.path.isdir(parent_folder) else os.path.expanduser("~")
    metric_filename = metric.replace("-", "").replace(" ", "")
    threshold_filename = f"{threshold:.2f}".replace('.', 'p') # e.g., 1p96
    suggested_filename = f"Stats_SignificanceCheck_{metric_filename}_Thresh{threshold_filename}.xlsx"

    save_path = filedialog.asksaveasfilename(
        title=f"Save Significance Check Results ({metric} > {threshold:.2f}) As",
        initialdir=initial_dir,
        initialfile=suggested_filename,
        defaultextension=".xlsx",
        filetypes=[("Excel Workbook", "*.xlsx"), ("All Files", "*.*")]
    )

    if not save_path:
        log_func("Export cancelled by user.")
        return

    # --- 4. Write to Excel with Formatting ---
    try:
        log_func(f"Exporting Significance Check results to: {save_path}")
        with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
            df_export.to_excel(writer, sheet_name='Significant Findings', index=False)
            workbook = writer.book
            worksheet = writer.sheets['Significant Findings']

            # Define cell formats (including centering)
            fmt_num_2dp_center = workbook.add_format({'num_format': '0.00', 'align': 'center', 'valign': 'vcenter'})
            fmt_int_center = workbook.add_format({'num_format': '0', 'align': 'center', 'valign': 'vcenter'})
            fmt_text_left = workbook.add_format({'align': 'left', 'valign': 'vcenter'}) # Default for text
            fmt_text_center = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

            # Column specific formats and widths
            col_settings = {
                'Condition': {'format': fmt_text_left, 'width': 25},
                'ROI': {'format': fmt_text_left, 'width': 18},
                'Frequency': {'format': fmt_text_center, 'width': 12},
                metric_col_name: {'format': fmt_num_2dp_center, 'width': 15},
                'N': {'format': fmt_int_center, 'width': 8},
                'Threshold_Used': {'format': fmt_num_2dp_center, 'width': 15}
                # Add settings for T_Stat, P_Value if they are included
                # 'T_Stat_vs_Threshold': {'format': fmt_num_2dp_center, 'width': 15},
                # 'P_Value_vs_Threshold': {'format': fmt_num_3dp_center, 'width': 15}, # Need 3dp format
            }

            # Apply formatting and auto-adjust column widths (using heuristic + defined minimums)
            for col_idx, col_name in enumerate(df_export.columns):
                settings = col_settings.get(col_name, {'format': fmt_text_left, 'width': 15}) # Default settings
                col_format = settings['format']
                min_width = settings['width']

                # Calculate width based on data and header length
                try:
                    max_len_data = df_export[col_name].astype(str).map(len).max()
                    if pd.isna(max_len_data): max_len_data = 0
                    # Consider header length, add padding
                    col_width = max(int(max_len_data) + 2, len(col_name) + 2, min_width)
                except Exception:
                    col_width = max(len(col_name) + 5, min_width) # Fallback width

                worksheet.set_column(col_idx, col_idx, col_width, col_format)

        log_func("Significance Check results export successful.")
        messagebox.showinfo("Export Successful", f"Significance Check results exported to:\n{save_path}")

    except PermissionError as e:
         log_func(f"!!! Export Failed: Permission denied writing to {save_path}. Check file/folder permissions.")
         messagebox.showerror("Export Failed", f"Permission denied writing file:\n{save_path}\nClose the file if it's open, or choose a different location.")
    except Exception as e:
        log_func(f"!!! Export Failed: An unexpected error occurred.\n{traceback.format_exc()}")
        messagebox.showerror("Export Failed", f"Could not save Excel file:\n{e}")

# You could add other export functions here later if needed, e.g., for paired results
# def export_paired_results_to_excel(results_structured, ...):
#    pass