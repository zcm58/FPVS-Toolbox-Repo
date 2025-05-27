# Tools/Stats/stats_ui_builder.py
# -*- coding: utf-8 -*-
"""
Builds the user interface for the StatsAnalysisWindow in the FPVS Toolbox.
Reflects removal of manual frequency checkboxes and correct usage of imported constants.
"""

import customtkinter as ctk
import tkinter as tk

# Import constants from the main project config file
# This assumes config.py is in a location Python can find (e.g., src directory, and src is in PYTHONPATH)
# If Tools/Stats is a sub-package of a larger package that includes config.py at its root,
# a relative import like 'from ..config import ...' might be needed if 'src' isn't directly on path.
# For now, assuming 'from config import ...' works based on your project setup.
try:
    from config import (ROIS, ALL_ROIS_OPTION,
                        STATS_PLACEHOLDER_SCAN_FOLDER,
                        STATS_PLACEHOLDER_SELECT_CONDITION_A,  # Needed for update_condition_B_options logic if called early
                        STATS_PLACEHOLDER_NO_OTHER_CONDITIONS  # Also for update_condition_B_options
                        )
except ImportError:
    print("Warning: Could not import UI constants from main config.py for stats_ui_builder. Using defaults.")
    ROIS = {"Frontal Lobe": ["F3", "F4", "Fz"], "Occipital Lobe": ["O1", "O2", "Oz"],
            "Parietal Lobe": ["P3", "P4", "Pz"], "Central Lobe": ["C3", "C4", "Cz"]}
    ALL_ROIS_OPTION = "(All ROIs)"
    STATS_PLACEHOLDER_SCAN_FOLDER = "(Scan Folder)"
    STATS_PLACEHOLDER_SELECT_CONDITION_A = "(Select Condition A)"
    STATS_PLACEHOLDER_NO_OTHER_CONDITIONS = "(No other conditions)"

# Import the QualityOutlierReviewFrame
from .stats_qc_outliers import QualityOutlierReviewFrame


def build_stats_interface(app_instance):
    """
    Creates and lays out all widgets for the StatsAnalysisWindow.

    Args:
        app_instance: The instance of the StatsAnalysisWindow class.
                      This instance is expected to have all necessary tk.StringVar,
                      tk.BooleanVar, callback methods, and attributes where
                      created widgets should be stored.
    """
    validate_num_cmd = (app_instance.register(app_instance._validate_numeric), '%P')

    app_instance.grid_rowconfigure(0, weight=1)
    app_instance.grid_columnconfigure(0, weight=1)

    main_panel_frame = ctk.CTkFrame(app_instance)
    main_panel_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    main_panel_frame.grid_columnconfigure(0, weight=1)

    main_panel_frame.grid_rowconfigure(0, weight=0)
    main_panel_frame.grid_rowconfigure(1, weight=0)
    main_panel_frame.grid_rowconfigure(2, weight=0)
    main_panel_frame.grid_rowconfigure(3, weight=0)
    main_panel_frame.grid_rowconfigure(4, weight=0)
    main_panel_frame.grid_rowconfigure(5, weight=1)

    # --- Row 0: Data Input Frame ---
    input_frame = ctk.CTkFrame(main_panel_frame)
    input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 10))
    input_frame.grid_columnconfigure(1, weight=1)

    ctk.CTkLabel(input_frame, text="Data Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    folder_entry = ctk.CTkEntry(input_frame, textvariable=app_instance.stats_data_folder_var, state="readonly")
    folder_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    browse_button = ctk.CTkButton(input_frame, text="Select Folder", command=app_instance.browse_folder)
    browse_button.grid(row=0, column=2, padx=5, pady=5)
    detected_label = ctk.CTkLabel(input_frame, textvariable=app_instance.detected_info_var, justify="left", anchor="w")
    detected_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")

    # --- Row 1: Common Analysis Setup Frame ---
    common_setup_frame = ctk.CTkFrame(main_panel_frame)
    common_setup_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
    common_setup_frame.grid_columnconfigure(1, weight=1)
    common_setup_frame.grid_columnconfigure(3, weight=1)

    ctk.CTkLabel(common_setup_frame, text="Base Freq (Hz) for Exclusions:").grid(row=0, column=0, padx=5, pady=5,
                                                                                 sticky="w")
    base_freq_entry = ctk.CTkEntry(common_setup_frame, textvariable=app_instance.base_freq_var,
                                   validate='key', validatecommand=validate_num_cmd, width=100)
    base_freq_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    ctk.CTkLabel(common_setup_frame, text="ROI (Paired/ANOVA):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    roi_options_list = [ALL_ROIS_OPTION] + list(ROIS.keys())
    app_instance.roi_menu = ctk.CTkOptionMenu(common_setup_frame, variable=app_instance.roi_var,
                                              values=roi_options_list)
    app_instance.roi_menu.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

    ctk.CTkLabel(common_setup_frame, text="Condition A (Paired):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    # Use the imported constant directly for the initial values list
    app_instance.cond_A_menu = ctk.CTkOptionMenu(common_setup_frame, variable=app_instance.condition_A_var,
                                                 values=[STATS_PLACEHOLDER_SCAN_FOLDER],
                                                 command=lambda *args: app_instance.update_condition_B_options())
    app_instance.cond_A_menu.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

    ctk.CTkLabel(common_setup_frame, text="Condition B (Paired):").grid(row=2, column=2, padx=(15, 5), pady=5,
                                                                        sticky="w")
    # Use the imported constant directly for the initial values list
    app_instance.cond_B_menu = ctk.CTkOptionMenu(common_setup_frame, variable=app_instance.condition_B_var,
                                                 values=[STATS_PLACEHOLDER_SCAN_FOLDER])
    app_instance.cond_B_menu.grid(row=2, column=3, padx=5, pady=5, sticky="ew")

    # --- Row 2: Quality Outlier Review Frame ---
    app_instance.qc_outlier_frame = QualityOutlierReviewFrame(
        master=main_panel_frame,
        app_log_func=app_instance.log_to_main_app,
        stats_data_folder_var=app_instance.stats_data_folder_var
    )
    app_instance.qc_outlier_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=(10, 5))

    # --- Row 3: Section A - Summed BCA Analysis ---
    summed_bca_frame = ctk.CTkFrame(main_panel_frame, fg_color="transparent")
    summed_bca_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

    ctk.CTkLabel(summed_bca_frame, text="Summed BCA Analysis:", font=ctk.CTkFont(weight="bold")).pack(anchor="w",
                                                                                                      pady=(0, 5))
    buttons_summed_frame = ctk.CTkFrame(summed_bca_frame)
    buttons_summed_frame.pack(fill="x", padx=0, pady=0)
    ctk.CTkButton(buttons_summed_frame, text="Run Paired Tests (Summed BCA)",
                  command=app_instance.run_paired_tests).pack(side="left", padx=(0, 5), pady=5)
    ctk.CTkButton(buttons_summed_frame, text="Run RM-ANOVA (Summed BCA)", command=app_instance.run_rm_anova).pack(
        side="left", padx=5, pady=5)
    app_instance.export_paired_tests_btn = ctk.CTkButton(buttons_summed_frame, text="Export Paired Results",
                                                         state="disabled", command=app_instance.export_paired_results)
    app_instance.export_paired_tests_btn.pack(side="left", padx=5, pady=5)
    app_instance.export_rm_anova_btn = ctk.CTkButton(buttons_summed_frame, text="Export RM-ANOVA", state="disabled",
                                                     command=app_instance.export_rm_anova_results)
    app_instance.export_rm_anova_btn.pack(side="left", padx=5, pady=5)

    # --- Row 4: Section B - Harmonic Significance Check ---
    harmonic_check_frame = ctk.CTkFrame(main_panel_frame, fg_color="transparent")
    harmonic_check_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

    ctk.CTkLabel(harmonic_check_frame, text="Per-Harmonic Significance Check:", font=ctk.CTkFont(weight="bold")).pack(
        anchor="w", pady=(0, 5))
    controls_harmonic_frame = ctk.CTkFrame(harmonic_check_frame)
    controls_harmonic_frame.pack(fill="x", padx=0, pady=0)
    ctk.CTkLabel(controls_harmonic_frame, text="Metric:").grid(row=0, column=0, padx=(0, 5), pady=5, sticky="w")
    app_instance.harmonic_metric_menu = ctk.CTkOptionMenu(controls_harmonic_frame,
                                                          variable=app_instance.harmonic_metric_var,
                                                          values=["SNR", "Z Score"])
    app_instance.harmonic_metric_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ctk.CTkLabel(controls_harmonic_frame, text="Mean Threshold:").grid(row=0, column=2, padx=(15, 5), pady=5,
                                                                       sticky="w")
    threshold_entry_harmonic = ctk.CTkEntry(controls_harmonic_frame,
                                            textvariable=app_instance.harmonic_threshold_var,
                                            validate='key', validatecommand=validate_num_cmd, width=100)
    threshold_entry_harmonic.grid(row=0, column=3, padx=5, pady=5, sticky="w")
    ctk.CTkButton(controls_harmonic_frame, text="Run Significance Check", command=app_instance.run_harmonic_check).grid(
        row=0, column=4, padx=15, pady=5)
    app_instance.export_harmonic_check_btn = ctk.CTkButton(controls_harmonic_frame, text="Export Significant Results",
                                                           state="disabled",
                                                           command=app_instance.export_harmonic_check_results)
    app_instance.export_harmonic_check_btn.grid(row=0, column=5, padx=5, pady=5)
    controls_harmonic_frame.grid_columnconfigure(1, weight=1)

    # --- Row 5: Results Textbox ---
    app_instance.results_textbox = ctk.CTkTextbox(main_panel_frame, wrap="word", state="disabled",
                                                  font=ctk.CTkFont(family="Courier New", size=12))
    app_instance.results_textbox.grid(row=5, column=0, sticky="nsew", padx=5, pady=(10, 5))

    # Initial update for condition B menu based on default or scan folder results
    # This is important to call after all relevant menus are created.
    app_instance.update_condition_B_options()