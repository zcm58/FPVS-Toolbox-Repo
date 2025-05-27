# Tools/Stats/stats_ui_builder.py
# -*- coding: utf-8 -*-
"""
Builds the user interface for the StatsAnalysisWindow in the FPVS Toolbox.
"""

import customtkinter as ctk
import tkinter as tk

# It's often cleaner if these are passed from app_instance or a shared config,
# but direct import can work if stats.py defines them at module level and
# this doesn't create circular import issues (usually fine if builder only uses them).
# For now, let's assume app_instance will provide access or they are globally available.
# If they are module-level constants in stats.py, you'd do:
# from .stats import ROIS, ALL_ROIS_OPTION # Assuming they are defined in stats.py
# For this example, I will assume they are accessible via app_instance or defined locally if not.

# We'll also need the QualityOutlierReviewFrame
from .stats_qc_outliers import QualityOutlierReviewFrame


def build_stats_interface(app_instance):
    """
    Creates and lays out all widgets for the StatsAnalysisWindow.

    Args:
        app_instance: The instance of the StatsAnalysisWindow class.
                      This instance is expected to have all necessary tk.StringVar,
                      tk.BooleanVar, callback methods, and attributes where
                      created widgets should be stored (e.g., app_instance.results_textbox).
    """
    # Access constants like ROIS, ALL_ROIS_OPTION if they are attributes of app_instance
    # or import them if they are module-level in stats.py or a config file.
    # For this example, assuming they are attributes for clarity if passed, or defined locally.
    # If they are module-level constants in stats.py, this import would be typical:
    from .stats import ROIS, ALL_ROIS_OPTION

    validate_num_cmd = (app_instance.register(app_instance._validate_numeric), '%P')

    # --- Main Frame Setup ---
    # The main_frame is now created in app_instance.create_stats_widgets before calling this.
    # We will assume app_instance is the main Toplevel window, and we add a primary frame to it.
    # Or, if create_stats_widgets already made a main_frame in app_instance, we use that.
    # Let's assume app_instance is the Toplevel, and we create the main_frame here.

    # Configure main_frame directly on app_instance (the Toplevel window)
    app_instance.grid_rowconfigure(0, weight=1)
    app_instance.grid_columnconfigure(0, weight=1)

    main_panel_frame = ctk.CTkFrame(app_instance)  # This is the main content holder
    main_panel_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    main_panel_frame.grid_columnconfigure(0, weight=1)  # Allow content column to expand

    # Define row weights for main_panel_frame sections
    # Adjust these row numbers and weights as sections are added/ordered
    main_panel_frame.grid_rowconfigure(0, weight=0)  # Input Frame
    main_panel_frame.grid_rowconfigure(1, weight=0)  # Common Setup Frame
    main_panel_frame.grid_rowconfigure(2, weight=0)  # Quality Outlier Review Frame (new)
    main_panel_frame.grid_rowconfigure(3, weight=0)  # Summed BCA Analysis Buttons
    main_panel_frame.grid_rowconfigure(4, weight=0)  # Harmonic Check Controls
    main_panel_frame.grid_rowconfigure(5, weight=1)  # Results Textbox (expands)

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
    common_setup_frame.grid_columnconfigure(1, weight=1)  # Col for ROI menu
    common_setup_frame.grid_columnconfigure(3, weight=1)  # Col for Cond B menu

    # Base Frequency
    ctk.CTkLabel(common_setup_frame, text="Base Freq (Hz):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    base_freq_entry = ctk.CTkEntry(common_setup_frame, textvariable=app_instance.base_freq_var,
                                   validate='key', validatecommand=validate_num_cmd, width=100)
    base_freq_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    # ROI (for Paired/ANOVA)
    ctk.CTkLabel(common_setup_frame, text="ROI (Paired/ANOVA):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    roi_options_list = [ALL_ROIS_OPTION] + list(ROIS.keys())
    app_instance.roi_menu = ctk.CTkOptionMenu(common_setup_frame, variable=app_instance.roi_var,
                                              values=roi_options_list)
    app_instance.roi_menu.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

    # Condition A (Paired)
    ctk.CTkLabel(common_setup_frame, text="Condition A (Paired):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    app_instance.cond_A_menu = ctk.CTkOptionMenu(common_setup_frame, variable=app_instance.condition_A_var,
                                                 values=["(Scan Folder)"],
                                                 command=lambda *args: app_instance.update_condition_B_options())
    app_instance.cond_A_menu.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

    # Condition B (Paired)
    ctk.CTkLabel(common_setup_frame, text="Condition B (Paired):").grid(row=2, column=2, padx=(15, 5), pady=5,
                                                                        sticky="w")
    app_instance.cond_B_menu = ctk.CTkOptionMenu(common_setup_frame, variable=app_instance.condition_B_var,
                                                 values=["(Scan Folder)"])
    app_instance.cond_B_menu.grid(row=2, column=3, padx=5, pady=5, sticky="ew")

    # Frequency Checkboxes (Example - if you still need them for some analysis)
    # If these are fully removed, this section can be deleted.
    # For now, assuming they are still relevant for the Harmonic Check, even if base_freq is primary.
    # If the harmonic check purely relies on _get_included_freqs, this UI might be less necessary.
    # For this build, let's assume it's still part of the harmonic check setup.
    ctk.CTkLabel(common_setup_frame, text="Frequencies (Harmonic Check):").grid(row=3, column=0, padx=5, pady=(10, 2),
                                                                                sticky="nw")
    freq_frame_outer = ctk.CTkFrame(common_setup_frame, fg_color="transparent")
    freq_frame_outer.grid(row=3, column=1, columnspan=3, sticky="nsew", padx=5, pady=5)

    app_instance.freq_scrollable_frame = ctk.CTkScrollableFrame(freq_frame_outer, label_text="", height=100)
    app_instance.freq_scrollable_frame.pack(fill="both", expand=True)

    # Populate Frequency Checkboxes (app_instance needs TARGET_FREQUENCIES and freq_checkbox_vars)
    # This logic would remain in stats.py or be passed to a helper if TARGET_FREQUENCIES is dynamic
    # For now, this implies app_instance.TARGET_FREQUENCIES exists or is accessible.
    # The actual creation of checkboxes can be a method on app_instance called here.
    app_instance.populate_frequency_checkboxes()  # Assumes this method exists on app_instance

    freq_button_frame = ctk.CTkFrame(freq_frame_outer, fg_color="transparent")
    freq_button_frame.pack(fill="x", pady=(2, 0))
    ctk.CTkButton(freq_button_frame, text="Select All Harmonics", command=app_instance.select_all_freqs,
                  width=150).pack(side="left", padx=(0, 5))
    ctk.CTkButton(freq_button_frame, text="Deselect All Harmonics", command=app_instance.deselect_all_freqs,
                  width=150).pack(side="left")

    # --- Row 2: Quality Outlier Review Frame ---
    # This frame is from stats_qc_outliers.py
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
                                                          values=["SNR",
                                                                  "Z Score"])  # Assuming Z-Score sheet name is "Z Score"
    app_instance.harmonic_metric_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ctk.CTkLabel(controls_harmonic_frame, text="Mean Threshold:").grid(row=0, column=2, padx=(15, 5), pady=5,
                                                                       sticky="w")
    threshold_entry_harmonic = ctk.CTkEntry(controls_harmonic_frame, textvariable=app_instance.harmonic_threshold_var,
                                            validate='key', validatecommand=validate_num_cmd, width=100)
    threshold_entry_harmonic.grid(row=0, column=3, padx=5, pady=5, sticky="w")
    ctk.CTkButton(controls_harmonic_frame, text="Run Harmonic Check", command=app_instance.run_harmonic_check).grid(
        row=0, column=4, padx=15, pady=5)
    app_instance.export_harmonic_check_btn = ctk.CTkButton(controls_harmonic_frame, text="Export Harmonic Results",
                                                           state="disabled",
                                                           command=app_instance.export_harmonic_check_results)
    app_instance.export_harmonic_check_btn.grid(row=0, column=5, padx=5, pady=5)
    controls_harmonic_frame.grid_columnconfigure(1, weight=1)

    # --- Row 5: Results Textbox ---
    app_instance.results_textbox = ctk.CTkTextbox(main_panel_frame, wrap="word", state="disabled",
                                                  font=ctk.CTkFont(family="Courier New", size=12))
    app_instance.results_textbox.grid(row=5, column=0, sticky="nsew", padx=5, pady=(10, 5))
    # main_panel_frame.grid_rowconfigure(5, weight=1) # Already set earlier

    # Initial update for condition B menu based on default scan folder (if any) or default empty lists
    app_instance.update_condition_B_options()