"""UI helper functions for the Stats tool."""

import customtkinter as ctk

from config import FONT_BOLD
from .stats_constants import ALL_ROIS_OPTION, ROIS


def build_ui(win):
    """Construct all widgets for a :class:`StatsAnalysisWindow`."""
    validate_num_cmd = (win.register(win._validate_numeric), '%P')

    main_frame = ctk.CTkFrame(win)
    main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    win.grid_rowconfigure(0, weight=1)
    win.grid_columnconfigure(0, weight=1)
    main_frame.grid_columnconfigure(0, weight=1)

    # --- Row 0: Folder Selection ---
    folder_frame = ctk.CTkFrame(main_frame)
    folder_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 10))
    folder_frame.grid_columnconfigure(1, weight=1)
    ctk.CTkLabel(folder_frame, text="Data Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    ctk.CTkEntry(folder_frame, textvariable=win.stats_data_folder_var, state="readonly").grid(
        row=0, column=1, padx=5, pady=5, sticky="ew")
    ctk.CTkButton(folder_frame, text="Browse...", command=win.browse_folder).grid(row=0, column=2, padx=5, pady=5)
    ctk.CTkLabel(folder_frame, textvariable=win.detected_info_var, justify="left", anchor="w").grid(
        row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")

    # --- Row 1: Common Setup ---
    common_setup_frame = ctk.CTkFrame(main_frame)
    common_setup_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
    common_setup_frame.grid_columnconfigure(1, weight=1)
    common_setup_frame.grid_columnconfigure(3, weight=1)
    ctk.CTkLabel(common_setup_frame, text="Base Freq (Hz):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    ctk.CTkEntry(common_setup_frame, textvariable=win.base_freq_var, validate='key',
                 validatecommand=validate_num_cmd, width=100).grid(row=0, column=1, padx=5, pady=5, sticky="w")
    ctk.CTkLabel(common_setup_frame, text="ROI (Paired/ANOVA):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    roi_options = [ALL_ROIS_OPTION] + list(ROIS.keys())
    win.roi_menu = ctk.CTkOptionMenu(common_setup_frame, variable=win.roi_var,
                                    values=roi_options)
    win.roi_menu.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    ctk.CTkLabel(common_setup_frame, text="Condition A (Paired):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    win.condA_menu = ctk.CTkOptionMenu(common_setup_frame, variable=win.condition_A_var, values=["(Scan Folder)"],
                                       command=lambda *_: win.update_condition_B_options())
    win.condA_menu.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
    ctk.CTkLabel(common_setup_frame, text="Condition B (Paired):").grid(row=2, column=2, padx=(15, 5), pady=5,
                                                                        sticky="w")
    win.condB_menu = ctk.CTkOptionMenu(common_setup_frame, variable=win.condition_B_var,
                                      values=["(Scan Folder)"])
    win.condB_menu.grid(row=2, column=3, padx=5, pady=5, sticky="ew")

    ctk.CTkLabel(common_setup_frame, text="Alpha (Sig. Level):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
    ctk.CTkEntry(common_setup_frame, textvariable=win.alpha_var, validate='key',
                 validatecommand=validate_num_cmd, width=100).grid(row=3, column=1, padx=5, pady=5, sticky="w")

    # --- Row 2: Section A - Summed BCA Analysis ---
    summed_bca_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
    summed_bca_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=(10, 5))
    ctk.CTkLabel(summed_bca_frame, text="Summed BCA Analysis:", font=FONT_BOLD).pack(anchor="w", pady=(0, 5))
    buttons_summed_frame = ctk.CTkFrame(summed_bca_frame)
    buttons_summed_frame.pack(fill="x", padx=0, pady=0)
    ctk.CTkButton(buttons_summed_frame, text="Run Paired Tests (Summed BCA)", command=win.run_paired_tests).pack(
        side="left", padx=(0, 5), pady=5)
    ctk.CTkButton(buttons_summed_frame, text="Run RM-ANOVA (Summed BCA)", command=win.run_rm_anova).pack(
        side="left", padx=5, pady=5)
    ctk.CTkButton(buttons_summed_frame, text="Run Mixed Model (Summed BCA)", command=win.run_mixed_model).pack(
        side="left", padx=5, pady=5)
    win.export_paired_tests_btn = ctk.CTkButton(buttons_summed_frame, text="Export Paired Results",
                                               state="disabled", command=win.export_paired_results)
    win.export_paired_tests_btn.pack(side="left", padx=5, pady=5)
    win.export_rm_anova_btn = ctk.CTkButton(buttons_summed_frame, text="Export RM-ANOVA", state="disabled",
                                            command=win.export_rm_anova_results)
    win.export_rm_anova_btn.pack(side="left", padx=5, pady=5)
    win.export_mixed_model_btn = ctk.CTkButton(buttons_summed_frame, text="Export Mixed Model", state="disabled",
                                              command=win.export_mixed_model_results)
    win.export_mixed_model_btn.pack(side="left", padx=5, pady=5)

    # --- Row 3: Section B - Harmonic Significance Check ---
    harmonic_check_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
    harmonic_check_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
    ctk.CTkLabel(harmonic_check_frame, text="Per-Harmonic Significance Check:",
                 font=FONT_BOLD).pack(anchor="w", pady=(0, 5))
    controls_harmonic_frame = ctk.CTkFrame(harmonic_check_frame)
    controls_harmonic_frame.pack(fill="x", padx=0, pady=0)
    ctk.CTkLabel(controls_harmonic_frame, text="Metric:").grid(row=0, column=0, padx=(0, 5), pady=5, sticky="w")
    win.harmonic_metric_menu = ctk.CTkOptionMenu(controls_harmonic_frame, variable=win.harmonic_metric_var,
                                                values=["SNR", "Z Score"])
    win.harmonic_metric_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ctk.CTkLabel(controls_harmonic_frame, text="Mean Threshold:").grid(row=0, column=2, padx=(15, 5), pady=5,
                                                                       sticky="w")
    ctk.CTkEntry(controls_harmonic_frame, textvariable=win.harmonic_threshold_var, validate='key',
                 validatecommand=validate_num_cmd, width=100).grid(row=0, column=3, padx=5, pady=5, sticky="w")
    ctk.CTkButton(controls_harmonic_frame, text="Run Harmonic Check", command=win.run_harmonic_check).grid(
        row=0, column=4, padx=15, pady=5)
    win.export_harmonic_check_btn = ctk.CTkButton(controls_harmonic_frame, text="Export Harmonic Results",
                                                 state="disabled", command=win.export_harmonic_check_results)
    win.export_harmonic_check_btn.grid(row=0, column=5, padx=5, pady=5)
    controls_harmonic_frame.grid_columnconfigure(1, weight=1)

    # --- Row 4: Results Textbox ---
    win.results_textbox = ctk.CTkTextbox(main_frame, wrap="word", state="disabled",
                                        font=ctk.CTkFont(family="Courier New", size=12))
    win.results_textbox.grid(row=4, column=0, sticky="nsew", padx=5, pady=(10, 5))
    main_frame.grid_rowconfigure(4, weight=1)
