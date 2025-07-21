# UI creation method extracted from stats.py

from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)
from PySide6.QtGui import QFont, QDoubleValidator
from PySide6.QtCore import Qt

from config import FONT_FAMILY
from . import stats_export


def create_widgets(self):
    bold_font = QFont(FONT_FAMILY, 12, QFont.Bold)

    main_layout = QGridLayout(self)
    self.setLayout(main_layout)
    main_layout.setContentsMargins(10, 10, 10, 10)
    main_layout.setRowStretch(3, 1)
    main_layout.setColumnStretch(0, 1)

    # --- Row 0: Folder Selection ---
    folder_frame = QFrame(self)
    folder_layout = QGridLayout(folder_frame)
    folder_layout.setContentsMargins(5, 5, 5, 10)
    folder_layout.setColumnStretch(1, 1)

    folder_label = QLabel("Data Folder:")
    folder_layout.addWidget(folder_label, 0, 0, alignment=Qt.AlignLeft)

    self.folder_entry = QLineEdit()
    self.folder_entry.setReadOnly(True)
    if hasattr(self.stats_data_folder_var, "get"):
        self.folder_entry.setText(self.stats_data_folder_var.get())
        self.folder_entry.textChanged.connect(lambda t: self.stats_data_folder_var.set(t))
    folder_layout.addWidget(self.folder_entry, 0, 1)

    browse_btn = QPushButton("Browse...")
    browse_btn.clicked.connect(self.browse_folder)
    folder_layout.addWidget(browse_btn, 0, 2)

    self.detected_info_label = QLabel(
        self.detected_info_var.get() if hasattr(self.detected_info_var, "get") else ""
    )
    self.detected_info_label.setAlignment(Qt.AlignLeft)
    self.detected_info_label.setWordWrap(True)
    folder_layout.addWidget(self.detected_info_label, 1, 0, 1, 3)

    main_layout.addWidget(folder_frame, 0, 0)

    # --- Row 1: Section A - Summed BCA Analysis ---
    summed_bca_frame = QFrame(self)
    summed_bca_layout = QVBoxLayout(summed_bca_frame)
    label = QLabel("Summed BCA Analysis:")
    label.setFont(bold_font)
    summed_bca_layout.addWidget(label, alignment=Qt.AlignLeft)

    buttons_frame = QFrame()
    buttons_layout = QGridLayout(buttons_frame)
    buttons_layout.setColumnStretch(0, 1)
    buttons_layout.setColumnStretch(1, 1)

    btn_rm_anova = QPushButton("Run RM-ANOVA (Summed BCA)")
    btn_rm_anova.clicked.connect(self.run_rm_anova)
    buttons_layout.addWidget(btn_rm_anova, 0, 0)

    btn_mixed = QPushButton("Run Mixed Model")
    btn_mixed.clicked.connect(self.run_mixed_model)
    buttons_layout.addWidget(btn_mixed, 1, 0)

    self.run_posthoc_btn = QPushButton("Run Interaction Post-hocs")
    self.run_posthoc_btn.clicked.connect(self.run_interaction_posthocs)
    buttons_layout.addWidget(self.run_posthoc_btn, 2, 0)

    self.export_rm_anova_btn = QPushButton("Export RM-ANOVA")
    self.export_rm_anova_btn.setEnabled(False)
    self.export_rm_anova_btn.clicked.connect(
        lambda: stats_export.export_rm_anova_results_to_excel(
            anova_table=self.rm_anova_results_data,
            parent_folder=self.stats_data_folder_var.get(),
            log_func=self.log_to_main_app,
        )
    )
    buttons_layout.addWidget(self.export_rm_anova_btn, 0, 1)

    self.export_mixed_model_btn = QPushButton("Export Mixed Model")
    self.export_mixed_model_btn.setEnabled(False)
    self.export_mixed_model_btn.clicked.connect(
        lambda: stats_export.export_mixed_model_results_to_excel(
            results_df=self.mixed_model_results_data,
            parent_folder=self.stats_data_folder_var.get(),
            log_func=self.log_to_main_app,
        )
    )
    buttons_layout.addWidget(self.export_mixed_model_btn, 1, 1)

    self.export_posthoc_btn = QPushButton("Export Post-hoc Results")
    self.export_posthoc_btn.setEnabled(False)
    self.export_posthoc_btn.clicked.connect(
        lambda: stats_export.export_posthoc_results_to_excel(
            results_df=self.posthoc_results_data,
            factor=self.posthoc_factor_var.get(),
            parent_folder=self.stats_data_folder_var.get(),
            log_func=self.log_to_main_app,
        )
    )
    buttons_layout.addWidget(self.export_posthoc_btn, 2, 1)

    summed_bca_layout.addWidget(buttons_frame)
    main_layout.addWidget(summed_bca_frame, 1, 0)

    # --- Row 2: Section B - Harmonic Significance Check ---
    harmonic_check_frame = QFrame(self)
    harmonic_layout = QVBoxLayout(harmonic_check_frame)
    label = QLabel("Per-Harmonic Significance Check:")
    label.setFont(bold_font)
    harmonic_layout.addWidget(label, alignment=Qt.AlignLeft)

    controls_frame = QFrame()
    controls_layout = QGridLayout(controls_frame)
    controls_layout.setColumnStretch(1, 1)

    metric_label = QLabel("Metric:")
    controls_layout.addWidget(metric_label, 0, 0, alignment=Qt.AlignLeft)

    self.harmonic_metric_menu = QComboBox()
    self.harmonic_metric_menu.addItems(["SNR", "Z Score"])
    if hasattr(self.harmonic_metric_var, "get"):
        self.harmonic_metric_menu.setCurrentText(self.harmonic_metric_var.get())
        self.harmonic_metric_menu.currentTextChanged.connect(self.harmonic_metric_var.set)
    controls_layout.addWidget(self.harmonic_metric_menu, 0, 1)

    threshold_label = QLabel("Mean Threshold:")
    controls_layout.addWidget(threshold_label, 0, 2, alignment=Qt.AlignLeft)

    self.harmonic_threshold_entry = QLineEdit()
    self.harmonic_threshold_entry.setValidator(QDoubleValidator())
    if hasattr(self.harmonic_threshold_var, "get"):
        self.harmonic_threshold_entry.setText(self.harmonic_threshold_var.get())
        self.harmonic_threshold_entry.textChanged.connect(self.harmonic_threshold_var.set)
    controls_layout.addWidget(self.harmonic_threshold_entry, 0, 3)

    run_harmonic_btn = QPushButton("Run Harmonic Check")
    run_harmonic_btn.clicked.connect(self.run_harmonic_check)
    controls_layout.addWidget(run_harmonic_btn, 0, 4)

    self.export_harmonic_check_btn = QPushButton("Export Harmonic Results")
    self.export_harmonic_check_btn.setEnabled(False)
    self.export_harmonic_check_btn.clicked.connect(
        lambda: stats_export.export_significance_results_to_excel(
            findings_dict=self._structure_harmonic_results(),
            metric=self.harmonic_metric_var.get(),
            threshold=float(self.harmonic_threshold_var.get()),
            parent_folder=self.stats_data_folder_var.get(),
            log_func=self.log_to_main_app,
        )
    )
    controls_layout.addWidget(self.export_harmonic_check_btn, 0, 5)

    harmonic_layout.addWidget(controls_frame)
    main_layout.addWidget(harmonic_check_frame, 2, 0)

    # --- Row 3: Results Textbox ---
    self.results_textbox = QTextEdit()
    self.results_textbox.setReadOnly(True)
    self.results_textbox.setFont(QFont("Courier New", 12))
    main_layout.addWidget(self.results_textbox, 3, 0)
