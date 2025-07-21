"""Qt-based UI builder for the Stats tool."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QValidator, QTextOption
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from . import stats_export


class _Button(QPushButton):
    """QPushButton with a ``configure`` helper for compatibility."""

    def configure(self, **kwargs) -> None:  # pragma: no cover - simple wrapper
        if "state" in kwargs:
            self.setEnabled(kwargs["state"] != "disabled")


class _ComboBox(QComboBox):
    """QComboBox with a ``configure`` helper for compatibility."""

    def configure(self, **kwargs) -> None:  # pragma: no cover - simple wrapper
        if "values" in kwargs:
            self.clear()
            self.addItems(list(kwargs["values"]))


class _TextEdit(QTextEdit):
    """QTextEdit mimicking Tk text widget API used elsewhere."""

    def configure(self, state: str | None = None) -> None:  # pragma: no cover - wrapper
        if state is not None:
            self.setReadOnly(state == "disabled")

    def delete(self, *_args) -> None:  # pragma: no cover - wrapper
        self.clear()

    def insert(self, _index: str, text: str) -> None:  # pragma: no cover - wrapper
        self.setPlainText(text)


class _NumericValidator(QValidator):
    """Validator replicating ``_validate_numeric`` behaviour."""

    def validate(self, text: str, pos: int):  # type: ignore[override]
        if text in ("", "-"):
            return QValidator.Acceptable, text, pos
        try:
            float(text)
            return QValidator.Acceptable, text, pos
        except ValueError:
            return QValidator.Invalid, text, pos



def create_widgets(self) -> None:
    """Build the Stats tool widgets using PySide6."""

    layout = QGridLayout(self)
    layout.setContentsMargins(10, 10, 10, 10)

    main_frame = QFrame(self)
    layout.addWidget(main_frame, 0, 0)
    layout.setRowStretch(0, 1)
    layout.setColumnStretch(0, 1)

    main_layout = QGridLayout(main_frame)
    main_layout.setColumnStretch(0, 1)

    # --- Row 0: Folder Selection ---
    folder_frame = QFrame(main_frame)
    folder_layout = QGridLayout(folder_frame)
    folder_layout.setColumnStretch(1, 1)
    main_layout.addWidget(folder_frame, 0, 0)

    folder_layout.addWidget(QLabel("Data Folder:"), 0, 0)

    self.folder_entry = QLineEdit()
    self.folder_entry.setReadOnly(True)
    self.folder_entry.setText(self.stats_data_folder_var.get())
    folder_layout.addWidget(self.folder_entry, 0, 1)

    browse_btn = _Button("Browse...")
    browse_btn.clicked.connect(self.browse_folder)
    folder_layout.addWidget(browse_btn, 0, 2)

    self.detected_label = QLabel(self.detected_info_var.get())
    self.detected_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    folder_layout.addWidget(self.detected_label, 1, 0, 1, 3)

    if hasattr(self.stats_data_folder_var, "trace_add"):
        self.stats_data_folder_var.trace_add(
            "write",
            lambda *_: self.folder_entry.setText(self.stats_data_folder_var.get()),
        )
    if hasattr(self.detected_info_var, "trace_add"):
        self.detected_info_var.trace_add(
            "write",
            lambda *_: self.detected_label.setText(self.detected_info_var.get()),
        )

    # --- Row 1: Section A - Summed BCA Analysis ---
    summed_bca_frame = QFrame(main_frame)
    summed_layout = QVBoxLayout(summed_bca_frame)
    main_layout.addWidget(summed_bca_frame, 1, 0)

    heading = QLabel("Summed BCA Analysis:")
    heading.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
    summed_layout.addWidget(heading, alignment=Qt.AlignLeft)

    buttons_frame = QFrame(summed_bca_frame)
    btn_layout = QGridLayout(buttons_frame)
    btn_layout.setColumnStretch(0, 1)
    btn_layout.setColumnStretch(1, 1)
    summed_layout.addWidget(buttons_frame)

    btn_rm = _Button("Run RM-ANOVA (Summed BCA)")
    btn_rm.clicked.connect(self.run_rm_anova)
    btn_layout.addWidget(btn_rm, 0, 0)

    btn_mixed = _Button("Run Mixed Model")
    btn_mixed.clicked.connect(self.run_mixed_model)
    btn_layout.addWidget(btn_mixed, 1, 0)

    self.run_posthoc_btn = _Button("Run Interaction Post-hocs")
    self.run_posthoc_btn.clicked.connect(self.run_interaction_posthocs)
    btn_layout.addWidget(self.run_posthoc_btn, 2, 0)

    self.export_rm_anova_btn = _Button("Export RM-ANOVA")
    self.export_rm_anova_btn.setEnabled(False)
    self.export_rm_anova_btn.clicked.connect(
        lambda: stats_export.export_rm_anova_results_to_excel(
            anova_table=self.rm_anova_results_data,
            parent_folder=self.stats_data_folder_var.get(),
            log_func=self.log_to_main_app,
        )
    )
    btn_layout.addWidget(self.export_rm_anova_btn, 0, 1)

    self.export_mixed_model_btn = _Button("Export Mixed Model")
    self.export_mixed_model_btn.setEnabled(False)
    self.export_mixed_model_btn.clicked.connect(
        lambda: stats_export.export_mixed_model_results_to_excel(
            results_df=self.mixed_model_results_data,
            parent_folder=self.stats_data_folder_var.get(),
            log_func=self.log_to_main_app,
        )
    )
    btn_layout.addWidget(self.export_mixed_model_btn, 1, 1)

    self.export_posthoc_btn = _Button("Export Post-hoc Results")
    self.export_posthoc_btn.setEnabled(False)
    self.export_posthoc_btn.clicked.connect(
        lambda: stats_export.export_posthoc_results_to_excel(
            results_df=self.posthoc_results_data,
            factor=self.posthoc_factor_var.get(),
            parent_folder=self.stats_data_folder_var.get(),
            log_func=self.log_to_main_app,
        )
    )
    btn_layout.addWidget(self.export_posthoc_btn, 2, 1)

    # --- Row 2: Section B - Harmonic Significance Check ---
    harmonic_frame = QFrame(main_frame)
    harmonic_layout = QVBoxLayout(harmonic_frame)
    main_layout.addWidget(harmonic_frame, 2, 0)

    lbl_harmonic = QLabel("Per-Harmonic Significance Check:")
    lbl_harmonic.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
    harmonic_layout.addWidget(lbl_harmonic, alignment=Qt.AlignLeft)

    controls_frame = QFrame(harmonic_frame)
    controls_layout = QGridLayout(controls_frame)
    controls_layout.setColumnStretch(1, 1)
    harmonic_layout.addWidget(controls_frame)

    controls_layout.addWidget(QLabel("Metric:"), 0, 0)

    self.harmonic_metric_menu = _ComboBox()
    self.harmonic_metric_menu.addItems(["SNR", "Z Score"])
    self.harmonic_metric_menu.setCurrentText(self.harmonic_metric_var.get())
    self.harmonic_metric_menu.currentTextChanged.connect(
        lambda val: self.harmonic_metric_var.set(val)
    )
    controls_layout.addWidget(self.harmonic_metric_menu, 0, 1)

    controls_layout.addWidget(QLabel("Mean Threshold:"), 0, 2)

    self.harmonic_threshold_edit = QLineEdit(self.harmonic_threshold_var.get())
    self.harmonic_threshold_edit.setValidator(_NumericValidator())
    self.harmonic_threshold_edit.textChanged.connect(
        lambda text: self.harmonic_threshold_var.set(text)
    )
    controls_layout.addWidget(self.harmonic_threshold_edit, 0, 3)

    run_check_btn = _Button("Run Harmonic Check")
    run_check_btn.clicked.connect(self.run_harmonic_check)
    controls_layout.addWidget(run_check_btn, 0, 4)

    self.export_harmonic_check_btn = _Button("Export Harmonic Results")
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

    # --- Row 3: Results Textbox ---
    self.results_textbox = _TextEdit()
    self.results_textbox.setReadOnly(True)
    self.results_textbox.setWordWrapMode(QTextOption.WrapMode.WordWrap)
    self.results_textbox.setFont(QFont("Courier New", 12))
    main_layout.addWidget(self.results_textbox, 3, 0)
    main_layout.setRowStretch(3, 1)
