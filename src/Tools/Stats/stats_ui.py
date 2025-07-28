from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFileDialog,
)


class StatsToolWidget(QWidget):
    """Qt-based statistics tool placeholder.

    This widget mirrors the layout of the legacy CustomTkinter tool but does not
    yet implement the statistical logic. Slots are provided as TODOs for later
    wiring into the analysis backend.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("FPVS Statistical Analysis Tool")
        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # -- Data folder selection -------------------------------------------------
        folder_row = QHBoxLayout()
        main_layout.addLayout(folder_row)
        folder_row.addWidget(QLabel("Data Folder:"))
        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        folder_row.addWidget(self.folder_edit, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(browse_btn)

        self.scan_label = QLabel("Scan complete: Found 0 subjects and 0 conditions")
        main_layout.addWidget(self.scan_label)

        # -- Summed BCA Analysis ---------------------------------------------------
        summed_layout = QGridLayout()
        main_layout.addLayout(summed_layout)

        row_labels = [
            ("Run RM-ANOVA", self._run_rm_anova),
            ("Run Mixed Model", self._run_mixed_model),
            ("Run Interaction Post-hocs", self._run_posthocs),
        ]
        export_labels = [
            ("Export RM-ANOVA", self._export_rm_anova),
            ("Export Mixed Model", self._export_mixed_model),
            ("Export Post-hoc Results", self._export_posthoc),
        ]
        for i, (text, slot) in enumerate(row_labels):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            summed_layout.addWidget(btn, i, 0)
        for i, (text, slot) in enumerate(export_labels):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            summed_layout.addWidget(btn, i, 1)

        # -- Per-harmonic section --------------------------------------------------
        harm_layout = QHBoxLayout()
        main_layout.addLayout(harm_layout)
        harm_layout.addWidget(QLabel("Metric:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["SNR", "Z Score"])
        harm_layout.addWidget(self.metric_combo)
        harm_layout.addWidget(QLabel("Mean Threshold:"))
        self.threshold_edit = QLineEdit("1.96")
        harm_layout.addWidget(self.threshold_edit)
        run_harm_btn = QPushButton("Run Harmonic Check")
        run_harm_btn.clicked.connect(self._run_harmonic)
        harm_layout.addWidget(run_harm_btn)
        export_harm_btn = QPushButton("Export Harmonic Results")
        export_harm_btn.clicked.connect(self._export_harmonic)
        harm_layout.addWidget(export_harm_btn)
        harm_layout.addStretch(1)

        # -- Log output ------------------------------------------------------------
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        main_layout.addWidget(self.log_edit, 1)

    # ------------------------------------------------------------------
    # Placeholder slots -------------------------------------------------
    def _browse_folder(self) -> None:  # pragma: no cover - GUI stub
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if folder:
            self.folder_edit.setText(folder)
            # TODO: scan folder contents and update scan_label

    def _run_rm_anova(self) -> None:  # pragma: no cover - GUI stub
        # TODO: hook up RM-ANOVA processing
        pass

    def _run_mixed_model(self) -> None:  # pragma: no cover - GUI stub
        # TODO: hook up mixed model processing
        pass

    def _run_posthocs(self) -> None:  # pragma: no cover - GUI stub
        # TODO: hook up post-hoc testing
        pass

    def _export_rm_anova(self) -> None:  # pragma: no cover - GUI stub
        # TODO: export ANOVA results
        pass

    def _export_mixed_model(self) -> None:  # pragma: no cover - GUI stub
        # TODO: export mixed model results
        pass

    def _export_posthoc(self) -> None:  # pragma: no cover - GUI stub
        # TODO: export post-hoc results
        pass

    def _run_harmonic(self) -> None:  # pragma: no cover - GUI stub
        # TODO: perform per-harmonic significance checks
        pass

    def _export_harmonic(self) -> None:  # pragma: no cover - GUI stub
        # TODO: export harmonic results
        pass

