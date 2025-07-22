from __future__ import annotations

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QToolBar,
    QPushButton,
    QLabel,
    QGroupBox,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QRadioButton,
    QCheckBox,
    QLineEdit,
    QScrollArea,
    QTextEdit,
    QStatusBar,
    QFileDialog,
    QMessageBox,
    QButtonGroup,
    QDockWidget,
    QToolButton,
    QSizePolicy,
    QStackedWidget,
    QGraphicsOpacityEffect,
)
from PySide6.QtCore import Qt, QPropertyAnimation
from PySide6.QtGui import QIcon
import logging
import pandas as pd
from pathlib import Path
import subprocess
import sys

from Main_App.GUI.menu_bar import build_menu_bar
from Main_App.GUI.settings_panel import SettingsDialog
from Main_App.settings_manager import SettingsManager

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window implemented with PySide6."""

    def __init__(self) -> None:
        super().__init__()
        self.settings = SettingsManager()
        self.output_folder: str = ""
        self.data_paths: list[str] = []
        self.event_rows: list[QWidget] = []
        self._settings_dialog = None
        self.setWindowTitle("FPVS Toolbox")
        self.setMinimumSize(1024, 768)
        self.initSidebar()
        self.stack = QStackedWidget(self)
        self.setCentralWidget(self.stack)
        self.homeWidget = self._init_ui()
        self.settingsWidget = SettingsDialog(self.settings, self)
        self.statsToolWidget = QLabel("Stats Tool Page", self)
        self.snrPlotWidget = QLabel("Graphs Page", self)
        self.imageResizerWidget = QLabel("Image Resizer Page", self)
        self.epochAveragingWidget = QLabel("Epoch Averaging Page", self)
        for w in (
            self.homeWidget,
            self.statsToolWidget,
            self.snrPlotWidget,
            self.imageResizerWidget,
            self.epochAveragingWidget,
            self.settingsWidget,
        ):
            self.stack.addWidget(w)
        self.setStatusBar(QStatusBar(self))
        self._connect_sidebar()
        self.switchPage(0)
        self.log("Welcome to the FPVS Toolbox!")
        self.log(
            f"Appearance Mode: {self.settings.get('appearance', 'mode', 'System')}"
        )

    # ------------------------------------------------------------------
    def _init_ui(self) -> QWidget:
        # Menu bar
        menu = build_menu_bar(self)
        self.setMenuBar(menu)

        # Top toolbar
        toolbar = QToolBar(self)
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        self.btn_open_eeg = QPushButton("Select EEG File…", self)
        self.btn_open_output = QPushButton("Select Output Folder…", self)
        self.btn_start = QPushButton("Start Processing", self)
        self.lbl_debug = QLabel("DEBUG MODE ENABLED", self)
        self.lbl_debug.setStyleSheet("color: red;")
        self.lbl_debug.setVisible(self.settings.debug_enabled())
        for w in (
            self.btn_open_eeg,
            self.btn_open_output,
            self.btn_start,
            self.lbl_debug,
        ):
            toolbar.addWidget(w)

        # Central container
        container = QWidget(self)
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # Processing Options group
        grp_proc = QGroupBox("Processing Options", container)
        gl = QGridLayout(grp_proc)
        gl.addWidget(QLabel("Mode:"), 0, 0)
        self.rb_single = QRadioButton("Single File", grp_proc)
        self.rb_batch = QRadioButton("Batch Folder", grp_proc)
        gl.addWidget(self.rb_single, 0, 1)
        gl.addWidget(self.rb_batch, 0, 2)
        self.cb_loreta = QCheckBox("Run LORETA during processing?", grp_proc)
        gl.addWidget(self.cb_loreta, 1, 0, 1, 3)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.rb_single)
        self.mode_group.addButton(self.rb_batch)
        self.rb_single.toggled.connect(
            lambda checked: checked and self._on_mode_changed("single")
        )
        self.rb_batch.toggled.connect(
            lambda checked: checked and self._on_mode_changed("batch")
        )
        main_layout.addWidget(grp_proc)

        # Load saved processing options
        mode = self.settings.get("processing", "mode", "batch").lower()
        if mode == "batch":
            self.rb_batch.setChecked(True)
        else:
            self.rb_single.setChecked(True)

        loreta_enabled = (
            self.settings.get("processing", "run_loreta", "False").lower() == "true"
        )
        self.cb_loreta.setChecked(loreta_enabled)

        # Preprocessing placeholder
        main_layout.addWidget(QLabel("Configure Preprocessing in Settings.", container))

        # Event Map group
        grp_event = QGroupBox(
            "Event Map (Condition Label → Numerical ID)", container
        )
        vlay = QVBoxLayout(grp_event)
        scroll = QScrollArea(grp_event)
        scroll.setWidgetResizable(True)
        self.event_container = QWidget()
        self.event_layout = QVBoxLayout(self.event_container)
        scroll.setWidget(self.event_container)
        vlay.addWidget(scroll)
        btns = QHBoxLayout()
        self.btn_detect = QPushButton("Detect Trigger IDs", grp_event)
        self.btn_add_row = QPushButton("+ Add Condition", grp_event)
        btns.addWidget(self.btn_detect)
        btns.addWidget(self.btn_add_row)
        vlay.addLayout(btns)
        main_layout.addWidget(grp_event)

        # Populate saved event map rows
        saved_pairs = self.settings.get_event_pairs()
        if saved_pairs:
            for label, ident in saved_pairs:
                self.add_event_row(label, ident)
        else:
            self.add_event_row()

        # Log group
        grp_log = QGroupBox("Log", container)
        lay_log = QVBoxLayout(grp_log)
        self.text_log = QTextEdit(grp_log)
        self.text_log.setReadOnly(True)
        lay_log.addWidget(self.text_log)
        main_layout.addWidget(grp_log)

        # Finalize
        self.homeWidget = container

        # Connect toolbar buttons to methods
        self.btn_open_eeg.clicked.connect(self.select_eeg_file)
        self.btn_open_output.clicked.connect(self.select_output_folder)
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_add_row.clicked.connect(lambda: self.add_event_row())
        self.btn_detect.clicked.connect(self.detect_trigger_ids)

        # Sync the select button label with the current mode
        self._update_select_button_text()
        return container

    # ------------------------------------------------------------------
    def initSidebar(self) -> None:
        """Create the dark sidebar with tool buttons."""
        sidebar = QWidget(self)
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(200)
        lay = QVBoxLayout(sidebar)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.sidebar_buttons = []

        def make_button(name: str, text: str, icon_name: str) -> QToolButton:
            btn = QToolButton()
            btn.setObjectName(name)
            btn.setText(text)
            btn.setIcon(QIcon.fromTheme(icon_name))
            btn.setCheckable(True)
            btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.sidebar_buttons.append(btn)
            lay.addWidget(btn)
            return btn

        self.btn_home = make_button("btn_home", "Home", "go-home")
        self.btn_data = make_button("btn_data", "Data Analysis", "view-statistics")
        self.btn_graphs = make_button("btn_graphs", "Graphs", "view-media-visualization")
        self.btn_image = make_button("btn_image", "Image Resizer", "camera-photo")
        self.btn_epoch = make_button("btn_epoch", "Epoch Averaging", "view-refresh")

        lay.addStretch(1)

        self.btn_settings = make_button("btn_settings", "Settings", "settings")
        self.btn_info = make_button("btn_info", "Information", "help-about")
        self.btn_help = make_button("btn_help", "Help", "help-contents")

        dock = QDockWidget("", self)
        dock.setWidget(sidebar)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    # ------------------------------------------------------------------
    def _connect_sidebar(self) -> None:
        buttons = [
            self.btn_home,
            self.btn_data,
            self.btn_graphs,
            self.btn_image,
            self.btn_epoch,
            self.btn_settings,
        ]
        for i, btn in enumerate(buttons):
            btn.clicked.connect(lambda _, ix=i: self.switchPage(ix))

    # ------------------------------------------------------------------
    def switchPage(self, index: int) -> None:
        old = self.stack.currentWidget()
        new = self.stack.widget(index)
        if old is new:
            return
        for i, btn in enumerate([
            self.btn_home,
            self.btn_data,
            self.btn_graphs,
            self.btn_image,
            self.btn_epoch,
            self.btn_settings,
        ]):
            btn.setChecked(i == index)
        old_effect = QGraphicsOpacityEffect(old)
        old.setGraphicsEffect(old_effect)
        anim_out = QPropertyAnimation(old_effect, b"opacity")
        anim_out.setDuration(200)
        anim_out.setStartValue(1.0)
        anim_out.setEndValue(0.0)

        def _finish() -> None:
            self.stack.setCurrentIndex(index)
            new_effect = QGraphicsOpacityEffect(new)
            new.setGraphicsEffect(new_effect)
            anim_in = QPropertyAnimation(new_effect, b"opacity")
            anim_in.setDuration(200)
            anim_in.setStartValue(0.0)
            anim_in.setEndValue(1.0)
            anim_in.finished.connect(lambda: new.setGraphicsEffect(None))
            anim_in.start()

        anim_out.finished.connect(_finish)
        anim_out.start()

    # ------------------------------------------------------------------
    def log(self, message: str, level: int = logging.INFO) -> None:
        ts = pd.Timestamp.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"{ts} [GUI]: {message}"
        if hasattr(self, "text_log") and self.text_log:
            self.text_log.append(formatted)
        logger.log(level, message)

    def debug(self, message: str) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            self.log(f"[DEBUG] {message}", level=logging.DEBUG)

    # ------------------------------------------------------------------
    def _on_mode_changed(self, mode: str) -> None:
        self.settings.set("processing", "mode", mode)
        self.settings.save()
        self._update_select_button_text()

    def _update_select_button_text(self) -> None:
        """Update the label of the EEG selection button based on the mode."""
        if self.rb_batch.isChecked():
            self.btn_open_eeg.setText("Select Input Folder…")
        else:
            self.btn_open_eeg.setText("Select .BDF File…")

    # ------------------------------------------------------------------
    def open_settings_window(self) -> None:
        self.switchPage(5)
        self.lbl_debug.setVisible(self.settings.debug_enabled())

    def check_for_updates(self) -> None:
        from Main_App import update_manager

        update_manager.check_for_updates_async(self, silent=False)

    def quit(self) -> None:
        self.close()

    def open_stats_analyzer(self) -> None:
        QMessageBox.information(
            self,
            "Stats Toolbox",
            "The statistics tool is not yet available in the Qt interface.",
        )

    def open_image_resizer(self) -> None:
        cmd = [sys.executable]
        if not getattr(sys, "frozen", False):
            cmd.append(str(Path(__file__).resolve().parents[2] / "main.py"))
        cmd.append("--run-image-resizer")
        subprocess.Popen(cmd, close_fds=True)

    def open_plot_generator(self) -> None:
        cmd = [sys.executable]
        if not getattr(sys, "frozen", False):
            cmd.append(str(Path(__file__).resolve().parents[2] / "main.py"))
        cmd.append("--run-plot-generator")
        subprocess.Popen(cmd, close_fds=True)

    def open_advanced_analysis_window(self) -> None:
        QMessageBox.information(
            self,
            "Advanced Analysis",
            "The advanced preprocessing tool is not yet available in the Qt interface.",
        )

    def show_relevant_publications(self) -> None:
        QMessageBox.information(
            self,
            "Relevant Publications",
            "This dialog is not yet implemented in the Qt interface.",
        )

    def show_about_dialog(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        from config import FPVS_TOOLBOX_VERSION

        QMessageBox.information(
            self,
            "About FPVS ToolBox",
            f"Version: {FPVS_TOOLBOX_VERSION} was developed by Zack Murphy at Mississippi State University.",
        )

    # ------------------------------------------------------------------
    def select_eeg_file(self) -> None:  # pragma: no cover - GUI stub
        """Prompt the user to select a single .BDF file or an input folder."""
        if self.rb_batch.isChecked():
            folder = QFileDialog.getExistingDirectory(
                self,
                "Select Input Folder",
                self.settings.get("paths", "data_folder", ""),
            )
            if folder:
                bdf_files = sorted(Path(folder).glob("*.bdf"))
                if bdf_files:
                    self.data_paths = [str(p) for p in bdf_files]
                    self.log(
                        f"Selected folder: {folder}, Found {len(bdf_files)} '.bdf' file(s)."
                    )
                else:
                    self.log(f"No '.bdf' files found in {folder}.")
                    QMessageBox.warning(
                        self, "No Files Found", f"No '.bdf' files found in:\n{folder}"
                    )
            else:
                self.log("No folder selected.")
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select EEG File",
                self.settings.get("paths", "data_folder", ""),
                "BDF Files (*.bdf);;All Files (*)",
            )
            if file_path:
                self.data_paths = [file_path]
                self.log(f"Selected file: {Path(file_path).name}")
            else:
                self.log("No file selected.")

    def select_output_folder(self) -> None:  # pragma: no cover - GUI stub
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self.output_folder or ""
        )
        if folder:
            self.output_folder = folder
            self.log(f"Output folder set: {folder}")

    def start_processing(self) -> None:  # pragma: no cover - GUI stub
        self.log("start_processing() stub")

    def detect_trigger_ids(self) -> None:  # pragma: no cover - GUI stub
        try:
            from Tools.Event_Map import event_map_utils

            pairs = event_map_utils.detect_trigger_ids(self.data_paths)
        except Exception as err:
            QMessageBox.warning(self, "Detection Failed", str(err))
            return

        added = 0
        for lbl, ident in pairs:
            self.add_event_row(str(lbl), str(ident))
            added += 1
        self.log(f"Detected and added {added} trigger ID(s)")

    def add_event_row(self, label: str = "", id: str = "") -> None:
        row = QWidget(self.event_container)
        hl = QHBoxLayout(row)
        le_label = QLineEdit(label, row)
        le_id = QLineEdit(id, row)
        btn_rm = QPushButton("✕", row)
        def _remove() -> None:
            row.setParent(None)
            self.log("Event map row removed.")
        btn_rm.clicked.connect(_remove)
        hl.addWidget(le_label)
        hl.addWidget(le_id)
        hl.addWidget(btn_rm)
        self.event_layout.addWidget(row)
        self.event_rows.append(row)
        self.log("Added event map row")


def main() -> None:
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
