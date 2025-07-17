"""Standalone demo for the PySide6 Advanced Analysis window."""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from PySide6.QtWidgets import QApplication

from .advanced_analysis_qt import AdvancedAnalysisWindow
from .advanced_analysis_core import run_advanced_averaging_processing

logger = logging.getLogger(__name__)


class DummyMasterApp:
    def __init__(self) -> None:
        self.validated_params = {
            "low_pass": 0.1,
            "high_pass": 50,
            "epoch_start": -1,
            "epoch_end": 2,
            "stim_channel": "Status",
        }
        self.save_folder_path = type("Obj", (), {"get": lambda self: str(Path(os.getcwd()) / "test_adv_output_qt")})()
        os.makedirs(self.save_folder_path.get(), exist_ok=True)

    def log(self, message: str) -> None:
        logger.info("[DummyMasterLog] %s", message)

    def load_eeg_file(self, filepath):
        logger.info("DummyLoad: %s", filepath)
        return "dummy_raw_obj_for_test"

    def preprocess_raw(self, raw, **params):
        logger.info("DummyPreproc of '%s' with %s", raw, params)
        return "dummy_proc_raw_obj_for_test"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    master = DummyMasterApp()
    if run_advanced_averaging_processing is None:
        logger.critical("CRITICAL ERROR: core processing not imported.")
    win = AdvancedAnalysisWindow(master)
    win.show()
    app.exec()
