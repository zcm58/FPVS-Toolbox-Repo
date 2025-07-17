# advanced_analysis_qt_processing.py
"""Processing mixin using QThread for the PySide6 advanced analysis window."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QMessageBox

from .advanced_analysis_core import run_advanced_averaging_processing
from Main_App.post_process import post_process as _external_post_process_actual

logger = logging.getLogger(__name__)


class _Worker(QObject):
    progress = Signal(float)
    log = Signal(str)
    finished = Signal()

    def __init__(
        self,
        defined_groups: list,
        main_app_params: dict,
        load_file_method,
        preprocess_raw_method,
        output_directory: str,
        pid_extractor,
        stop_event: threading.Event,
    ) -> None:
        super().__init__()
        self.defined_groups = defined_groups
        self.main_app_params = main_app_params
        self.load_file_method = load_file_method
        self.preprocess_raw_method = preprocess_raw_method
        self.output_directory = output_directory
        self.pid_extractor = pid_extractor
        self.stop_event = stop_event

    def run(self) -> None:
        try:
            run_advanced_averaging_processing(
                self.defined_groups,
                self.main_app_params,
                self.load_file_method,
                self.preprocess_raw_method,
                _external_post_process_actual,
                self.output_directory,
                self.pid_extractor,
                self.log.emit,
                self.progress.emit,
                self.stop_event,
            )
        except Exception as e:  # pragma: no cover - user display
            self.log.emit(f"!!! CRITICAL THREAD ERROR !!!\n{e}")
        finally:
            self.finished.emit()


class AdvancedAnalysisProcessingMixin:
    def _update_start_processing_button_state(self) -> None:
        all_valid = False
        if self.defined_groups:
            all_valid = all(
                g.get('config_saved') and g.get('file_paths') and g.get('condition_mappings')
                for g in self.defined_groups
            )
        self.start_btn.setEnabled(all_valid)

    def _validate_processing_setup(self) -> Optional[tuple]:
        if not self.defined_groups:
            QMessageBox.critical(self, "Error", "No averaging groups defined.")
            return None
        for i, group in enumerate(self.defined_groups):
            if not group.get('config_saved'):
                QMessageBox.critical(self, "Error", f"Group '{group['name']}' has unsaved changes.")
                self.groups_list.setCurrentRow(i)
                return None
            if not group.get('file_paths'):
                QMessageBox.critical(self, "Error", f"Group '{group['name']}' contains no files.")
                self.groups_list.setCurrentRow(i)
                return None
            if not group.get('condition_mappings'):
                QMessageBox.critical(self, "Error", f"Group '{group['name']}' has no mapping rules defined.")
                self.groups_list.setCurrentRow(i)
                return None
        params = getattr(self.master_app, 'validated_params', None)
        if not params:

            ok = False
            if hasattr(self.master_app, '_validate_inputs'):
                try:
                    ok = self.master_app._validate_inputs()
                except Exception as e:  # pragma: no cover - user display
                    QMessageBox.critical(self, "Error", f"Error validating main application inputs:\n{e}")
                    return None
                params = getattr(self.master_app, 'validated_params', None)
            if not ok or not params:
                QMessageBox.critical(self, "Error", "Main application parameters not set.")
                return None

        out_obj = getattr(self.master_app, 'save_folder_path', None)
        if not out_obj or not hasattr(out_obj, 'get'):
            QMessageBox.critical(self, "Error", "Main application output folder path is not configured.")
            return None
        output_directory = out_obj.get()
        return params, output_directory

    def _launch_processing_thread(self, main_app_params: Dict[str, Any], output_directory: str) -> None:
        self.log("Starting processing thread...")
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.close_btn.setEnabled(False)
        self._stop_requested.clear()

        if hasattr(self.master_app, 'load_eeg_file'):
            load_file_method = self.master_app.load_eeg_file
        else:
            from Main_App.load_utils import load_eeg_file as _load

            def load_file_method(fp):
                return _load(self.master_app, fp)

        if hasattr(self.master_app, 'preprocess_raw'):
            preprocess_raw_method = self.master_app.preprocess_raw
        else:
            from Main_App.eeg_preprocessing import perform_preprocessing as _pp

            def preprocess_raw_method(raw, **params):
                return _pp(raw_input=raw, params=params, log_func=self.master_app.log)[0]


        thread = QThread()
        worker = _Worker(

            self.defined_groups,
            main_app_params,
            load_file_method,
            preprocess_raw_method,
            output_directory,
            self._extract_pid_for_group,
            self._stop_requested,
        )

        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self.log)
        worker.progress.connect(lambda v: self.progress_bar.setValue(int(v * 100)))
        worker.finished.connect(lambda: self._on_worker_finished(thread, worker))
        self._active_threads.append((thread, worker))
        thread.start()

    def _on_worker_finished(self, thread: QThread, worker: QObject) -> None:
        if (thread, worker) in self._active_threads:
            self._active_threads.remove((thread, worker))
        thread.requestInterruption()
        thread.quit()
        thread.wait()
        worker.deleteLater()
        thread.deleteLater()

        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.progress_bar.hide()
        self._update_start_processing_button_state()
        self._stop_requested.clear()
        self.log("Processing finished.")

    def start_advanced_processing(self) -> None:
        self.log("Attempting to start advanced processing...")
        validation = self._validate_processing_setup()
        if not validation:
            return
        params, out_dir = validation
        self._launch_processing_thread(params, out_dir)

    def stop_processing(self) -> None:

        if not self._active_threads:
            self.log("Processing is not currently running.")
            return
        self._stop_requested.set()
        self.log("Stop requested. Waiting for processing to terminate...")
        self.stop_btn.setEnabled(False)
        for thread, _ in list(self._active_threads):
            thread.requestInterruption()
        # threads will shut down via _on_worker_finished