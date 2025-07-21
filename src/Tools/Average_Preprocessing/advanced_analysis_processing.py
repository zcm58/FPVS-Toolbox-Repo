"""Processing helpers for advanced averaging."""

from __future__ import annotations

import threading
import traceback
from typing import Any, Callable, Dict, Optional

from .advanced_analysis_base import AdvancedAnalysisBase
from .advanced_analysis_core import run_advanced_averaging_processing
from Main_App.post_process import post_process as _external_post_process_actual


class AdvancedAnalysisProcessingMixin(AdvancedAnalysisBase):
    """Mix-in running the averaging pipeline in a background thread."""

    def _thread_target(
        self,
        defined_groups: list,
        main_params: dict,
        load_file: Callable[[str], Any],
        preprocess_raw: Callable[[Any, Dict[str, Any]], Any],
        output_dir: str,
        log_callback: Callable[[str], None],
        progress_callback: Callable[[float], None],
        stop_event: threading.Event,
    ) -> None:
        try:
            run_advanced_averaging_processing(
                defined_groups,
                main_params,
                load_file,
                preprocess_raw,
                _external_post_process_actual,
                output_dir,
                self._extract_pid_for_group,
                log_callback,
                progress_callback,
                stop_event,
            )
        except Exception:
            err = traceback.format_exc()
            log_callback(f"Processing thread crashed:\n{err}")

    def start_processing(
        self,
        main_app_params: Dict[str, Any],
        load_file_func: Callable[[str], Any],
        preprocess_raw_func: Callable[[Any, Dict[str, Any]], Any],
        output_directory: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        if self.processing_thread and self.processing_thread.is_alive():
            raise RuntimeError("Processing already running")
        progress_cb = progress_callback or (lambda v: None)
        self._stop_requested.clear()
        self.processing_thread = threading.Thread(
            target=self._thread_target,
            args=(
                self.defined_groups,
                main_app_params,
                load_file_func,
                preprocess_raw_func,
                output_directory,
                self.log,
                progress_cb,
                self._stop_requested,
            ),
            daemon=True,
        )
        self.processing_thread.start()

    def stop_processing(self) -> None:
        if self.processing_thread and self.processing_thread.is_alive():
            self._stop_requested.set()
            self.processing_thread.join(timeout=1)
            self.processing_thread = None

