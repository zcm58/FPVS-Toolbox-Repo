"""Adapter between PySide6 GUI and legacy processing functions."""
from __future__ import annotations

from Main_App.Legacy_App.load_utils import load_eeg_file
from Main_App.Legacy_App.app_logic import preprocess_raw
from Main_App.Legacy_App.eeg_preprocessing import perform_preprocessing
from Main_App.Legacy_App.processing import process_data
from Main_App.Legacy_App.post_process import post_process


def run_full_pipeline(app, filepaths, output_dir, run_loreta, condition_labels):
    """Run the complete legacy pipeline for a sequence of files."""
    raws = [load_eeg_file(app, fp) for fp in filepaths]
    pre1 = [preprocess_raw(raw) for raw in raws if raw is not None]
    pre2 = [perform_preprocessing(p) for p in pre1 if p is not None]
    results = [process_data(p, output_dir, run_loreta) for p in pre2]
    for res in results:
        post_process(app, res, condition_labels)
    return results
