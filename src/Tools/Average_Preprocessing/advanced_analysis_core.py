# advanced_analysis_core.py
"""Core processing logic for the advanced averaging window.

This module implements functions that take the groups configured in the
``advanced_analysis`` GUI and perform the actual averaging, leveraging the main
FPVS application's loading, preprocessing and post-processing callbacks.
"""

import gc
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Union, Tuple

import numpy as np

import mne
import threading
from Main_App.settings_manager import SettingsManager

# Set up module level logger
logger = logging.getLogger(__name__)
DEBUG_ENABLED = SettingsManager().debug_enabled()


class PostProcessContextForAdvanced:
    """
    A context object to pass to the _external_post_process function,
    mimicking the necessary parts of the FPVSApp instance for averaged data.
    """

    def __init__(self,
                 log_callback: Callable[[str], None],
                 save_folder_path_str: str,
                 pid_for_output: str,  # This will be the participant's specific PID (e.g., "P1")
                 averaged_epochs_dict: Dict[str, List[Union[mne.Epochs, mne.Evoked]]],
                 condition_name_for_output: str):  # This is the recipe name (e.g., "Average A")
        self.log_callback = log_callback
        self.save_folder_path_str = save_folder_path_str

        # Store participant PID using the attribute name post_process.py expects for "group" PIDs
        self.pid_for_group = pid_for_output
        self.averaged_epochs_dict = averaged_epochs_dict
        # Store recipe name using the attribute name post_process.py expects for "group" names
        self.group_name_for_output = condition_name_for_output

    def log(self, msg: str) -> None:
        self.log_callback(f"[AdvCorePostProc] {msg}")

    @property
    def save_folder_path(self):
        class Getter:
            def __init__(self, val): self._val = val

            def get(self): return self._val

        return Getter(self.save_folder_path_str)

    @property
    def data_paths(self) -> List[str]:
        # This dummy path helps post_process.py form a unique base if needed,
        # reflecting the participant and condition.
        return [f"{self.pid_for_group}_{self.group_name_for_output}_averaged.fif"]

    @property
    def preprocessed_data(self) -> Dict[str, List[Union[mne.Epochs, mne.Evoked]]]:
        return self.averaged_epochs_dict


def run_advanced_averaging_processing(
        defined_groups: List[Dict[str, Any]],
        main_app_params: Dict[str, Any],
        main_app_load_file_func: Callable[[str], Optional[mne.io.Raw]],
        main_app_preprocess_raw_func: Callable[[mne.io.Raw, Dict[str, Any]], Optional[mne.io.Raw]],
        external_post_process_func: Callable[[Any, List[str]], None],
        output_dir_str: str,
        pid_extraction_func: Callable[[Dict[str, Any]], str],  # Expects group-like data
        log_callback: Callable[[str], None],
        progress_callback: Callable[[float], None],
        stop_event: threading.Event
) -> bool:
    """Run advanced averaging for each participant using provided groups.

    Parameters
    ----------
    defined_groups : List[Dict[str, Any]]
        Group definitions including file paths and mapping rules.
    main_app_params : Dict[str, Any]
        Validated parameters from the main application.
    main_app_load_file_func : Callable[[str], Optional[mne.io.Raw]]
        Function used to load raw EEG files.
    main_app_preprocess_raw_func : Callable[[mne.io.Raw, Dict[str, Any]], Optional[mne.io.Raw]]
        Function performing preprocessing on loaded data.
    external_post_process_func : Callable[[Any, List[str]], None]
        Callback invoked after averaging to run post-processing.
    output_dir_str : str
        Directory where results should be written.
    pid_extraction_func : Callable[[Dict[str, Any]], str]
        Function returning a participant ID based on group data.
    log_callback : Callable[[str], None]
        Callback used for logging messages to the UI.
    progress_callback : Callable[[float], None]
        Callback for updating progress in the UI.
    stop_event : threading.Event
        Event used to cancel processing.

    Returns
    -------
    bool
        ``True`` if processing completed without major errors, otherwise ``False``.
    """

    def _debug(msg: str) -> None:
        if DEBUG_ENABLED:
            log_callback(f"[DEBUG] {msg}")
            logger.debug(msg)

    log_callback("Starting advanced averaging core processing (per-participant output)...")
    if DEBUG_ENABLED:
        logger.debug("Advanced averaging core processing started with %d group definitions", len(defined_groups))

    # Helper functions -----------------------------------------------------
    def load_and_preprocess(fpath: str) -> Optional[mne.io.Raw]:
        raw = main_app_load_file_func(fpath)
        if raw is None or stop_event.is_set():
            return None
        processed = main_app_preprocess_raw_func(raw.copy(), **main_app_params.copy())
        del raw
        return processed

    def get_events(raw: mne.io.Raw) -> np.ndarray:
        stim_channel = main_app_params.get('stim_channel', 'Status')
        return mne.find_events(raw, stim_channel=stim_channel, consecutive=True, verbose=False)

    def get_epochs(raw: mne.io.Raw, events: np.ndarray, event_id: int) -> mne.Epochs:
        return mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=main_app_params['epoch_start'],
            tmax=main_app_params['epoch_end'],
            preload=True,
            baseline=None,
            verbose=False,
            on_missing='warn',
        )

    # Estimate total operations for progress bar
    total_source_ops = 0
    total_participant_post_ops = 0

    for group_data_template in defined_groups:
        unique_files_in_group = list(set(group_data_template.get('file_paths', [])))
        total_participant_post_ops += len(unique_files_in_group)
        for participant_file_path_for_estimate in unique_files_in_group:
            for mapping_rule_template in group_data_template.get('condition_mappings', []):
                for source_info_template in mapping_rule_template.get('sources', []):
                    if source_info_template['file_path'] == participant_file_path_for_estimate:
                        total_source_ops += 1

    total_operations_estimate = total_source_ops + total_participant_post_ops
    if total_operations_estimate == 0:
        total_operations_estimate = 1
    _debug(f"Total operation estimate: {total_operations_estimate}")
    current_operation_count = 0
    overall_success = True

    # Count how many times each file and event ID will be used
    file_usage_counts: Dict[str, int] = {}
    event_usage_counts: Dict[Tuple[str, int], int] = {}
    events_needed_per_file: Dict[str, set[int]] = {}
    for group in defined_groups:
        for mapping in group.get('condition_mappings', []):
            for source_info in mapping.get('sources', []):
                fpath = source_info['file_path']
                eid = source_info['original_id']
                file_usage_counts[fpath] = file_usage_counts.get(fpath, 0) + 1
                event_usage_counts[(fpath, eid)] = event_usage_counts.get((fpath, eid), 0) + 1
                events_needed_per_file.setdefault(fpath, set()).add(eid)

    # Caches ---------------------------------------------------------------
    preprocessed_cache: Dict[str, mne.io.Raw] = {}
    events_cache: Dict[str, np.ndarray] = {}
    epochs_cache: Dict[Tuple[str, int], Optional[mne.Epochs]] = {}

    # Single-pass preprocessing phase -------------------------------------
    for fpath, event_ids in events_needed_per_file.items():
        if stop_event.is_set():
            return False

        file_basename = Path(fpath).name
        log_callback(f"Preprocessing {file_basename} and extracting epochs...")
        try:
            raw_proc = load_and_preprocess(fpath)
            if raw_proc is None:
                log_callback(f"  Skipping {file_basename} (load or preprocess failed).")
                overall_success = False
                for eid in event_ids:
                    epochs_cache[(fpath, eid)] = None
                continue

            preprocessed_cache[fpath] = raw_proc
            events = get_events(raw_proc)
            events_cache[fpath] = events

            for eid in event_ids:
                if eid not in events[:, 2]:
                    epochs_cache[(fpath, eid)] = None
                    continue
                epochs_cache[(fpath, eid)] = get_epochs(raw_proc, events, eid)
        except Exception as e:
            logger.exception("Error during preprocessing for %s", file_basename)
            log_callback(
                f"  !!! Error preprocessing {file_basename}: {e}\n{traceback.format_exc()}"
            )
            overall_success = False
            for eid in event_ids:
                epochs_cache[(fpath, eid)] = None

    for group_definition in defined_groups:  # A "group definition" is like a recipe, e.g., "Average A"
        if stop_event.is_set():
            log_callback("Processing cancelled by user.")
            return False

        group_recipe_name = group_definition['name']  # e.g., "Average A"
        log_callback(f"--- Applying Group Recipe: {group_recipe_name} ---")
        _debug(f"Processing group '{group_recipe_name}' with {len(group_definition.get('file_paths', []))} files")

        unique_participant_files = sorted(list(set(group_definition.get('file_paths', []))))
        if not unique_participant_files:
            log_callback(f"  No files specified for recipe '{group_recipe_name}'. Skipping.")
            continue

        # Loop through each unique participant file associated with this group recipe
        for participant_file_path in unique_participant_files:
            if stop_event.is_set():
                return False

            participant_pid = pid_extraction_func({'file_paths': [participant_file_path]})
            log_callback(
                f"  -- Processing Participant: {participant_pid} (File: {Path(participant_file_path).name}) for Recipe: '{group_recipe_name}' --")
            _debug(f"Participant {participant_pid} file {participant_file_path}")

            averaged_data_for_this_participant: Dict[str, List[Union[mne.Epochs, mne.Evoked]]] = {}

            for mapping_rule in group_definition.get('condition_mappings', []):
                if stop_event.is_set():
                    return False

                output_label_for_average = mapping_rule['output_label']
                log_callback(f"    Applying rule for: '{output_label_for_average}' to participant {participant_pid}")
                _debug(f"Rule '{output_label_for_average}' with {len(mapping_rule.get('sources', []))} sources")

                epochs_for_this_participant_this_rule: List[mne.Epochs] = []

                for source_info in mapping_rule.get('sources', []):
                    if stop_event.is_set():
                        return False

                    if source_info['file_path'] != participant_file_path:
                        continue

                    file_path_of_source = source_info['file_path']
                    original_event_id = source_info['original_id']
                    file_basename = Path(file_path_of_source).name

                    log_callback(
                        f"      Retrieving cached epochs: {file_basename} ID {original_event_id}")

                    key = (file_path_of_source, original_event_id)
                    participant_source_epochs = None
                    try:
                        cached_epochs = epochs_cache.get(key)
                        if cached_epochs is None:
                            log_callback(
                                f"      Warning: Event ID {original_event_id} not found in {file_basename}. Skipping this event for {participant_pid}.")
                            continue

                        if len(cached_epochs) > 0:
                            participant_source_epochs = cached_epochs.copy()
                            log_callback(
                                f"        Reused {len(participant_source_epochs)} epochs from {file_basename} for ID {original_event_id}.")
                            epochs_for_this_participant_this_rule.append(participant_source_epochs)
                        else:
                            log_callback(
                                f"        No epochs created from {file_basename} for ID {original_event_id}.")
                    except Exception as e:
                        logger.exception("Error during epoch retrieval for %s", file_basename)
                        log_callback(
                            f"      !!! Error retrieving epochs for {file_basename}, ID {original_event_id} for {participant_pid}: {e}\n{traceback.format_exc()}")
                        overall_success = False
                    finally:
                        if participant_source_epochs is not None:
                            del participant_source_epochs
                        current_operation_count += 1
                        progress_callback(current_operation_count / total_operations_estimate)
                        file_usage_counts[file_path_of_source] -= 1
                        event_usage_counts[key] -= 1
                        if event_usage_counts[key] == 0 and key in epochs_cache:
                            del epochs_cache[key]
                        if file_usage_counts[file_path_of_source] == 0:
                            if file_path_of_source in preprocessed_cache:
                                del preprocessed_cache[file_path_of_source]
                            if file_path_of_source in events_cache:
                                del events_cache[file_path_of_source]
                            gc.collect()

                if epochs_for_this_participant_this_rule:
                    averaging_method = group_definition.get('averaging_method', "Pool Trials")
                    log_callback(
                        f"    Averaging {len(epochs_for_this_participant_this_rule)} sets of epochs for '{output_label_for_average}' (Participant: {participant_pid}, Method: {averaging_method})")
                    _debug(f"Averaging method '{averaging_method}' on {len(epochs_for_this_participant_this_rule)} epoch sets")
                    try:
                        if averaging_method == "Pool Trials":
                            concatenated_epochs = mne.concatenate_epochs(epochs_for_this_participant_this_rule)
                            averaged_evoked = concatenated_epochs.average()
                            averaged_data_for_this_participant[output_label_for_average] = [averaged_evoked]
                            log_callback(
                                f"      Pooled {len(concatenated_epochs)} total trials for '{output_label_for_average}'.")
                        elif averaging_method == "Average of Averages":
                            evokeds_to_average = [ep.average() for ep in epochs_for_this_participant_this_rule if
                                                  len(ep) > 0]
                            if evokeds_to_average:
                                averaged_evoked = mne.grand_average(evokeds_to_average,
                                                                    interpolate_bads=False)
                                averaged_data_for_this_participant[output_label_for_average] = [averaged_evoked]
                                log_callback(
                                    f"      Averaged {len(evokeds_to_average)} evoked responses for '{output_label_for_average}'.")
                                _debug("Grand averaged evoked responses")
                            else:
                                log_callback(
                                    f"      No valid evoked responses to average for '{output_label_for_average}'.")
                        else:
                            log_callback(f"    Unknown averaging method: {averaging_method}. Skipping.")
                            overall_success = False
                    except Exception as e:
                        logger.exception("Error during averaging for participant %s", participant_pid)
                        log_callback(
                            f"    !!! Error during averaging for '{output_label_for_average}', PID {participant_pid}: {e}\n{traceback.format_exc()}")
                        overall_success = False
                else:
                    log_callback(
                        f"    No epochs collected for rule '{output_label_for_average}' for participant {participant_pid}. Cannot average.")

                del epochs_for_this_participant_this_rule
                gc.collect()

            if averaged_data_for_this_participant:
                if stop_event.is_set():
                    return False
                log_callback(
                    f"  --- Initiating Post-Processing for Participant {participant_pid} (Recipe: '{group_recipe_name}') ---")
                _debug(f"Post-processing data for participant {participant_pid}")

                post_proc_context = PostProcessContextForAdvanced(
                    log_callback=log_callback,
                    save_folder_path_str=output_dir_str,
                    pid_for_output=participant_pid,
                    averaged_epochs_dict=averaged_data_for_this_participant,
                    condition_name_for_output=group_recipe_name
                )
                try:
                    external_post_process_func(post_proc_context, list(averaged_data_for_this_participant.keys()))
                    log_callback(
                        f"  --- Post-processing for {participant_pid} (Recipe: '{group_recipe_name}') completed. ---")
                except Exception as e:
                    logger.exception("Error during post-processing for participant %s", participant_pid)
                    log_callback(
                        f"  !!! Error during post-processing for {participant_pid} (Recipe: '{group_recipe_name}'): {e}\n{traceback.format_exc()}")
                    overall_success = False

                current_operation_count += 1
                progress_callback(current_operation_count / total_operations_estimate)
            else:
                log_callback(
                    f"  No data averaged for participant {participant_pid} (Recipe: '{group_recipe_name}'). Skipping post-processing.")

            del averaged_data_for_this_participant
            gc.collect()
            log_callback(
                f"  -- Finished processing for Participant: {participant_pid} (Recipe: '{group_recipe_name}') --")
            _debug(f"Completed participant {participant_pid}")

        log_callback(f"--- All participants processed for Group Recipe: {group_recipe_name} ---")

    log_callback("Advanced averaging core processing finished.")
    _debug("Core processing complete")
    return overall_success
