# advanced_analysis_core.py
import os
import gc
import traceback
from typing import List, Dict, Any, Callable, Optional, Union  # Added Union here

import mne  # For MNE specific types and operations
import numpy as np
import threading  # Added for stop_event type hint


# Placeholder for the actual _external_post_process function if it's not directly importable
# from post_process import post_process as _external_post_process_actual

class PostProcessContextForAdvanced:
    """
    A context object to pass to the _external_post_process function,
    mimicking the necessary parts of the FPVSApp instance for averaged data.
    """

    def __init__(self,
                 log_callback: Callable[[str], None],
                 save_folder_path_str: str,
                 pid_for_group: str,  # PID derived for the group
                 averaged_epochs_dict: Dict[str, List[Union[mne.Epochs, mne.Evoked]]],  # The averaged data
                 group_name_for_output: str):  # Name of the group for output naming
        self.log_callback = log_callback
        self.save_folder_path_str = save_folder_path_str
        self.pid_for_group = pid_for_group
        self.averaged_epochs_dict = averaged_epochs_dict
        self.group_name_for_output = group_name_for_output

    def log(self, msg: str) -> None:
        # Routes log messages through the callback provided by AdvancedAnalysisWindow
        self.log_callback(f"[AdvCorePostProc] {msg}")

    @property
    def save_folder_path(self):  # Mimics tk.StringVar().get()
        class Getter:
            def __init__(self, val): self._val = val

            def get(self): return self._val

        return Getter(self.save_folder_path_str)

    @property
    def data_paths(self) -> List[str]:
        # For post_process, data_paths is often used for PID. Here, we provide a dummy
        # path incorporating the group name, or rely on pid_for_group directly.
        # This ensures post_process can form a unique base for filenames if it uses data_paths[0].
        return [f"{self.pid_for_group}_{self.group_name_for_output}_averaged.fif"]  # Dummy path

    @property
    def preprocessed_data(self) -> Dict[str, List[Union[mne.Epochs, mne.Evoked]]]:
        # This is the crucial part: it provides the averaged data to post_process
        return self.averaged_epochs_dict


def run_advanced_averaging_processing(
        defined_groups: List[Dict[str, Any]],
        main_app_params: Dict[str, Any],
        main_app_load_file_func: Callable[[str], Optional[mne.io.Raw]],
        main_app_preprocess_raw_func: Callable[[mne.io.Raw, Dict[str, Any]], Optional[mne.io.Raw]],
        external_post_process_func: Callable[[Any, List[str]], None],  # Type 'Any' for context
        output_dir_str: str,
        pid_extraction_func: Callable[[Dict[str, Any]], str],
        log_callback: Callable[[str], None],
        progress_callback: Callable[[float], None],
        stop_event: threading.Event  # Corrected type hint for threading.Event
) -> bool:
    """
    Main function to perform advanced averaging processing.
    Iterates through groups, processes files, averages epochs, and calls post-processing.
    """
    log_callback("Starting advanced averaging core processing...")
    total_operations_estimate = sum(
        1 + len(g.get("condition_mappings", [])) +  # Group setup + rules
        sum(len(m.get("sources", [])) for m in g.get("condition_mappings", [])) +  # Sources per rule
        len(g.get("condition_mappings", [])) +  # Averaging per rule
        1  # Post-processing per group
        for g in defined_groups
    )
    if total_operations_estimate == 0:  # Avoid division by zero if no groups/ops
        total_operations_estimate = 1

    current_operation = 0
    overall_success = True

    for group_data in defined_groups:
        if stop_event.is_set():
            log_callback("Processing cancelled by user (detected in core).")
            return False

        group_name = group_data['name']
        log_callback(f"--- Processing Group: {group_name} ---")
        current_operation += 1
        progress_callback(current_operation / total_operations_estimate)

        pid = pid_extraction_func(group_data)
        log_callback(f"  PID for group '{group_name}': {pid}")

        final_averaged_data_for_group: Dict[str, List[Union[mne.Epochs, mne.Evoked]]] = {}

        for mapping_rule in group_data.get('condition_mappings', []):
            if stop_event.is_set(): return False

            output_label = mapping_rule['output_label']
            log_callback(f"  Processing mapping rule for output: '{output_label}'")
            current_operation += 1
            progress_callback(current_operation / total_operations_estimate)

            epochs_from_all_sources_for_this_rule: List[mne.Epochs] = []

            for source_info in mapping_rule.get('sources', []):
                if stop_event.is_set(): return False

                file_path = source_info['file_path']
                original_label = source_info['original_label']
                original_id = source_info['original_id']
                file_basename = os.path.basename(file_path)

                log_callback(
                    f"    Loading and preprocessing: {file_basename} for ID {original_id} ('{original_label}')")

                raw = main_app_load_file_func(file_path)
                if raw is None or stop_event.is_set():
                    log_callback(f"    Skipping source {file_basename} (load failed or stop requested).")
                    overall_success = False;
                    continue

                raw_copy_for_preproc = raw.copy()  # Work on a copy
                del raw;
                gc.collect()  # Free original raw immediately

                raw_proc = main_app_preprocess_raw_func(raw_copy_for_preproc, **main_app_params.copy())
                del raw_copy_for_preproc;
                gc.collect()
                if raw_proc is None or stop_event.is_set():
                    log_callback(f"    Skipping source {file_basename} (preprocess failed or stop requested).")
                    overall_success = False;
                    continue

                try:
                    stim_channel = main_app_params.get('stim_channel', 'Status')
                    events = mne.find_events(raw_proc, stim_channel=stim_channel, consecutive=True, verbose=False)

                    if original_id not in events[:, 2]:
                        log_callback(
                            f"    Warning: Event ID {original_id} not found in {file_basename} for rule '{output_label}'. Skipping this source.")
                        del raw_proc;
                        gc.collect();
                        continue

                    event_id_dict_for_source = {original_label: original_id}

                    source_epochs = mne.Epochs(
                        raw_proc, events, event_id=event_id_dict_for_source,
                        tmin=main_app_params['epoch_start'], tmax=main_app_params['epoch_end'],
                        preload=True, baseline=None, verbose=False, on_missing='warn'
                    )
                    if len(source_epochs) > 0:
                        log_callback(
                            f"      Created {len(source_epochs)} epochs from {file_basename} for ID {original_id}.")
                        epochs_from_all_sources_for_this_rule.append(source_epochs)
                    else:
                        log_callback(
                            f"      No epochs created from {file_basename} for ID {original_id} (check event timing or data).")
                except Exception as e:
                    log_callback(
                        f"    !!! Error epoching {file_basename} for ID {original_id}: {e}\n{traceback.format_exc()}")
                    overall_success = False
                finally:
                    if 'raw_proc' in locals() and raw_proc is not None: del raw_proc; gc.collect()

                current_operation += 1
                progress_callback(current_operation / total_operations_estimate)

            if epochs_from_all_sources_for_this_rule:
                log_callback(
                    f"    Averaging {len(epochs_from_all_sources_for_this_rule)} sets of epochs for '{output_label}' using method: {group_data['averaging_method']}")
                try:
                    if group_data['averaging_method'] == "Pool Trials":
                        concatenated_epochs = mne.concatenate_epochs(epochs_from_all_sources_for_this_rule)
                        averaged_evoked = concatenated_epochs.average()
                        final_averaged_data_for_group[output_label] = [averaged_evoked]
                        log_callback(f"      Pooled {len(concatenated_epochs)} total trials for '{output_label}'.")

                    elif group_data['averaging_method'] == "Average of Averages":
                        evokeds_to_average = [ep.average() for ep in epochs_from_all_sources_for_this_rule if
                                              len(ep) > 0]
                        if evokeds_to_average:
                            averaged_evoked = mne.grand_average(evokeds_to_average, interpolate_bads=False)
                            final_averaged_data_for_group[output_label] = [averaged_evoked]
                            log_callback(
                                f"      Averaged {len(evokeds_to_average)} evoked responses for '{output_label}'.")
                        else:
                            log_callback(f"      No valid evoked responses to average for '{output_label}'.")
                    else:
                        log_callback(
                            f"    Unknown averaging method: {group_data['averaging_method']}. Skipping averaging for '{output_label}'.")
                        overall_success = False
                except Exception as e:
                    log_callback(f"    !!! Error during averaging for '{output_label}': {e}\n{traceback.format_exc()}")
                    overall_success = False
            else:
                log_callback(f"    No epochs collected for rule '{output_label}'. Cannot average.")

            del epochs_from_all_sources_for_this_rule;
            gc.collect()
            current_operation += 1
            progress_callback(current_operation / total_operations_estimate)

        if final_averaged_data_for_group:
            if stop_event.is_set(): return False
            log_callback(f"  --- Initiating Post-Processing for averaged data in group: {group_name} ---")

            post_proc_context = PostProcessContextForAdvanced(
                log_callback=log_callback,
                save_folder_path_str=output_dir_str,
                pid_for_group=pid,
                averaged_epochs_dict=final_averaged_data_for_group,
                group_name_for_output=group_name
            )
            try:
                external_post_process_func(post_proc_context, list(final_averaged_data_for_group.keys()))
                log_callback(f"  --- Post-processing for group {group_name} completed. ---")
            except Exception as e:
                log_callback(
                    f"  !!! Error during post-processing for group {group_name}: {e}\n{traceback.format_exc()}")
                overall_success = False
        else:
            log_callback(f"  No averaged data generated for group {group_name}. Skipping post-processing.")

        del final_averaged_data_for_group;
        gc.collect()
        current_operation += 1
        progress_callback(current_operation / total_operations_estimate)
        log_callback(f"--- Finished processing group: {group_name} ---")

    log_callback("Advanced averaging core processing finished.")
    return overall_success
