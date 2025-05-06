# post_process.py

import os
import pandas as pd
import numpy as np
import traceback
import gc
import mne
import re  # Ensure re is imported for pid_base sanitization if used
from config import TARGET_FREQUENCIES, DEFAULT_ELECTRODE_NAMES_64
from typing import List, Dict, Any, Union  # For type hints


# Define a protocol for the app context to satisfy type checkers if desired,
# though structural typing (duck typing) is Python's norm.
# from typing import Protocol, List, Dict, Any, Union
# class AppContext(Protocol):
#     def log(self, msg: str) -> None: ...
#     @property
#     def save_folder_path(self) -> Any: ... # Should have a .get() method returning str
#     @property
#     def data_paths(self) -> List[str]: ...
#     @property
#     def preprocessed_data(self) -> Dict[str, List[Union[mne.Epochs, mne.Evoked]]]: ...
#     # For advanced context
#     pid_for_group: Optional[str] = None
#     group_name_for_output: Optional[str] = None


def post_process(app: Any, condition_labels_present: List[str]) -> None:
    """
    Calculates metrics (FFT, SNR, Z-score, BCA) and saves results to Excel.
    Adapted to handle both single-file processing (from FPVSApp) and
    averaged group processing (from AdvancedAnalysisWindow via PostProcessContextForAdvanced).
    """
    app.log("--- Post-processing: Calculating Metrics & Saving Excel ---")
    parent_folder = app.save_folder_path.get()  # Works for both tk.StringVar and simple getter
    if not parent_folder or not os.path.isdir(parent_folder):
        app.log(f"Error: Invalid save folder: '{parent_folder}'")
        return

    # --- PID Determination ---
    pid = "UnknownPID"
    if hasattr(app, 'pid_for_group') and app.pid_for_group:  # Check for advanced context
        pid = app.pid_for_group
        app.log(f"Using PID for averaged group: {pid}")
    elif app.data_paths:  # Fallback to original method
        try:
            # Ensure app.data_paths[0] is a path string, not just a filename for PID extraction
            first_file_path = app.data_paths[0]
            first_file_basename = os.path.basename(first_file_path)
            pid_base = os.path.splitext(first_file_basename)[0]

            # Regex for PID (case-insensitive, using common patterns)
            # This regex is similar to the one in advanced_analysis_core.py
            pid_regex = r'\b(P\d+|Sub\d+|S\d+)\b'
            match = re.search(pid_regex, pid_base, re.IGNORECASE)
            if match:
                pid = match.group(1).upper()
            else:
                # Fallback: clean common suffixes from the base name
                pid_cleaned = re.sub(r'(_unamb|_ambig|_mid|_run\d*|_sess\d*|_task\w*|_eeg|_fpvs|_raw|_preproc|_ica).*$',
                                     '', pid_base, flags=re.IGNORECASE)
                pid_cleaned = re.sub(r'[^a-zA-Z0-9]', '', pid_cleaned)  # Remove remaining non-alphanumeric
                pid = pid_cleaned if pid_cleaned else pid_base  # Use cleaned or original base
            app.log(f"Extracted PID for single file processing: {pid}")
        except Exception as e:
            app.log(
                f"Warn: Could not extract PID from app.data_paths[0] ('{app.data_paths[0] if app.data_paths else 'N/A'}'): {e}")
    else:
        app.log("Warning: Could not determine PID (no pid_for_group and no data_paths). Using default 'UnknownPID'.")

    any_results_saved = False
    current_epochs_data_source = app.preprocessed_data  # This will be the averaged data for advanced context

    # Iterate through the conditions/output labels to be processed
    for cond_label_or_group_name in condition_labels_present:
        # Retrieve the list of Epochs or Evoked objects for this condition/group
        # For averaged data, this list will contain one Evoked object.
        # For original processing, it contains one Epochs object.
        data_list = current_epochs_data_source.get(cond_label_or_group_name, [])

        if not data_list:
            app.log(f"\nSkipping post-processing for '{cond_label_or_group_name}': No data found.")
            continue

        app.log(f"\nPost-processing '{cond_label_or_group_name}' ({len(data_list)} data object(s))...")

        # Accumulators for metrics (if averaging multiple Evoked/Epochs, though typically 1 for this function)
        accum = {'fft': None, 'snr': None, 'z': None, 'bca': None}
        valid_data_count = 0
        final_num_channels = 0
        final_electrode_names_ordered = []

        for data_idx, data_object in enumerate(data_list):  # data_object can be mne.Epochs or mne.Evoked
            is_evoked = isinstance(data_object, mne.Evoked)

            if not (hasattr(data_object, 'info') and (is_evoked or hasattr(data_object, 'get_data'))):
                app.log(f"    Item {data_idx + 1} is not a valid MNE data object. Skipping.")
                continue

            app.log(f"  Processing data object {data_idx + 1}/{len(data_list)} for '{cond_label_or_group_name}'...")
            gc.collect()

            try:
                # For Epochs, ensure data is loaded and there are events
                if not is_evoked:  # It's an Epochs object
                    if not data_object.preload:
                        try:
                            data_object.load_data()
                        except Exception as load_err:
                            app.log(f"    Error loading data for Epochs object {data_idx + 1}: {load_err}. Skipping.");
                            continue
                    if len(data_object.events) == 0:
                        app.log("    Epochs object contains 0 events after loading. Skipping.");
                        continue

                # Pick good EEG channels (for Evoked, it's already averaged, so pick from its channels)
                # For Epochs, pick from the Epochs object directly.
                if is_evoked:
                    # Evoked objects don't have 'bads' in info in the same way Epochs do for picking.
                    # We assume bads were handled before averaging to Evoked.
                    # We pick all 'eeg' channels available in the Evoked object.
                    data_eeg = data_object.copy().pick('eeg')
                else:  # Epochs object
                    data_eeg = data_object.copy().pick('eeg', exclude='bads')

                if not data_eeg.ch_names:  # Check if any EEG channels are left
                    app.log("    No good EEG channels found in this data object. Skipping.")
                    continue

                # Get averaged data:
                # If Evoked, data_eeg.data is already (n_channels, n_times)
                # If Epochs, average it now.
                if is_evoked:
                    avg_data = data_eeg.data  # (n_channels, n_times)
                else:  # Epochs
                    ep_data = data_eeg.get_data()  # (n_epochs, n_channels, n_times)
                    avg_data = np.mean(ep_data.astype(np.float64), axis=0)  # (n_channels, n_times)

                num_channels, num_times = avg_data.shape

                # Determine electrode names from the picked object (data_eeg)
                current_ch_names_from_obj = data_eeg.info['ch_names']
                ordered_electrode_names_for_df = []

                # Standardize channel order if possible (using DEFAULT_ELECTRODE_NAMES_64)
                if num_channels == len(DEFAULT_ELECTRODE_NAMES_64) and \
                        set(current_ch_names_from_obj) == set(DEFAULT_ELECTRODE_NAMES_64):
                    ordered_electrode_names_for_df = [name for name in DEFAULT_ELECTRODE_NAMES_64 if
                                                      name in current_ch_names_from_obj]
                    name_to_idx_map = {name: i for i, name in enumerate(current_ch_names_from_obj)}
                    reorder_indices = [name_to_idx_map[name] for name in ordered_electrode_names_for_df]
                    avg_data = avg_data[reorder_indices, :]
                    app.log(f"    Standardized channel order to {len(ordered_electrode_names_for_df)} channels.")
                elif num_channels == len(DEFAULT_ELECTRODE_NAMES_64):  # Names don't match but count does
                    app.log(
                        f"    Warn: Found {num_channels} channels, but names don't perfectly match default list. Using default names/order for output. Ensure consistency.")
                    ordered_electrode_names_for_df = DEFAULT_ELECTRODE_NAMES_64
                else:  # Different number of channels
                    app.log(
                        f"    Warn: Found {num_channels} channels (not {len(DEFAULT_ELECTRODE_NAMES_64)}). Using actual channel names and order from data.")
                    ordered_electrode_names_for_df = current_ch_names_from_obj

                # Store channel info from the first valid data object processed
                if valid_data_count == 0:
                    final_num_channels = num_channels
                    final_electrode_names_ordered = ordered_electrode_names_for_df

                # Consistency check if multiple data_objects are being processed (e.g. if averaging Evokeds later)
                if num_channels != final_num_channels or ordered_electrode_names_for_df != final_electrode_names_ordered:
                    app.log(
                        f"    Error: Channel mismatch between data objects for '{cond_label_or_group_name}'. Skipping this object.")
                    continue

                # Scale averaged data from Volts (MNE default) to Microvolts
                avg_data_uv = avg_data * 1e6
                if data_idx == 0:  # Log only once per condition/group
                    app.log(
                        f"    Scaling averaged data to microvolts (multiplied by 1e6). Max value: {np.max(np.abs(avg_data_uv)):.2f} uV")

                # FFT setup
                sfreq = data_eeg.info['sfreq']
                num_fft_bins = num_times // 2 + 1
                fft_frequencies = np.linspace(0, sfreq / 2.0, num=num_fft_bins, endpoint=True)
                fft_full_spectrum = np.fft.fft(avg_data_uv, axis=1)  # FFT on microvolt-scaled data
                fft_amplitudes = np.abs(fft_full_spectrum[:, :num_fft_bins]) / num_times * 2  # Normalize

                # Allocate metric arrays
                num_target_freqs = len(TARGET_FREQUENCIES)
                metrics_fft = np.zeros((final_num_channels, num_target_freqs))
                metrics_snr = np.zeros((final_num_channels, num_target_freqs))
                metrics_z = np.zeros((final_num_channels, num_target_freqs))
                metrics_bca = np.zeros((final_num_channels, num_target_freqs))

                # Compute metrics for each channel and target frequency
                for chan_idx in range(final_num_channels):
                    for freq_idx, target_freq in enumerate(TARGET_FREQUENCIES):
                        if not (fft_frequencies[0] <= target_freq <= fft_frequencies[-1]):
                            if chan_idx == 0 and data_idx == 0: app.log(
                                f"    Skipping target frequency {target_freq} Hz (out of FFT range).");
                            continue

                        target_bin_index = np.argmin(np.abs(fft_frequencies - target_freq))

                        # Define noise bins (e.g., +/- 12 bins, excluding target and immediate neighbors)
                        noise_bin_low = target_bin_index - 12
                        noise_bin_high = target_bin_index + 12
                        bins_to_exclude_from_noise = {target_bin_index - 2, target_bin_index - 1, target_bin_index,
                                                      target_bin_index + 1, target_bin_index + 2}  # Wider exclusion

                        valid_noise_indices = [
                            i for i in range(noise_bin_low, noise_bin_high + 1)
                            if 0 <= i < len(fft_frequencies) and i not in bins_to_exclude_from_noise
                        ]

                        noise_mean_val, noise_std_val = 0.0, 0.0
                        if len(valid_noise_indices) >= 4:  # Need enough bins for stable noise estimate
                            noise_amplitudes = fft_amplitudes[chan_idx, valid_noise_indices]
                            noise_mean_val = np.mean(noise_amplitudes)
                            noise_std_val = np.std(noise_amplitudes)
                        elif data_idx == 0 and chan_idx == 0:  # Log warning only once
                            app.log(
                                f"    Warn: Not enough noise bins near {target_freq:.1f} Hz (Bin {target_bin_index}). Check epoch length/FFT resolution.")

                        signal_amplitude = fft_amplitudes[chan_idx, target_bin_index]
                        # Peak value for Z-score (max in target_bin +/- 1 bin)
                        peak_val_slice = slice(max(0, target_bin_index - 1),
                                               min(len(fft_frequencies), target_bin_index + 2))
                        peak_signal_amplitude = np.max(fft_amplitudes[chan_idx, peak_val_slice])

                        snr_val = signal_amplitude / noise_mean_val if noise_mean_val > 1e-12 else 0
                        z_score_val = (
                                                  peak_signal_amplitude - noise_mean_val) / noise_std_val if noise_std_val > 1e-12 else 0
                        bca_val = signal_amplitude - noise_mean_val

                        metrics_fft[chan_idx, freq_idx] = signal_amplitude
                        metrics_snr[chan_idx, freq_idx] = snr_val
                        metrics_z[chan_idx, freq_idx] = z_score_val
                        metrics_bca[chan_idx, freq_idx] = bca_val

                # Accumulate (though typically only one data_object per cond_label_or_group_name here)
                if accum['fft'] is None:
                    accum = {'fft': metrics_fft, 'snr': metrics_snr, 'z': metrics_z, 'bca': metrics_bca}
                else:  # This part might not be strictly necessary if always one Evoked/Epochs per call
                    accum['fft'] += metrics_fft;
                    accum['snr'] += metrics_snr;
                    accum['z'] += metrics_z;
                    accum['bca'] += metrics_bca;
                valid_data_count += 1

            except Exception as e:
                app.log(
                    f"!!! Error during post-processing data object {data_idx + 1} for '{cond_label_or_group_name}': {e}")
                app.log(traceback.format_exc())
            finally:
                del data_eeg;
                gc.collect()  # Clean up copied object

        # --- Averaging metrics (if multiple valid_data_count, though usually 1) & Saving ---
        if valid_data_count > 0 and final_electrode_names_ordered:
            avg_metrics = {k: v / valid_data_count for k, v in accum.items()}
            freq_column_names = [f"{f:.1f}_Hz" for f in TARGET_FREQUENCIES]

            dataframes_to_save = {
                'FFT Amplitude (uV)': pd.DataFrame(avg_metrics['fft'], index=final_electrode_names_ordered,
                                                   columns=freq_column_names),
                'SNR': pd.DataFrame(avg_metrics['snr'], index=final_electrode_names_ordered, columns=freq_column_names),
                'Z Score': pd.DataFrame(avg_metrics['z'], index=final_electrode_names_ordered,
                                        columns=freq_column_names),
                'BCA (uV)': pd.DataFrame(avg_metrics['bca'], index=final_electrode_names_ordered,
                                         columns=freq_column_names)
            }
            for df_name in dataframes_to_save:  # Add 'Electrode' column
                dataframes_to_save[df_name].insert(0, 'Electrode', dataframes_to_save[df_name].index)

            # --- Output Naming Logic ---
            output_base_name = ""
            is_averaged_group_output = hasattr(app,
                                               'group_name_for_output') and app.group_name_for_output == cond_label_or_group_name

            if is_averaged_group_output:
                # For averaged groups, use the group name for subfolder and filename
                # cond_label_or_group_name IS the group name here.
                output_base_name = app.group_name_for_output
                excel_suffix = "_AvgResults.xlsx"  # Indicate it's an averaged result
            else:
                # For original single-file processing, use the condition label
                output_base_name = cond_label_or_group_name
                excel_suffix = "_Results.xlsx"

            # Sanitize base name for folder/file
            # Replace slashes, remove leading/trailing whitespace, remove leading numbers like "1 - "
            safe_output_base = re.sub(r'^\d+\s*-\s*', '', output_base_name.replace('/', '-').replace('\\', '-').strip())

            # Create subfolder
            output_subfolder_path = os.path.join(parent_folder, safe_output_base)
            try:
                os.makedirs(output_subfolder_path, exist_ok=True)
            except OSError as e:
                app.log(
                    f"Error creating subfolder {output_subfolder_path}: {e}. Saving to parent folder: {parent_folder}")
                output_subfolder_path = parent_folder  # Fallback

            excel_filename = f"{pid}_{safe_output_base}{excel_suffix}"
            full_excel_path = os.path.join(output_subfolder_path, excel_filename)

            app.log(f"Writing Excel for '{cond_label_or_group_name}': {full_excel_path}")
            try:
                with pd.ExcelWriter(full_excel_path, engine='xlsxwriter') as writer:
                    workbook = writer.book
                    center_fmt = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

                    for sheet_name, df_to_write in dataframes_to_save.items():
                        df_to_write.to_excel(writer, sheet_name=sheet_name, index=False)
                        worksheet = writer.sheets[sheet_name]
                        # Auto-adjust column widths and center text
                        for col_idx, header_name in enumerate(df_to_write.columns):
                            max_len = max(
                                len(str(header_name)),
                                df_to_write[header_name].astype(str).map(len).max() if not df_to_write[
                                    header_name].empty else 0
                            )
                            worksheet.set_column(col_idx, col_idx, max_len + 4, center_fmt)  # Add padding

                app.log(f"Successfully saved Excel: {excel_filename}")
                any_results_saved = True
            except Exception as write_err:
                app.log(f"!!! Error writing Excel file {full_excel_path}: {write_err}\n{traceback.format_exc()}")
        else:
            app.log(f"No valid data processed for '{cond_label_or_group_name}'. No Excel file generated.")

    if not any_results_saved:
        app.log("Warning: Processing completed, but no Excel files were saved for any condition/group.")

    # Clean up the main data source passed to the function
    del current_epochs_data_source;
    gc.collect()
    app.log("--- Post-processing finished. ---")

