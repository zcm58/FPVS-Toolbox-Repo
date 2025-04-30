# post_process.py

import os
import pandas as pd
import numpy as np
import traceback
import gc
import mne
from config import TARGET_FREQUENCIES, DEFAULT_ELECTRODE_NAMES_64
# Removed unused imports if any

def post_process(app, condition_labels_present):
    """
    Calculates metrics (FFT, SNR, Z-score, BCA) using time-domain epoch averaging
    after scaling data to microvolts, and saves results to Excel.
    """
    app.log("--- Post-processing: Calculating Metrics & Saving Excel ---")
    parent_folder = app.save_folder_path.get()
    if not parent_folder or not os.path.isdir(parent_folder):
        app.log(f"Error: Invalid save folder: '{parent_folder}'")
        # Avoid showing messagebox from background thread if possible, rely on log
        # from tkinter import messagebox
        # messagebox.showerror("Save Error", f"Invalid output folder:\n{parent_folder}")
        return

    # Extract participant ID from first filename (ensure paths exist)
    pid = "UnknownPID" # Default PID
    if app.data_paths:
        try:
            first_file = os.path.basename(app.data_paths[0])
            pid_base = os.path.splitext(first_file)[0]
            # Adapt this logic if your PID extraction needs differ
            # Example: find first occurrence of P followed by digits
            import re
            match = re.search(r'(P\d+)', pid_base, re.IGNORECASE)
            if match:
                pid = match.group(1).upper() # Use uppercase P for consistency maybe
            else:
                # Fallback if pattern not found (e.g., use base name)
                pid = pid_base
        except Exception as e:
             app.log(f"Warn: Could not extract PID from {app.data_paths[0]}: {e}")

    any_results_saved = False # Track if any file was successfully saved

    # Get the actual dictionary of epochs for the current file from app state
    # This assumes _finalize_processing has narrowed down app.preprocessed_data correctly
    # If not, adjust data access logic here.
    current_file_epochs = app.preprocessed_data # Passed from _finalize_processing

    # Iterate through the conditions *actually present* for this file
    for cond_label in condition_labels_present:
        # Retrieve the list of Epochs objects for this condition (should usually be 1 after finalize)
        epochs_list = current_file_epochs.get(cond_label, [])

        if not epochs_list:
            app.log(f"\nSkipping post-processing for '{cond_label}': No epoch data found for this file.")
            continue

        # Although finalize narrows it down, the original structure expects a list.
        # We'll average across this list if it contains multiple Epochs objects.
        app.log(f"\nPost-processing '{cond_label}' ({len(epochs_list)} Epochs object(s) from this file)...")

        accum = {'fft': None, 'snr': None, 'z': None, 'bca': None}
        valid_count = 0
        n_ch_final = 0 # Store number of channels
        final_electrode_names = [] # Store electrode names

        for file_idx, epochs in enumerate(epochs_list):
            # Ensure 'epochs' is a valid MNE Epochs object
            if not hasattr(epochs, 'info') or not hasattr(epochs, 'get_data'):
                 app.log(f"    Item {file_idx+1} is not a valid Epochs object. Skipping.")
                 continue

            app.log(f"  Processing Epochs object {file_idx + 1}/{len(epochs_list)} for '{cond_label}'...")
            gc.collect()

            try:
                # Ensure data is loaded
                if not epochs.preload:
                    try:
                        epochs.load_data()
                    except Exception as load_err:
                        app.log(f"    Error loading data for Epochs object {file_idx+1}: {load_err}. Skipping.")
                        continue

                # Skip if empty after loading
                if len(epochs.events) == 0:
                    app.log("    Epochs object contains 0 events after loading. Skipping.")
                    continue

                # Pick good EEG channels
                # Use inst.pick('eeg', exclude='bads') for modern MNE
                epochs_eeg = epochs.copy().pick('eeg', exclude='bads')
                picks = mne.pick_types(epochs_eeg.info, eeg=True) # Get indices relative to the picked object

                if picks.size == 0:
                    app.log("    No good EEG channels found in this Epochs object. Skipping.")
                    continue

                # Time‑domain averaging
                # Use the picked object to get data only for good EEG channels
                ep_data = epochs_eeg.get_data() # Should be (n_epochs, n_channels, n_times)
                avg_data = np.mean(ep_data.astype(np.float64), axis=0) # Average over epochs -> (n_channels, n_times)
                n_ch, n_t = avg_data.shape

                # Determine electrode names from the picked object
                ch_names = epochs_eeg.info['ch_names']
                current_electrode_names = []
                if n_ch == len(DEFAULT_ELECTRODE_NAMES_64) and set(ch_names) == set(DEFAULT_ELECTRODE_NAMES_64):
                    # If names match exactly (ignoring order), use standard order
                    current_electrode_names = [name for name in DEFAULT_ELECTRODE_NAMES_64 if name in ch_names]
                    # Reorder avg_data to match DEFAULT_ELECTRODE_NAMES_64 if necessary
                    name_to_idx = {name: i for i, name in enumerate(ch_names)}
                    indices = [name_to_idx[name] for name in current_electrode_names]
                    avg_data = avg_data[indices, :]
                elif n_ch == len(DEFAULT_ELECTRODE_NAMES_64):
                     app.log(f"    Warn: Found {n_ch} channels, but names don't match default list. Using default names/order.")
                     current_electrode_names = DEFAULT_ELECTRODE_NAMES_64 # Assume order is okay
                else:
                    app.log(f"    Warn: Found {n_ch} channels, not 64. Using actual channel names.")
                    current_electrode_names = ch_names # Use the names directly

                # Store names/channel count from first valid object
                if valid_count == 0:
                    n_ch_final = n_ch
                    final_electrode_names = current_electrode_names

                # Check consistency
                if n_ch != n_ch_final or current_electrode_names != final_electrode_names:
                     app.log(f"    Error: Channel mismatch between Epochs objects for condition '{cond_label}'. Skipping this object.")
                     app.log(f"      Expected {n_ch_final} channels with names {final_electrode_names}")
                     app.log(f"      Got {n_ch} channels with names {current_electrode_names}")
                     continue # Skip this Epochs object

                # <<< APPLY SCALING V -> µV >>>
                # Scale averaged data from Volts (MNE default) to Microvolts
                avg_data_uv = avg_data * 1e6
                # Only log scaling once per condition if averaging multiple files
                if file_idx == 0:
                     app.log(f"    Scaling avg_data to microvolts (multiplied by 1e6)")
                # <<< END SCALING >>>

                # FFT setup - USE THE SCALED DATA (avg_data_uv)
                sfreq = epochs_eeg.info['sfreq']
                num_bins = n_t // 2 + 1
                freqs = np.linspace(0, sfreq / 2.0, num=num_bins, endpoint=True)
                # Perform FFT on the microvolt-scaled data
                fft_full = np.fft.fft(avg_data_uv, axis=1)
                # Normalization remains the same, but fft_vals are now in µV scale
                fft_vals = np.abs(fft_full[:, :num_bins]) / n_t * 2

                # Allocate metric arrays for this Epochs object
                n_tf = len(TARGET_FREQUENCIES)
                f_fft = np.zeros((n_ch_final, n_tf))
                f_snr = np.zeros((n_ch_final, n_tf))
                f_z = np.zeros((n_ch_final, n_tf))
                f_bca = np.zeros((n_ch_final, n_tf))

                # Compute FFT, SNR, Z, BCA
                for c_idx in range(n_ch_final):
                    for f_idx, t_freq in enumerate(TARGET_FREQUENCIES):
                        if not (freqs[0] <= t_freq <= freqs[-1]):
                            continue # Skip if target frequency is outside FFT range

                        t_bin = np.argmin(np.abs(freqs - t_freq))

                        # Define noise bins relative to t_bin
                        # Exclude t_bin-2, t_bin-1, t_bin (matching interpretation of MATLAB code)
                        low_noise_idx = t_bin - 12
                        high_noise_idx = t_bin + 12 # Index up to (but not including) high_noise_idx+1
                        exclude_bins = {t_bin - 2, t_bin - 1, t_bin}

                        noise_idx = [i for i in range(low_noise_idx, high_noise_idx + 1) # Include high index
                                     if 0 <= i < len(freqs) and i not in exclude_bins]

                        # Calculate noise stats
                        noise_mean = 0.0
                        noise_std = 0.0
                        if len(noise_idx) >= 4: # Need sufficient bins for stable estimate
                            noise_vals = fft_vals[c_idx, noise_idx]
                            noise_mean = np.mean(noise_vals)
                            noise_std = np.std(noise_vals)
                        else:
                             # Log if not enough noise bins found near a target frequency
                             if valid_count == 0 and c_idx == 0: # Log only once per run
                                 app.log(f"    Warn: Not enough noise bins found near {t_freq:.1f} Hz (Bin {t_bin}). Check epoch length/FFT resolution.")

                        # Calculate metrics
                        amp_val = fft_vals[c_idx, t_bin]
                        # Use max amplitude in t_bin +/- 1 bin for Z-score calculation
                        peak_bins = slice(max(0, t_bin - 1), min(len(freqs), t_bin + 2)) # Slice up to t_bin+1
                        peak_val = np.max(fft_vals[c_idx, peak_bins])

                        # Avoid division by zero or near-zero
                        snr_val = amp_val / noise_mean if noise_mean > 1e-12 else 0
                        z_val = (peak_val - noise_mean) / noise_std if noise_std > 1e-12 else 0
                        bca_val = amp_val - noise_mean # Baseline-corrected amplitude

                        # Store results
                        f_fft[c_idx, f_idx] = amp_val
                        f_snr[c_idx, f_idx] = snr_val
                        f_z[c_idx, f_idx] = z_val
                        f_bca[c_idx, f_idx] = bca_val

                # Accumulate results across Epochs objects (if multiple for this condition)
                if accum['fft'] is None:
                    accum = {'fft': f_fft, 'snr': f_snr, 'z': f_z, 'bca': f_bca}
                else:
                    accum['fft'] += f_fft
                    accum['snr'] += f_snr
                    accum['z'] += f_z
                    accum['bca'] += f_bca
                valid_count += 1

            except Exception as e:
                app.log(f"!!! Error during post-processing Epochs object {file_idx+1} for '{cond_label}': {e}")
                app.log(traceback.format_exc()) # Print detailed error

            finally:
                del epochs_eeg # Clean up copied object
                gc.collect()

        # --- Averaging & Saving ---
        # Average metrics across all valid Epochs objects processed for this condition
        if valid_count > 0 and final_electrode_names:
            avg = {k: v / valid_count for k, v in accum.items()}
            cols = [f"{f:.1f}_Hz" for f in TARGET_FREQUENCIES]

            # Create DataFrames
            dfs = {
                'FFT Amplitude (uV)': pd.DataFrame(avg['fft'], index=final_electrode_names, columns=cols),
                'SNR': pd.DataFrame(avg['snr'], index=final_electrode_names, columns=cols),
                'Z Score': pd.DataFrame(avg['z'], index=final_electrode_names, columns=cols),
                'BCA (uV)': pd.DataFrame(avg['bca'], index=final_electrode_names, columns=cols)
            }
            # Add electrode column
            for df in dfs.values():
                df.insert(0, 'Electrode', df.index)

            # --- Excel Writing ---
            # Sanitize condition label for folder/file name
            # Replace slashes, remove leading/trailing whitespace
            safe_label = cond_label.replace('/', '-').replace('\\', '-').strip()
            # Remove potential leading numbers like "1 - " if desired, adapt regex if needed
            safe_label_base = re.sub(r'^\d+\s*-\s*', '', safe_label)

            # Create subfolder named after the sanitized condition label base name
            sub_folder_path = os.path.join(parent_folder, safe_label_base)
            try:
                 os.makedirs(sub_folder_path, exist_ok=True)
            except OSError as e:
                 app.log(f"Error creating subfolder {sub_folder_path}: {e}. Saving to parent folder.")
                 sub_folder_path = parent_folder # Fallback

            # Construct filename using PID and sanitized label base name
            excel_filename = f"{pid}_{safe_label_base}_Results.xlsx"
            excel_path = os.path.join(sub_folder_path, excel_filename)

            app.log(f"Writing Excel for '{cond_label}': {excel_path}")
            try:
                with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                    workbook = writer.book
                    # Define format for centered text
                    center_fmt = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

                    for sheet_name, df_out in dfs.items():
                        df_out.to_excel(writer, sheet_name=sheet_name, index=False)
                        worksheet = writer.sheets[sheet_name]

                        # Auto-adjust column widths with padding and center alignment
                        for col_idx, col_name in enumerate(df_out.columns):
                            # Calculate max length of header or data
                            header_len = len(str(col_name))
                            try:
                                # Ensure data is treated as string for length calculation
                                # Handle potential NaN/None values gracefully
                                max_data_len = df_out[col_name].astype(str).map(len).max()
                                if pd.isna(max_data_len): max_data_len = 0
                            except Exception:
                                max_data_len = 0 # Fallback

                            # Set width with padding (e.g., + 4)
                            width = max(header_len, int(max_data_len)) + 4
                            worksheet.set_column(col_idx, col_idx, width, center_fmt)

                app.log(f"Successfully saved Excel for '{cond_label}'.")
                any_results_saved = True
            except Exception as write_err:
                 app.log(f"!!! Error writing Excel file {excel_path}: {write_err}")
                 app.log(traceback.format_exc())

        else:
            app.log(f"No valid data processed for '{cond_label}'. No Excel file generated for this condition.")

    # Final message based on whether any results were saved
    if not any_results_saved:
         app.log("Warning: Processing completed, but no Excel files were saved for any condition.")

    # Optional: Clean up large variables if needed
    del current_file_epochs
    gc.collect()