# FPVS Toolbox

The FPVS Toolbox is a GUI based application for preprocessing, cleaning, and analyzing EEG data collected from Fast Periodic Visual Stimulation (FPVS) paradigm experiments on BioSemi systems. The purpose of this project is to standardize a processing method for FPVS data and to reduce the amount of time that researchers spend analyzing data. 

## Features

- Batch processing of BioSemi `.BDF` data files
- Automated preprocessing pipeline: referencing, filtering, resampling, kurtosis-based channel rejection and interpolation
- Extraction of epochs and post-processing metrics (FFT, SNR, BCA, Z-score)
- Built in Statistical analysis tool with repeated-measures ANOVA, linear mixed-effects models, and post-hoc pairwise tests to check for significant FPVS oddball responses in the Frontal, Central, Parietal, and Occipital lobes
- Image Resizer tool for quickly resizing images for PsychoPy experiments
- Averaging utility for combining epochs across files prior to post‑processing (useful if one needs to combine two similar FPVS experiments prior to calculating BCA)
- Optional saving of preprocessed data as `.fif` files for advanced analyses
- Interactive eLORETA/sLORETA source localization with 3‑D glass brain viewer
  (automatically downloads the `fsaverage` template if no MRI is specified) and

  a viewer tool to open saved results with adjustable transparency. The 3‑D
  viewer uses the `PyVistaQt` backend which provides interactive window support.



## Features currently under development:

- Publication quality 2D heatmaps
- Publication quality BCA frequency plots


## Installation

Prebuilt installers are provided on the GitHub Releases page. Each release contains a file named `FPVS_Toolbox_<version>-Setup.exe` created with Inno Setup.

1. Download the installer from the Releases page.
2. Double-click the `.exe` file. Because the installer is unsigned, Windows SmartScreen may warn that the app is unrecognized.
3. Choose **More info** and then **Run anyway** to continue.
4. Follow the prompts to select an install location. The installer places `FPVS_Toolbox.exe` and required libraries in the target folder.

Always verify that you obtained the installer directly from this repository's Releases page before bypassing SmartScreen.

### Running from Source

The application can also be launched from source on Windows. Clone the repository and ensure you have Python 3.9+ with the following packages installed:

- `mne`
- `numpy`
- `pandas`
- `scipy`
- `customtkinter`
- `statsmodels` (for statistical analyses)
- `nibabel` (required for source localization)
- `pyvista` and `pyvistaqt` (for the 3‑D viewer)

After installing the dependencies, start the GUI with:

```bash
python src/main.py
```

### Saving Preprocessed FIF Files


The app saves preprocessed data as FIF files by default (you can uncheck
the **Save Preprocessed .fif** option if not needed). A `.fif files`
subfolder is created in your output directory, and one `*-epo.fif` file is
written for each condition of every input BDF (for example,
`P1_Return_Fruit_vs_Veg-epo.fif`). These files can be selected when running the
source localization tool.


### Source Localization

Choose **Source Localization (eLORETA/sLORETA)** from the Tools menu to run an
inverse solution on a preprocessed `.fif` file. Select the desired method and an
output folder. An interactive 3‑D viewer will open showing both hemispheres with
anatomical labels. You can open any saved `source-lh.stc`/`source-rh.stc`
pair later using the **View STC** button to inspect the results interactively and
adjust the brain transparency. A hemisphere selector lets you view the left
hemisphere, right hemisphere, both together, or a split layout. Activity is
displayed with a red heatmap so you can easily spot the peak response. The
color bar at the bottom of the viewer shows the magnitude of the estimated
current density and is labeled **Source amplitude** for clarity.

Saved source estimates can also be opened directly from the command line using
`pyqt_viewer.py`:

```bash
python src/Tools/SourceLocalization/pyqt_viewer.py --stc path/to/source
```

The viewer provides sliders for adjusting cortex opacity and selecting the time
point displayed. A new threshold slider lets you interactively hide low-level
activity so only the strongest sources remain visible.

The underlying module exposes an `is_pyvistaqt_backend()` helper which
returns ``True`` when the viewer will use the interactive PyVistaQt
backend. This can be queried before opening any 3‑D windows to ensure
that PyVistaQt is active. The selected backend is also printed to the
on‑screen log whenever a 3‑D viewer is opened so you can verify it
directly from the GUI.

Additional parameters for band‑pass filtering and oddball cycle localisation can be
configured under the **LORETA** tab in the Settings window. Define the low and
high filter bounds and list the oddball harmonics to sum in the frequency domain
(for example ``1.2, 2.4, 3.6``). The summed harmonic amplitude is used when
applying sLORETA, and the SNR value controls regularisation. These values are
populated in the Source Localization dialog so they can be tweaked per analysis.




## License

This repository is released under a proprietary license. Viewing the source is permitted for personal or educational purposes only. See [LICENSE](LICENSE) for the full terms.

## Support

If you encounter issues or have feature requests, please open an issue on the GitHub project page. Contributions are welcome, but please respect the modular structure and guidelines noted in the `AGENTS.md` files when proposing changes.
