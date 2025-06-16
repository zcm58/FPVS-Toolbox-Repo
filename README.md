# FPVS Toolbox

The FPVS Toolbox is a GUI based application for preprocessing, cleaning, and analyzing EEG data collected from Fast Periodic Visual Stimulation (FPVS) paradigm experiments on BioSemi systems. The purpose of this project is to standardize a processing method for FPVS data and to reduce the amount of time that researchers spend analyzing data. 

## Features

- Batch processing of BioSemi `.BDF` data files
- Automated preprocessing pipeline: referencing, filtering, resampling, kurtosis-based channel rejection and interpolation
- Extraction of epochs and post-processing metrics (FFT, SNR, BCA, Z-score)
- Built in Statistical analysis tool with repeated-measures ANOVA, linear mixed-effects models, and post-hoc pairwise tests to check for significant FPVS oddball responses in the Frontal, Central, Parietal, and Occipital lobes
- Image Resizer tool for quickly resizing images for PsychoPy experiments
- Averaging utility for combining epochs across files prior to post‑processing (useful if one needs to combine two similar FPVS experiments prior to calculating BCA)
- Publication quality BCA frequency plots

## Features currently under development:

- Publication quality 2D heatmaps


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
- `matplotlib`

After installing the dependencies, start the GUI with:

```bash
python src/main.py
```

To generate a BCA frequency plot from an Excel results file:

```bash
python -m Tools.bca_plotter results.xlsx -o bca_plot.png
```

The BCA Plotter is also available from the **Tools** menu inside the GUI.



## License

This repository is released under a proprietary license. Viewing the source is permitted for personal or educational purposes only. See [LICENSE](LICENSE) for the full terms.

## Support

If you encounter issues or have feature requests, please open an issue on the GitHub project page. Contributions are welcome, but please respect the modular structure and guidelines noted in the `AGENTS.md` files when proposing changes.
