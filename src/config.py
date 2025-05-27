import numpy as np
import customtkinter as ctk

# — FPVS Toolbox version & update API —
FPVS_TOOLBOX_VERSION       = "0.9.0"
FPVS_TOOLBOX_UPDATE_API    = "https://api.github.com/repos/zcm58/FPVS-Toolbox-Repo/releases/latest"
FPVS_TOOLBOX_REPO_PAGE     = "https://github.com/zcm58/FPVS-Toolbox-Repo/releases"

# — metrics —
TARGET_FREQUENCIES = np.arange(1.2, 16.8 + 1.2, 1.2)
DEFAULT_ELECTRODE_NAMES_64 = [
    'Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5','T7','TP7',
    'CP5','CP3','CP1','P1','P3','P5','P7','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz',
    'Fpz','Fp2','AF8','AF4','AFz','Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz','Cz',
    'C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2'
]
DEFAULT_STIM_CHANNEL = 'Status'

# — GUI defaults —
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
CORNER_RADIUS         = 8
PAD_X                 = 5
PAD_Y                 = 5
ENTRY_WIDTH           = 100
LABEL_ID_ENTRY_WIDTH  = 120

# === Constants for Stats Toolbox ===
ROIS = {
    "Frontal Lobe": ["F3", "F4", "Fz"],
    "Occipital Lobe": ["O1", "O2", "Oz"],
    "Parietal Lobe": ["P3", "P4", "Pz"],
    "Central Lobe": ["C3", "C4", "Cz"]
    # Add more ROIs here if needed
}
ALL_ROIS_OPTION = "(All ROIs)"

HARMONIC_CHECK_ALPHA = 0.05 # Significance level for one-sample t-test in harmonic check

# UI Placeholder Strings for Stats Toolbox
STATS_PLACEHOLDER_SCAN_FOLDER = "(Scan Folder)"
STATS_PLACEHOLDER_NO_OTHER_CONDITIONS = "(No other conditions)"
STATS_PLACEHOLDER_SELECT_CONDITION_A = "(Select Condition A)"

# Filename for Quality Flags (used by fpvs_app.py to write and stats.py to read)
QUALITY_FLAGS_FILENAME = "Potential_Outlier_Participants.txt"

# Default for populating frequency checkboxes if needed (though currently removed from stats UI)
# If any other part of stats.py might still conceptually need a default list of target frequencies
# (even if not for UI checkboxes anymore), it could live here.
# For now, assuming it's not directly needed by stats.py if checkboxes are gone.
# DEFAULT_STATS_TARGET_FREQUENCIES = np.round(np.arange(1.2, 17.0, 1.2), 1)