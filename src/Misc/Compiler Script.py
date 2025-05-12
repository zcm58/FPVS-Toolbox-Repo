cd "C:\Users\zcm58\PycharmProjects\FPVS Toolbox Repo"

pyinstaller `
  --clean `
  --noconfirm `
  --windowed `
  --name FPVS_Toolbox_0.8.2 `
  --icon "C:\Users\zcm58\OneDrive - Mississippi State University\Office Desktop\AppIcon.ico" `
  --collect-all mne `
  --hidden-import mne.io.bdf `
  --hidden-import mne.io.eeglab `
  --hidden-import scipy `
  --hidden-import pandas `
  --hidden-import numpy `
  --paths .\src `
  -F `
  .\src\main.py
