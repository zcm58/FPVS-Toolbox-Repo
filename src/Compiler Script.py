pyinstaller `
  --clean `
  --noconfirm `
  --windowed `
  -n FPVS_Toolbox_0.9.0 `
  --paths=src `
  -i "C:\Users\zackm\OneDrive - Mississippi State University\Office Desktop\ToolBox Icon.ico" `
  --collect-all mne `
  --hidden-import=mne.io.bdf `
  --hidden-import=mne.io.eeglab `
  --hidden-import=scipy `
  --hidden-import=pandas `
  --hidden-import=numpy `
  -F src/main.py
