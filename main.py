# main.py

#!/usr/bin/env python3

from ctypes import windll
try:
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

from fpvs_app import FPVSApp

if __name__ == "__main__":
    app = FPVSApp()
    app.mainloop()
