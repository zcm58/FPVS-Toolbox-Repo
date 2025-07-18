# main.py

#!/usr/bin/env python3

"""Entry point for launching the FPVS Toolbox GUI application."""

try:
    from ctypes import windll  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - non-Windows
    windll = None

try:  # pragma: no cover - best effort on Windows
    if windll is not None:
        windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

from fpvs_app import FPVSApp
from Main_App.debug_utils import configure_logging, get_settings
import multiprocessing
import sys

def main() -> None:
    """Entry point for running the FPVS Toolbox or its sub-tools."""
    multiprocessing.freeze_support()

    if "--run-image-resizer" in sys.argv:
        from Tools.Image_Resizer import pyside_resizer

        pyside_resizer.main()
        return

    if "--run-plot-generator" in sys.argv:
        from Tools.Plot_Generator import plot_generator

        plot_generator.main()
        return

    settings = get_settings()
    configure_logging(settings.debug_enabled())
    app = FPVSApp()
    app.mainloop()


if __name__ == "__main__":
    main()
