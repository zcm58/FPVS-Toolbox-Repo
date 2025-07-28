"""Wrapper for launching the legacy CustomTkinter stats tool."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def launch_ctk_stats_tool() -> None:
    """Launch the old CustomTkinter-based stats tool in a separate process."""
    script = Path(__file__).resolve().with_name("stats.py")
    subprocess.Popen([sys.executable, str(script)], close_fds=True)


def create_widgets() -> None:
    """Entry point for launching the legacy CTk Stats Tool."""
    # Import locally to avoid circular imports when ``stats`` imports this module.
    from Tools.Stats.stats import StatsAnalysisWindow

    window = StatsAnalysisWindow(None)
    window.mainloop()


# Make sure it's importable when using ``from Tools.Stats.stats_ui_ctk import create_widgets``
__all__ = ["create_widgets"]
