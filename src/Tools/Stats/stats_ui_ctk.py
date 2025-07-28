"""Wrapper for launching the legacy CustomTkinter stats tool."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def launch_ctk_stats_tool() -> None:
    """Launch the old CustomTkinter-based stats tool in a separate process."""
    script = Path(__file__).resolve().with_name("stats.py")
    subprocess.Popen([sys.executable, str(script)], close_fds=True)
