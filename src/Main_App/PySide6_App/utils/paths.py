"""Path utilities for locating bundled resources in source and frozen builds."""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["bundle_path"]


def bundle_path(*parts: str) -> Path:
    """Return a resolved :class:`Path` inside the application bundle.

    Parameters
    ----------
    *parts:
        Relative path components inside the bundle. Each component is passed to
        :class:`~pathlib.Path` to build the final resource path.

    Returns
    -------
    Path
        The absolute path to the requested resource, valid for both source
        checkouts and PyInstaller frozen distributions.
    """

    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return (base / Path(*parts)).resolve()

