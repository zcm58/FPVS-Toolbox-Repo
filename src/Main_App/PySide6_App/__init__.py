"""PySide6 application namespace.

Keep this init lightweight. Do not import Legacy here.
Expose only what PySide6 callers need.
"""

from __future__ import annotations

# Re-exports that are safe to import at boot
from .Backend import Project  # noqa: F401

__all__ = ["Project"]
