"""Processing helpers extracted from main_window.py."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _animate_progress_to(self, value: int) -> None:
    """Animate the progress bar smoothly to the target value."""
    self._progress_anim.stop()
    self._progress_anim.setStartValue(self.progress_bar.value())
    self._progress_anim.setEndValue(value)
    self._progress_anim.start()


# ``start_processing`` has been replaced by :func:`legacy_adapter.run_full_pipeline`.
# The previous implementation has been removed to avoid duplication with the
# new adapter-based workflow.
