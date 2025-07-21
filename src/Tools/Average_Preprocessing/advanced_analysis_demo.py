"""Minimal demonstration of the advanced analysis logic without a GUI."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .advanced_analysis import AdvancedAnalysis

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    adv = AdvancedAnalysis()

    # Example setup
    example_files = [str(Path(os.getcwd()) / "file1.bdf"), str(Path(os.getcwd()) / "file2.bdf")]
    adv.add_source_files(example_files)
    adv.create_new_group("Example", [1, 2])
    logger.info("Defined groups: %s", adv.defined_groups)

