# bca_plotter.py
"""Utilities for plotting BCA frequency spectra.

This module loads BCA (baseline-corrected amplitude) values from Excel
results files and generates publication quality frequency plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_bca_values(excel_path: str | Path, roi_channels: Optional[Sequence[str]] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Load BCA amplitude values from an Excel results file.

    Parameters
    ----------
    excel_path : str or Path
        Path to the Excel file containing a sheet named ``"BCA (uV)"``.
    roi_channels : sequence of str, optional
        Electrode names to average. If ``None`` all channels are included.

    Returns
    -------
    tuple of numpy.ndarray
        ``(frequencies, amplitudes)`` arrays averaged across the specified
        channels.
    """
    df = pd.read_excel(excel_path, sheet_name="BCA (uV)", index_col="Electrode")
    freq_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("_Hz")]
    if roi_channels is not None:
        df = df.reindex(roi_channels).dropna(how="all")
    mean_vals = df[freq_cols].mean(axis=0, skipna=True)
    freqs = np.array([float(c[:-3]) for c in freq_cols], dtype=float)
    amps = mean_vals.to_numpy(dtype=float)
    return freqs, amps


def generate_bca_plot(
    excel_file: str | Path,
    output_path: str | Path | None = None,
    roi_channels: Optional[Sequence[str]] = None,
    max_freq: float = 10.0,
    y_max: float = 0.3,
    dpi: int = 300,
) -> None:
    """Create a BCA frequency plot from an Excel results file.

    Parameters
    ----------
    excel_file : str or Path
        Path to the results workbook.
    output_path : str or Path, optional
        If provided, the plot is saved to this path instead of shown.
    roi_channels : sequence of str, optional
        Channels to average. ``None`` averages across all channels.
    max_freq : float, default 10.0
        Highest frequency to display on the x-axis.
    y_max : float, default 0.3
        Upper limit for the amplitude axis.
    dpi : int, default 300
        Resolution used when saving the figure.
    """
    freqs, amps = load_bca_values(excel_file, roi_channels)
    mask = freqs <= max_freq
    freqs, amps = freqs[mask], amps[mask]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    ax.stem(freqs, amps, basefmt=" ", use_line_collection=True)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("BCA (uV)")
    ax.set_ylim(0, y_max)
    ax.set_xlim(0, max_freq)
    ax.set_title("Baseline Corrected Amplitude")
    ax.grid(True, linestyle=":", linewidth=0.5)

    if output_path:
        fig.tight_layout()
        plt.savefig(output_path, dpi=dpi)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a BCA frequency plot from an Excel results file.")
    parser.add_argument("excel_file", help="Path to results workbook")
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Save plot to this path instead of displaying",
    )
    parser.add_argument(
        "--roi",
        metavar="CH1,CH2",
        help="Comma separated list of ROI channels to average",
    )
    parser.add_argument("--max-freq", type=float, default=10.0, help="Max frequency to display")
    parser.add_argument("--y-max", type=float, default=0.3, help="Max amplitude for y-axis")
    parser.add_argument("--dpi", type=int, default=300, help="Figure resolution when saving")

    args = parser.parse_args()

    roi_list: Optional[Sequence[str]] = args.roi.split(",") if args.roi else None

    generate_bca_plot(
        args.excel_file,
        output_path=args.output,
        roi_channels=roi_list,
        max_freq=args.max_freq,
        y_max=args.y_max,
        dpi=args.dpi,
    )
