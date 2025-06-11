"""Wrapper functions that utilize the fast C++ extension when available."""

import numpy as np

from fast_cpp import EXTENSION_AVAILABLE, downsample, apply_fir_filter


def downsample_data(data: np.ndarray, orig_rate: float, target_rate: float) -> np.ndarray:
    """Downsample EEG data using the C++ extension."""
    if not EXTENSION_AVAILABLE:
        raise RuntimeError("Fast processing extension not available")
    factor = int(round(orig_rate / target_rate))
    return downsample(data, factor)


def fir_filter_data(data: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Apply FIR filter coefficients using the C++ extension."""
    if not EXTENSION_AVAILABLE:
        raise RuntimeError("Fast processing extension not available")
    return apply_fir_filter(data, coeffs)
