"""Helper utilities for SNR calculations used by the plot generator."""

from __future__ import annotations

from math import isfinite
from typing import Sequence, List

__all__ = ["calc_snr_matlab"]


def _mean_safe(values: Sequence[float]) -> float:
    """Mean that tolerates empty or non-finite inputs."""
    acc = 0.0
    cnt = 0
    for v in values:
        if isfinite(v):
            acc += float(v)
            cnt += 1
    return acc / cnt if cnt else float("nan")


def calc_snr_matlab(amplitudes: Sequence[float]) -> List[float]:
    """Return MATLAB-style SNR values for each amplitude bin.

    For each bin i, take a ±12-bin window around i, remove i and its
    immediate neighbours (i±1), drop one max and one min if available,
    compute the mean of the remaining bins as noise, then SNR = A[i]/noise.
    Non-finite values are ignored. If noise ≤ 1e-12 or non-finite, SNR = 0.
    """
    # Normalize to finite floats, preserve length
    amps = [float(a) if isfinite(float(a)) else 0.0 for a in amplitudes]
    n = len(amps)
    snr: List[float] = [0.0] * n

    for idx in range(n):
        # Window bounds
        low = max(0, idx - 12)
        high = min(n - 1, idx + 12)

        # Copy neighbourhood
        neigh = [amps[i] for i in range(low, high + 1)]

        # Remove center and immediate neighbours (idx±1) if present
        for remove in sorted({idx - 1, idx, idx + 1}, reverse=True):
            if low <= remove <= high:
                neigh.pop(remove - low)

        # Drop one max and one min to reduce outliers if enough points remain
        finite_neigh = [v for v in neigh if isfinite(v)]
        if len(finite_neigh) > 2:
            # Remove by value once each; safe because we built finite_neigh fresh
            finite_neigh.remove(max(finite_neigh))
            finite_neigh.remove(min(finite_neigh))

        mean_noise = _mean_safe(finite_neigh)
        a = amps[idx]

        snr[idx] = (a / mean_noise) if isfinite(mean_noise) and mean_noise > 1e-12 else 0.0

    return snr
