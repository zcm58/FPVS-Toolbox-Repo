"""Frequency-domain helper functions for oddball localization."""

from __future__ import annotations

import numpy as np
import mne


def evoked_from_harmonics(evoked: mne.Evoked, freqs: list[float], *, bandwidth: float = 0.1) -> mne.Evoked:
    """Return an Evoked with amplitudes summed across ``freqs``.

    Parameters
    ----------
    evoked : Evoked
        Time-domain evoked response representing a single oddball cycle.
    freqs : list of float
        Frequencies (in Hz) to include when computing the spectrum.
    bandwidth : float
        Frequency window (Hz) around each harmonic to average. Defaults to 0.1.
    """
    if len(freqs) == 0:
        raise ValueError("freqs must not be empty")
    sfreq = evoked.info["sfreq"]
    data = np.fft.rfft(evoked.data, axis=1)
    freq_bins = np.fft.rfftfreq(evoked.data.shape[1], 1.0 / sfreq)
    amps = np.zeros(evoked.data.shape[0])
    for f in freqs:
        mask = np.abs(freq_bins - f) <= bandwidth / 2
        if not np.any(mask):
            idx = np.argmin(np.abs(freq_bins - f))
            mask[idx] = True
        amps += np.abs(data[:, mask]).mean(axis=1)
    return mne.EvokedArray(amps[:, np.newaxis], evoked.info, tmin=0.0)
