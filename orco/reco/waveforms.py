"""
Operations for processing collections of sensor waveforms.

Operations return new ``Waveforms`` instances without modifying the input.
Window bounds are half-open sample indices; time and amplitude units are carried
through unchanged rather than converted here.
"""

from typing import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike

from ..internal.types import IndexRange
from ..internal.types import Waveforms


def remove_baseline(wfms  : Waveforms,
                    method: Callable = np.median) -> Waveforms:
    """
    Subtract the baseline from each sensor waveform independently.

    ``method`` reduces a one-dimensional sensor waveform to one baseline value;
    the default is the median over all samples, including signal samples.
    """
    baselines  = np.apply_along_axis(method, 1, wfms.amplitudes)
    amplitudes = wfms.amplitudes - baselines[:, np.newaxis]
    return Waveforms(wfms.time, wfms.sensors, amplitudes)


def invert_polarity(wfms: Waveforms) -> Waveforms:
    """
    Return floating-point waveforms with their polarity inverted.
    """
    return Waveforms(wfms.time, wfms.sensors, -wfms.amplitudes)


def calibrate(wfms: Waveforms, gains: ArrayLike) -> Waveforms:
    """
    Divide each sensor waveform by its corresponding non-zero gain.

    ``gains`` must be one-dimensional and ordered to match ``wfms.sensors``.
    With gains in ADC * us / pe, ADC amplitudes become rates in pe/us. Integrate
    those rates with the sampling interval to obtain charge in pe.
    Wrong length/dimensionality, zero values and NaNs raise ``ValueError``.
    Infinite gains are allowed; finite waveform amplitudes then become zero.
    """
    gains = np.asarray(gains, dtype=float)

    if gains.ndim != 1 or wfms.nsensors() != gains.size:
        raise ValueError("A gain is required for each waveform sensor")

    invalid = np.isnan(gains) | (gains == 0)
    if np.any(invalid):
        indices = np.flatnonzero(invalid)
        raise ValueError(f"Invalid gain values at indices {indices}: {gains[indices]}")

    amplitudes = wfms.amplitudes / gains[:, np.newaxis]
    return Waveforms(wfms.time, wfms.sensors, amplitudes)


def find_windows(wfsum : ArrayLike,
                 thr   : float,
                 holdon: int) -> list[IndexRange]:
    """
    Return half-open ranges covering regions at or above ``thr``.

    Two above-threshold regions are joined when the below-threshold gap
    between them contains at most ``holdon`` samples.

    Returns ``IndexRange(first, last + 1)`` for each group, so a Python slice
    includes both crossing samples and any merged gap. Equality to ``thr``
    counts as a crossing. An empty waveform or no crossings returns an empty
    list. Negative hold-on or non-one-dimensional input raises ``ValueError``.
    """
    if holdon < 0:
        raise ValueError("holdon must be non-negative")

    wfsum = np.asarray(wfsum)
    if wfsum.ndim != 1:
        raise ValueError("Summed waveform must be one-dimensional")

    above = np.flatnonzero(wfsum >= thr)
    if above.size == 0:
        return []

    groups = np.split(above, np.flatnonzero(np.diff(above) > holdon + 1) + 1)
    return [IndexRange(int(group[0]), int(group[-1]) + 1)
            for group in groups]


def slice_waveforms(wfms   : Waveforms,
                    windows : Iterable[IndexRange],
                    padding: int) -> list[Waveforms]:
    """
    Slice a waveform collection around validated half-open windows.

    Padding is clipped at the boundaries of the original waveform collection.

    Each window must be nonempty and lie within the sample axis; padding must
    be nonnegative. Violations raise ``ValueError``. Windows are sliced
    independently, so overlapping padded regions are retained in each result.
    """
    if padding < 0:
        raise ValueError("padding must be non-negative")

    slices = []
    for window in windows:
        if len(window) == 0 or window.upper > wfms.nsamples():
            raise ValueError("Window must be non-empty and within the waveforms.")

        lower = max(window.lower - padding, 0)
        upper = min(window.upper + padding, wfms.nsamples())
        sl = slice(lower, upper)
        slices.append(Waveforms(wfms.time[sl],
                                wfms.sensors,
                                wfms.amplitudes[:, sl]))

    return slices
