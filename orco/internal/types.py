"""
Small domain-specific value types shared by ORCO modules.
"""

from dataclasses import dataclass

import numpy as np

from . exceptions import InvalidIndexRange
from . exceptions import InvalidShape


@dataclass(frozen=True, slots=True)
class IndexRange:
    """
    A validated half-open index range ``[lower, upper)``. Equal bounds represent
    an empty range.

    Attributes:
        lower: Inclusive, non-negative start index.
        upper: Exclusive stop index, greater than or equal to ``lower``.

    Raises:
        InvalidIndexRange: If a bound is negative or bounds are reversed.
    """

    lower: int
    upper: int

    def __post_init__(self) -> None:
        if self.lower < 0:
            raise InvalidIndexRange("IndexRange: lower bound cannot be negative")
        if self.lower > self.upper:
            raise InvalidIndexRange("IndexRange: lower bound cannot exceed upper bound")

    def __len__(self) -> int:
        return self.upper - self.lower

    def __contains__(self, value: int) -> bool:
        return self.lower <= value < self.upper


@dataclass(frozen=True, slots=True)
class Waveforms:
    """
    A collection of sensor waveforms sharing a sampling-time array.
    The amplitude array follows ``(channel, sample)`` ordering.
    Waveforms are stored as floats.

    Attributes:
        time      : One-dimensional sampling-time array.
        channels  : One-dimensional sensor-identifier array.
        amplitudes: Two-dimensional waveform-amplitude array.

    Raises:
        InvalidShape: If an array has the wrong dimensionality or their sizes
            are not mutually consistent.
    """

    time      : np.ndarray
    channels  : np.ndarray
    amplitudes: np.ndarray

    def __post_init__(self) -> None:
        time       = np.array(self.time)
        channels   = np.array(self.channels)
        amplitudes = np.array(self.amplitudes, dtype=float)
        time      .setflags(write=False)
        channels  .setflags(write=False)
        amplitudes.setflags(write=False)

        if (time.ndim != 1 or channels.ndim != 1 or amplitudes.ndim != 2):
            raise InvalidShape("Waveforms: Invalid input shapes.")

        if channels.size == 0 or time.size == 0:
            raise InvalidShape("Waveforms: Waveforms must have at least one sensor and one sample")

        if (channels.size, time.size) != amplitudes.shape:
            raise InvalidShape("Waveforms: Shape of inputs is not consistent.")

        object.__setattr__(self, "time"      , time)
        object.__setattr__(self, "channels"  , channels)
        object.__setattr__(self, "amplitudes", amplitudes)

    def nchannels(self) -> int:
        return len(self.channels)

    def nsamples(self) -> int:
        return len(self.time)

    def sum_sensors(self) -> np.ndarray:
        return self.amplitudes.sum(axis=0, dtype=float)

    def sum_samples(self) -> np.ndarray:
        return self.amplitudes.sum(axis=1, dtype=float)
