"""
Small domain-specific value types shared by ORCO modules.
"""

from dataclasses import dataclass
from typing      import Generic, TypeVar

import numpy as np

from .exceptions import InvalidRange
from .exceptions import InvalidIndexRange
from .exceptions import InvalidShape


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Range(Generic[T]):
    """
    A validated half-open range ``[lower, upper)``. Equal bounds represent
    an empty range.

    Attributes:
        lower: Inclusive lower limit.
        upper: Exclusive upper limit, greater than or equal to ``lower``.

    Raises:
        InvalidRange: If the bounds are reversed.
    """
    lower: T
    upper: T

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise InvalidRange("Range: lower bound cannot exceed upper bound")

    @property
    def length(self) -> T:
        return self.upper - self.lower

    def __contains__(self, value: T) -> bool:
        return self.lower <= value < self.upper


@dataclass(frozen=True, slots=True)
class Interval(Range[float]):
    """
    A validated half-open range of real values ``[lower, upper)``.
    """


@dataclass(frozen=True, slots=True)
class ClosedInterval(Range[float]):
    """
    A validated closed range of real values ``[lower, upper]``.
    """

    def __contains__(self, value: float) -> bool:
        return self.lower <= value <= self.upper


@dataclass(frozen=True, slots=True)
class IndexRange(Range[int]):
    """
    A validated half-open index range ``[lower, upper)`` with non-negative
    integer bounds.

    Attributes:
        lower: Inclusive, non-negative start index.
        upper: Exclusive stop index, greater than or equal to ``lower``.

    Raises:
        InvalidIndexRange: If a bound is negative. Reversed bounds raise
            InvalidRange from the base class.
    """

    def __post_init__(self) -> None:
        Range.__post_init__(self)
        if self.lower < 0:
            raise InvalidIndexRange("IndexRange: lower bound cannot be negative")

    def __len__(self) -> int:
        return self.length


@dataclass(frozen=True, slots=True)
class Waveforms:
    """
    A collection of sensor waveforms sharing a sampling-time array.
    The amplitude array follows ``(sensor, sample)`` ordering.
    Waveforms are stored as floats. Input arrays are copied and marked read-only;
    attribute assignment is prevented by the frozen dataclass. Time/amplitude
    units are provided by callers, normally microseconds and calibrated amplitude
    in reconstruction. There must be at least one sensor and one sample.

    Attributes:
        time      : One-dimensional sampling-time array.
        sensors   : One-dimensional sensor-identifier array.
        amplitudes: Two-dimensional waveform-amplitude array.

    Raises:
        InvalidShape: If an array has the wrong dimensionality or their sizes
            are not mutually consistent.
    """

    time      : np.ndarray
    sensors   : np.ndarray
    amplitudes: np.ndarray

    def __post_init__(self) -> None:
        time       = np.array(self.time)
        sensors    = np.array(self.sensors)
        amplitudes = np.array(self.amplitudes, dtype=float)
        time      .setflags(write=False)
        sensors   .setflags(write=False)
        amplitudes.setflags(write=False)

        if (time.ndim != 1 or sensors.ndim != 1 or amplitudes.ndim != 2):
            raise InvalidShape("Waveforms: Invalid input shapes.")

        if sensors.size == 0 or time.size == 0:
            raise InvalidShape("Waveforms: Waveforms must have at least one sensor and one sample")

        if (sensors.size, time.size) != amplitudes.shape:
            raise InvalidShape("Waveforms: Shape of inputs is not consistent.")

        object.__setattr__(self, "time"      , time)
        object.__setattr__(self, "sensors"   , sensors)
        object.__setattr__(self, "amplitudes", amplitudes)

    def nsensors(self) -> int:
        return len(self.sensors)

    def nsamples(self) -> int:
        return len(self.time)

    def sum_sensors(self) -> np.ndarray:
        return self.amplitudes.sum(axis=0, dtype=float)

    def sum_samples(self) -> np.ndarray:
        return self.amplitudes.sum(axis=1, dtype=float)
