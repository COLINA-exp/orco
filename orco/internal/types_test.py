from dataclasses import FrozenInstanceError

import numpy as np

from numpy.testing import assert_array_equal
from pytest import mark
from pytest import raises

from .exceptions import InvalidIndexRange
from .exceptions import InvalidShape
from .types      import IndexRange
from .types      import Waveforms


def test_index_range_accepts_half_open_and_empty_ranges():
    assert IndexRange(2, 5) == IndexRange(lower=2, upper=5)
    assert IndexRange(3, 3).lower == IndexRange(3, 3).upper


def test_index_range_rejects_invalid_bounds():
    with raises(InvalidIndexRange, match="negative"):
        IndexRange(-1, 2)
    with raises(InvalidIndexRange, match="exceed"):
        IndexRange(3, 2)


def test_waveforms_accepts_consistent_shapes():
    waveforms = Waveforms(time       = np.arange(4),
                          sensors    = np.array([10, 11]),
                          amplitudes = np.zeros((2, 4)))

    assert waveforms.nsensors() == 2
    assert waveforms.nsamples () == 4
    assert np.issubdtype(waveforms.amplitudes.dtype, np.floating)


def test_waveforms_converts_integer_amplitudes_to_float():
    waveforms = Waveforms(time       = np.arange(1),
                          sensors    = np.array([10, 11]),
                          amplitudes = np.array([[40_000], [40_000]],
                                                dtype=np.uint16))

    assert np.issubdtype(waveforms.amplitudes.dtype, np.floating)
    assert_array_equal(waveforms.sum_sensors(), [80_000])


@mark.parametrize(
    "time, channels, amplitudes",
    ((np.array(0),       np.array([10, 11]), np.zeros((2, 4))),
     (np.zeros((1, 4)),  np.array([10, 11]), np.zeros((2, 4))),
     (np.arange(4),      np.array(10),       np.zeros((2, 4))),
     (np.arange(4),      np.zeros((1, 2)),   np.zeros((2, 4))),
     (np.arange(4),      np.array([10, 11]), np.zeros(4)),
     (np.arange(4),      np.array([10, 11]), np.zeros((1, 2, 4)))),
    ids=("scalar-time",
         "matrix-time",
         "scalar-channels",
         "matrix-channels",
         "vector-amplitudes",
         "tensor-amplitudes"),
)
def test_waveforms_rejects_invalid_dimensions(time, channels, amplitudes):
    with raises(InvalidShape, match="Invalid input shapes"):
        Waveforms(time=time, sensors=channels, amplitudes=amplitudes)


def test_waveforms_rejects_inconsistent_shapes():
    with raises(InvalidShape, match="not consistent"):
        Waveforms(time       = np.arange(4),
                  sensors    = np.array([10, 11]),
                  amplitudes = np.zeros((3, 4)))


def test_waveforms_rejects_empty_axes():
    with raises(InvalidShape, match="at least one sensor and one sample"):
        Waveforms(time       = np.array([]),
                  sensors    = np.array([10, 11]),
                  amplitudes = np.empty((2, 0)))


def test_waveforms_cannot_be_changed_after_construction():
    time = np.arange(4)
    waveforms = Waveforms(time       = time,
                          sensors    = np.array([10, 11]),
                          amplitudes = np.zeros((2, 4)))

    time[0] = 100
    assert waveforms.time[0] == 0

    with raises(FrozenInstanceError):
        waveforms.time = np.arange(4)
    with raises(ValueError, match="read-only"):
        waveforms.amplitudes[0, 0] = 1
