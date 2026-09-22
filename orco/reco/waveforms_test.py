import numpy as np

from numpy.testing import assert_array_equal
from pytest import raises

from ..internal.types import IndexRange
from ..internal.types import Waveforms

from . waveforms import calibrate
from . waveforms import find_windows
from . waveforms import invert_polarity
from . waveforms import remove_baseline
from . waveforms import slice_waveforms


def waveform_collection(amplitudes):
    amplitudes = np.asarray(amplitudes)
    return Waveforms(time=np.arange(amplitudes.shape[1]),
                     sensors=np.arange(amplitudes.shape[0]),
                     amplitudes=amplitudes)


def test_remove_baseline_subtracts_each_sensor_median():
    waveforms = waveform_collection([[2, 2, 0, 2, 2],
                                     [5, 5, 1, 5, 5]])

    corrected = remove_baseline(waveforms)

    assert_array_equal(corrected.amplitudes, [[0, 0, -2, 0, 0],
                                              [0, 0, -4, 0, 0]])


def test_invert_polarity_negates_amplitudes():
    inverted = invert_polarity(waveform_collection([[0, 2]]))

    assert_array_equal(inverted.amplitudes, [[0, -2]])


def test_calibrate_applies_one_gain_per_channel():
    calibrated = calibrate(waveform_collection([[1, 2], [3, 4]]), [2, 0.5])

    assert_array_equal(calibrated.amplitudes, [[0.5, 1], [6, 8]])


def test_calibrate_rejects_invalid_gains():
    with raises(ValueError, match="each waveform sensor"):
        calibrate(waveform_collection(np.ones((2, 3))), [1])
    with raises(ValueError, match="Invalid gain"):
        calibrate(waveform_collection(np.ones((1, 3))), [0])


def test_find_windows_merges_short_gaps():
    waveform = [0, 2, 2, 0, 2, 0, 0]

    windows = find_windows(waveform, thr=1, holdon=1)

    assert windows == [IndexRange(1, 5)]


def test_find_windows_returns_empty_list_without_crossings():
    assert find_windows(np.zeros(5), thr=1, holdon=2) == []


def test_find_windows_rejects_invalid_arguments():
    with raises(ValueError, match="non-negative"):
        find_windows(np.ones(5), thr=1, holdon=-1)
    with raises(ValueError, match="one-dimensional"):
        find_windows(np.ones((2, 2)), thr=1, holdon=1)


def test_slice_waveforms_applies_padding_and_clamps_to_edges():
    waveforms = waveform_collection(np.arange(10).reshape(2, 5))

    slices = slice_waveforms(waveforms,
                            [IndexRange(0, 1), IndexRange(3, 5)],
                            padding=1)

    assert_array_equal(slices[0].time, [0, 1])
    assert_array_equal(slices[0].amplitudes, waveforms.amplitudes[:, :2])
    assert_array_equal(slices[1].time, [2, 3, 4])
    assert_array_equal(slices[1].amplitudes, waveforms.amplitudes[:, 2:])


def test_slice_waveforms_rejects_invalid_inputs():
    waveforms = waveform_collection(np.ones((2, 5)))
    with raises(ValueError, match="non-negative"):
        slice_waveforms(waveforms, [IndexRange(1, 2)], padding=-1)
    with raises(ValueError, match="non-empty"):
        slice_waveforms(waveforms, [IndexRange(2, 2)], padding=0)
    with raises(ValueError, match="within"):
        slice_waveforms(waveforms, [IndexRange(2, 6)], padding=0)
