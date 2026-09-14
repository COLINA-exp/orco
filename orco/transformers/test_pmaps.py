from textwrap import dedent

import numpy  as np
import tables as tb

from numpy.testing import assert_array_equal
from pytest import mark
from pytest import raises

from ..internal.symbols import Detector
from ..internal.types import IndexRange
from ..internal.types import Waveforms
from ..internal.units import units
from . pmaps          import PeakConfig
from . pmaps          import PMapConfig
from . pmaps          import calibration_gains
from . pmaps          import create_pmap
from . pmaps          import find_peaks
from . pmaps          import read_wfms
from . pmaps          import select_windows


SAMPLING_TIME = 0.016


def peak_config(**overrides):
    values = dict(min_samples=1,
                  max_samples=10,
                  min_time=0,
                  max_time=1,
                  threshold=1,
                  holdon=1,
                  padding=1)
    values.update(overrides)
    return PeakConfig(**values)


@mark.parametrize(
    "overrides",
    (dict(min_samples=0),
     dict(min_samples=5, max_samples=4),
     dict(min_time=-1),
     dict(min_time=2, max_time=1),
     dict(holdon=-1),
     dict(padding=-1)),
)
def test_peak_config_rejects_invalid_ranges(overrides):
    with raises(ValueError):
        peak_config(**overrides)


def test_pmap_config_loads_transformer_settings(tmp_path):
    filename = tmp_path / "pmaps.conf"
    filename.write_text(dedent("""
        detector = pColina
        sampling_time = 16 * nanosecond

        s1_min_samples = 1
        s1_max_samples = 10
        s1_min_time = 0 * microsecond
        s1_max_time = 1000 * nanosecond
        s1_threshold = 2
        s1_holdon = 3
        s1_padding = 4

        s2_min_samples = 5
        s2_max_samples = 20
        s2_min_time = 1 * microsecond
        s2_max_time = 2 * microsecond
        s2_threshold = 6
        s2_holdon = 7
        s2_padding = 8
    """))

    config = PMapConfig.load(filename)

    assert config.s1.min_samples == 1
    assert config.s1.max_samples == 10
    assert config.s1.min_time == 0 * units.microsecond
    assert config.s1.max_time == 1 * units.microsecond
    assert config.s2.min_time == 1 * units.microsecond
    assert config.s2.max_time == 2 * units.microsecond
    assert config.sampling_time == 16 * units.nanosecond
    assert config.detector is Detector.pColina


def test_select_windows_includes_time_and_length_boundaries():
    windows = [IndexRange(0, 2), IndexRange(1, 4), IndexRange(2, 5)]

    selected = select_windows(windows,
                              min_time=0,
                              max_time=3 * SAMPLING_TIME,
                              min_samples=2,
                              max_samples=3,
                              sampling_time=SAMPLING_TIME)

    assert selected == windows[:2]


def test_find_peaks_numbers_multiple_peaks():
    amplitudes = np.zeros((1, 10))
    amplitudes[:, [2, 7]] = 2
    waveforms = Waveforms(time=np.arange(10) * SAMPLING_TIME,
                         channels=np.array([10]),
                         amplitudes=amplitudes)

    peaks = find_peaks(waveforms,
                       peak_config(holdon=0, padding=0),
                       SAMPLING_TIME)

    assert peaks.peak.tolist() == [0, 1]
    assert peaks.time.tolist() == [2 * SAMPLING_TIME,
                                   7 * SAMPLING_TIME]


def test_create_pmap_handles_empty_and_nonempty_peak_sets():
    channels = np.array([10, 11])
    amplitudes = np.zeros((2, 20))
    amplitudes[:, 3:5] = -2
    waveforms = Waveforms(time=np.arange(20) * SAMPLING_TIME,
                         channels=channels,
                         amplitudes=amplitudes)
    config = PMapConfig(peak_config(),
                        peak_config(threshold=10),
                        SAMPLING_TIME,
                        Detector.pColina)

    s1, s2 = create_pmap(42, waveforms, np.ones(2), config)

    assert list(s1.columns) == ["event", "peak", "time", "summed",
                                "channel_010", "channel_011"]
    assert set(s1.event) == {42}
    assert set(s1.peak) == {0}
    assert s2.empty


def test_create_pmap_peak_height_matches_summed_waveform():
    # Custom waveform with a Gaussian S1-like pulse, recorded as a negative
    # dip: create_pmap inverts the polarity, so it becomes a positive peak.
    channels = np.array([10, 11])
    samples  = np.arange(60)
    center, sigma = 10, 1
    pulse = -5 * np.exp(-0.5 * ((samples - center) / sigma) ** 2)
    amplitudes = np.tile(pulse, (len(channels), 1))
    waveforms = Waveforms(time=samples * SAMPLING_TIME,
                          channels=channels,
                          amplitudes=amplitudes)
    config = PMapConfig(peak_config(),
                        peak_config(threshold=11),
                        SAMPLING_TIME,
                        Detector.pColina)

    s1, s2 = create_pmap(7, waveforms, np.ones(len(channels)), config)

    # Only the S1 pulse is found, as a single peak.
    assert s2.empty
    assert set(s1.peak) == {0}

    # Height of the summed waveform after the polarity inversion.
    summed_height = (-amplitudes).sum(axis=0).max()

    # The resulting peak reproduces the summed-waveform height exactly,
    # at the center of the pulse.
    assert s1["summed"].max() == summed_height
    assert s1.loc[s1["summed"].idxmax(), "time"] == center * SAMPLING_TIME


def test_read_wfms_yields_waveform_collections(tmp_path):
    filename = tmp_path / "waveforms.h5"
    amplitudes = np.arange(12).reshape(2, 2, 3)
    channels = np.array([10, 11])
    events = np.array([100, 101])
    with tb.open_file(filename, "w") as h5out:
        group = h5out.create_group("/", "Waveforms")
        h5out.create_array(group, "adcs", amplitudes)
        h5out.create_array(group, "channel_id", channels)
        h5out.create_array(group, "event_id", events)

    sampling_time = 16 * units.nanosecond
    records = list(read_wfms(filename, sampling_time))

    assert [record["event"] for record in records] == events.tolist()
    assert_array_equal(records[0]["wfms"].channels, channels)
    assert_array_equal(records[1]["wfms"].amplitudes, amplitudes[1])
    assert_array_equal(records[0]["wfms"].time,
                       np.arange(3) * SAMPLING_TIME)


def test_calibration_gains_follow_waveform_channel_order():
    gains = calibration_gains(Detector.pColina, -1, [11, 10])

    assert_array_equal(gains, [1., 1.])
