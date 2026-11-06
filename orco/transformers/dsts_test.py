from textwrap import dedent

import numpy as np
import pandas as pd
import tables as tb

from pytest import approx
from pytest import mark

from ..internal.types import ClosedInterval
from ..internal.types import Interval
from ..io.hdf5 import df_writer
from . import dsts
from .dsts import DstConfig
from .dsts import PeakPropertiesConfig
from .dsts import barycenter
from .dsts import blame_peaks
from .dsts import count_peaks
from .dsts import empty_peak
from .dsts import filter_peaks
from .dsts import peak_barycenter
from .dsts import read_pmaps
from .dsts import sensor_multiplicity
from .dsts import summarize_peak


def peak_properties_config(**overrides):
    values = dict(count        = ClosedInterval(0,  10),
                  width        =       Interval(0,  10),
                  height       =       Interval(0,  10),
                  integral     =       Interval(0, 100),
                  multiplicity = ClosedInterval(0,  10),
                  threshold    = 1)
    values.update(overrides)
    return PeakPropertiesConfig(**values)


def test_dst_config_loads_properties_and_units(tmp_path):
    filename = tmp_path / "dsts.conf"
    filename.write_text(dedent("""
        detector = pColina

        s1_min_count = 1
        s1_max_count = 3
        s1_min_width = 200 * ns
        s1_max_width = 1 * microsecond
        s1_min_height = 2
        s1_max_height = 10
        s1_min_integral = 5
        s1_max_integral = 100
        s1_min_multiplicity = 1
        s1_max_multiplicity = 4
        s1_threshold = 1

        s2_min_count = 0
        s2_max_count = 2
        s2_min_width = 300 * ns
        s2_max_width = 2 * microsecond
        s2_min_height = 3
        s2_max_height = 20
        s2_min_integral = 6
        s2_max_integral = 200
        s2_min_multiplicity = 2
        s2_max_multiplicity = 5
        s2_threshold = 2
    """))

    config = DstConfig.load(filename)

    assert config.detector.name == "pColina"
    assert config.s1_props.count == ClosedInterval(1, 3)
    assert config.s1_props.width == Interval(0.2, 1)
    assert config.s1_props.height == Interval(2, 10)
    assert config.s1_props.integral == Interval(5, 100)
    assert config.s1_props.multiplicity == ClosedInterval(1, 4)
    assert config.s1_props.threshold == 1
    assert config.s2_props.width == Interval(0.3, 2)


def test_summarize_peak_uses_file_sampling_time_and_sensor_charges():
    peak = pd.DataFrame({"time": [0.1, 0.2, 0.3],
                         "summed": [3.0, 5.0, 4.0],
                         "sensor_001": [1.0, 2.0, 0.0],
                         "sensor_002": [10.0, 0.0, 1.0],
                         "sensor_003": [1.0, 1.0, 1.0]})

    summary = summarize_peak(peak, threshold=2, ids_np=[1, 3], sampling_time=0.032)

    assert summary.width == approx(3 * 0.032)
    assert summary.height == 5
    assert summary.time == 0.2
    assert summary.integral == 12
    assert summary.multiplicity == 2


def test_summarize_peak_with_no_charge_has_undefined_properties():
    peak = pd.DataFrame({"time": [0.1, 0.2],
                         "summed": [0.0, 0.0],
                         "sensor_001": [0.0, 0.0]})

    summary = summarize_peak(peak, threshold=1, ids_np=[1], sampling_time=0.032)

    assert np.isnan(summary.width)
    assert np.isnan(summary.height)
    assert np.isnan(summary.time)
    assert np.isnan(summary.integral)
    assert summary.multiplicity == 0


def test_sensor_multiplicity_sums_sensor_charge_over_peak_samples():
    peak = pd.DataFrame({"sensor_001": [0.5, 0.5],
                         "sensor_002": [1.0, 1.0],
                         "sensor_003": [4.0, 0.0]})

    assert sensor_multiplicity(peak, threshold=2, sensors_np=[1, 2]) == 1


def test_blame_peaks_reports_failed_properties_and_filter_removes_them():
    peaks = pd.DataFrame({"peak": [0, 0, 1],
                          "time": [0.1, 0.2, 0.3],
                          "summed": [2.0, 3.0, 0.1],
                          "sensor_001": [2.0, 2.0, 0.1]})
    config = peak_properties_config(width=Interval(0.5, 1.1),
                                    height=Interval(2, 4),
                                    integral=Interval(4, 6),
                                    multiplicity=ClosedInterval(1, 1))

    blame = blame_peaks(peaks, config, ids_np=[1], sampling_time=0.5)
    selected = filter_peaks(peaks, blame)

    assert blame.to_dict("records") == [
        {"peak": 1, "blame": "height, integral, multiplicity"},
    ]
    assert selected.peak.unique().tolist() == [0]
    assert len(selected) == 2


def test_blame_peaks_preserves_schema_when_no_peaks_are_rejected():
    peaks = pd.DataFrame({"peak": [0],
                          "time": [0.1],
                          "summed": [2.0],
                          "sensor_001": [2.0]})

    blame = blame_peaks(peaks, peak_properties_config(), ids_np=[1], sampling_time=0.5)

    assert blame.empty
    assert blame.columns.tolist() == ["peak", "blame"]
    assert pd.api.types.is_integer_dtype(blame.peak.dtype)
    assert pd.api.types.is_object_dtype(blame.blame.dtype)


@mark.parametrize(("s1_peaks", "s2_peaks", "expected_blame"),
                  (([0], [0, 1], ""),
                   ([], [0], "S1"),
                   ([0, 1, 2], [0, 1, 2], "S1, S2")))
def test_count_peaks_checks_inclusive_count_ranges(s1_peaks, s2_peaks, expected_blame):
    s1s = pd.DataFrame({"peak": s1_peaks})
    s2s = pd.DataFrame({"peak": s2_peaks})

    counts = count_peaks(s1s, s2s,
                         s1_range=ClosedInterval(1, 2),
                         s2_range=ClosedInterval(1, 2))

    assert counts.s1_count == len(set(s1_peaks))
    assert counts.s2_count == len(set(s2_peaks))
    assert counts.blame == expected_blame


def test_barycenter_uses_sensors_at_the_threshold():
    x, y = barycenter(np.array([0.0, 10.0, 20.0]),
                      np.array([0.0, 10.0, 20.0]),
                      np.array([1.0, 2.0, 3.0]),
                      threshold=2)

    assert x == approx(16)
    assert y == approx(16)


def test_barycenter_without_qualifying_sensors_is_undefined():
    x, y = barycenter(np.array([1.0]), np.array([2.0]), np.array([1.0]), threshold=2)

    assert np.isnan(x)
    assert np.isnan(y)


def test_peak_barycenter_uses_only_near_plane_sensors():
    peak = pd.DataFrame({"sensor_010": [1.0, 2.0],
                         "sensor_011": [3.0, 1.0],
                         "sensor_099": [100.0, 100.0]})
    sensors_np = pd.DataFrame({"SensorID": [10, 11],
                               "X": [0.0, 10.0],
                               "Y": [5.0, 15.0]})

    x, y = peak_barycenter(peak, sensors_np, threshold=3)

    assert x == approx(40 / 7)
    assert y == approx(75 / 7)


def test_empty_peak_keeps_sensor_schema():
    sensors = pd.DataFrame({"SensorID": [1, 20]})

    peak = empty_peak(sensors)

    assert peak.columns.tolist() == ["peak", "time", "summed", "sensor_001", "sensor_020"]
    assert peak.peak.tolist() == [-1]
    assert peak.time.tolist() == [-1]
    assert np.isnan(peak.summed.iloc[0])
    assert peak.filter(like="sensor_").isna().all().all()


def test_main_uses_pmap_sampling_time_when_writing_dst(tmp_path, monkeypatch):
    sampling_time = 0.032
    input_file = tmp_path / "pmaps.h5"
    peaks = pd.DataFrame({"event": [42, 42],
                          "peak": [0, 0],
                          "time": [1.0, 1.032],
                          "summed": [2.0, 3.0],
                          "sensor_001": [1.0, 2.0],
                          "sensor_002": [1.0, 2.0]})
    with tb.open_file(input_file, "w") as h5out:
        h5out.root._v_attrs.sampling_time_microseconds = sampling_time
        df_writer(h5out, pd.DataFrame({"event": [42]}), "PMaps", "events")
        df_writer(h5out, peaks, "PMaps", "S1")
        df_writer(h5out, peaks, "PMaps", "S2")

    config_file = tmp_path / "dsts.conf"
    config_file.write_text(dedent("""
        detector = pColina

        s1_min_count = 1
        s1_max_count = 1
        s1_min_width = 0 * microsecond
        s1_max_width = 1 * microsecond
        s1_min_height = 0
        s1_max_height = 10
        s1_min_integral = 0
        s1_max_integral = 10
        s1_min_multiplicity = 0
        s1_max_multiplicity = 2
        s1_threshold = 1

        s2_min_count = 1
        s2_max_count = 1
        s2_min_width = 0 * microsecond
        s2_max_width = 1 * microsecond
        s2_min_height = 0
        s2_max_height = 10
        s2_min_integral = 0
        s2_max_integral = 10
        s2_min_multiplicity = 0
        s2_max_multiplicity = 2
        s2_threshold = 1
    """))
    sensors = pd.DataFrame({"SensorID": [1, 2],
                            "Plane": ["NP", "NP"],
                            "X": [0.0, 10.0],
                            "Y": [0.0, 20.0]})
    monkeypatch.setattr(dsts, "load_sensor_position", lambda detector, run: sensors)
    output_file = tmp_path / "dsts.h5"

    dsts.main(["-i", str(input_file),
               "-o", str(output_file),
               "-c", str(config_file),
               "-r", "0"])

    with tb.open_file(output_file) as h5out:
        counts = pd.DataFrame(h5out.root.DST.EventPeakCount.read(),
                              columns=h5out.root.DST.EventPeakCount.colnames)
        pointlike = pd.DataFrame(h5out.root.DST.Pointlike.read(),
                                 columns=h5out.root.DST.Pointlike.colnames)

    assert counts.loc[0, "event"] == 42
    assert counts.loc[0, "s1_count"] == 1
    assert counts.loc[0, "s2_count"] == 1
    assert counts.loc[0, "blame"] == b""
    assert pointlike.loc[0, "s1_width"] == approx(2 * sampling_time)
    assert pointlike.loc[0, "s2_width"] == approx(2 * sampling_time)
    assert pointlike.loc[0, "x"] == approx(5)
    assert pointlike.loc[0, "y"] == approx(10)
