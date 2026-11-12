from textwrap import dedent
from pathlib import Path
from json import dumps

import numpy as np
import pandas as pd
import tables as tb

from pytest import approx
from pytest import mark
from pytest import raises

from ..internal.symbols import Detector
from ..internal.types import ClosedInterval
from ..internal.types import Interval
from ..internal.types import Waveforms
from ..io.hdf5 import df_writer
from ..reco.waveforms import calibrate
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
from .dsts import sensor_charges
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


@mark.parametrize(("field", "expression"),
                  ((None, None),
                   ("s1_min_width", "0"),
                   ("s1_min_height", "2 * pe"),
                   ("s1_min_integral", "5 * pe / microsecond"),
                   ("s2_threshold", "1 * adc"),
                   ("s1_min_count", "1 * microsecond"),
                   ("s2_max_multiplicity", "1.5")))
def test_dst_config_loads_properties_and_units(tmp_path, field, expression):
    filename = tmp_path / "dsts.conf"
    filename.write_text(dedent("""
        s1_min_count = 1
        s1_max_count = 3
        s1_min_width = 200 * ns
        s1_max_width = 1 * microsecond
        s1_min_height = 2000 * pe / millisecond
        s1_max_height = 10 * pe / microsecond
        s1_min_integral = 5 * pe
        s1_max_integral = 0.1 * kilophotoelectron
        s1_min_multiplicity = 1
        s1_max_multiplicity = 4
        s1_threshold = 1 * pe

        s2_min_count = 0
        s2_max_count = 2
        s2_min_width = 300 * ns
        s2_max_width = 2 * microsecond
        s2_min_height = 3 * pe / microsecond
        s2_max_height = 20 * pe / microsecond
        s2_min_integral = 6 * pe
        s2_max_integral = 200 * pe
        s2_min_multiplicity = 2
        s2_max_multiplicity = 5
        s2_threshold = 2 * photoelectron
    """))

    if field is not None:
        filename.write_text(filename.read_text() + f"\n{field} = {expression}\n")
        with raises(ValueError, match=field):
            DstConfig.load(filename)
        return

    config = DstConfig.load(filename)

    assert not hasattr(config, "detector")
    assert config.s1_props.count == ClosedInterval(1, 3)
    assert config.s1_props.width == Interval(0.2, 1)
    assert config.s1_props.height == Interval(2, 10)
    assert config.s1_props.integral == Interval(5, 100)
    assert config.s1_props.multiplicity == ClosedInterval(1, 4)
    assert config.s1_props.threshold == 1
    assert config.s2_props.width == Interval(0.3, 2)
    assert config.s2_props.threshold == 2


@mark.parametrize("sampling_time", (0.016, 0.032, 0.5, 1.0))
def test_summarize_peak_uses_file_sampling_time_and_sensor_charges(sampling_time):
    peak = pd.DataFrame({"time": [0.1, 0.2, 0.3],
                         "summed": [3.0, 5.0, 4.0],
                         "sensor_001": [1.0, 2.0, 0.0],
                         "sensor_002": [10.0, 0.0, 1.0],
                         "sensor_003": [1.0, 1.0, 1.0]})

    summary = summarize_peak(peak, threshold=0.04, sensor_ids=[1, 3], sampling_time=sampling_time)

    assert summary.width == approx(3 * sampling_time)
    assert summary.height == 5
    assert summary.time == 0.2
    assert summary.integral == approx(12 * sampling_time)
    assert summary.multiplicity == 2


def test_summarize_peak_with_no_charge_has_undefined_properties():
    peak = pd.DataFrame({"time": [0.1, 0.2],
                         "summed": [0.0, 0.0],
                         "sensor_001": [0.0, 0.0]})

    summary = summarize_peak(peak, threshold=1, sensor_ids=[1], sampling_time=0.032)

    assert np.isnan(summary.width)
    assert np.isnan(summary.height)
    assert np.isnan(summary.time)
    assert np.isnan(summary.integral)
    assert summary.multiplicity == 0


def test_sensor_multiplicity_sums_sensor_charge_over_peak_samples():
    peak = pd.DataFrame({"sensor_001": [0.5, 0.5],
                         "sensor_002": [1.0, 1.0],
                         "sensor_003": [4.0, 0.0]})

    assert sensor_multiplicity(peak, threshold=1, sensors=[1, 2], sampling_time=0.5) == 1


def test_blame_peaks_reports_failed_properties_and_filter_removes_them():
    peaks = pd.DataFrame({"peak": [0, 0, 1],
                          "time": [0.1, 0.2, 0.3],
                          "summed": [2.0, 3.0, 0.1],
                          "sensor_001": [2.0, 2.0, 0.1]})
    config = peak_properties_config(width=Interval(0.5, 1.1),
                                    height=Interval(2, 4),
                                    integral=Interval(2, 3),
                                    multiplicity=ClosedInterval(1, 1))

    blame = blame_peaks(peaks, config, sensor_ids=[1], sampling_time=0.5)
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

    blame = blame_peaks(peaks, peak_properties_config(), sensor_ids=[1], sampling_time=0.5)

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

    x, y = peak_barycenter(peak, sensors_np, threshold=1.5, sampling_time=0.5)

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


@mark.parametrize("sampling_time", (0.016, 0.032, 0.1))
@mark.parametrize("has_acquisition", (False, True))
@mark.parametrize("attribute_time", (None, 999))
def test_main_uses_json_sampling_time_when_writing_dst(tmp_path, monkeypatch, sampling_time,
                                                    has_acquisition, attribute_time):
    input_file = tmp_path / "pmaps.h5"
    acquisition_config = dict(run_number=0, detector="pCOLINA",
                              effective=dict(sample_period_ns=sampling_time * 1000))
    payload = dumps(acquisition_config).encode("utf-8")
    peaks = pd.DataFrame({"event": [42, 42],
                          "peak": [0, 0],
                          "time": [1.0, 1.0 + sampling_time],
                          "summed": [2.0, 3.0],
                          "sensor_001": [1.0, 2.0],
                          "sensor_002": [1.0, 2.0]})
    with tb.open_file(input_file, "w") as h5out:
        if has_acquisition:
            acquisition = h5out.create_group("/", "Acquisition")
            acquisition._v_attrs.run_number = 0
            h5out.create_array(acquisition, "config_json", payload)
        if attribute_time is not None:
            h5out.root._v_attrs.sampling_time_microseconds = attribute_time
        df_writer(h5out, pd.DataFrame({"event": [42]}), "PMaps", "events")
        df_writer(h5out, peaks, "PMaps", "S1")
        df_writer(h5out, peaks, "PMaps", "S2")

    config_file = tmp_path / "dsts.conf"
    config_file.write_text(dedent("""
        # A stale setting must not override the acquisition detector.
        detector = Colina
        sampling_time = 999 * microsecond

        s1_min_count = 1
        s1_max_count = 1
        s1_min_width = 0 * microsecond
        s1_max_width = 1 * microsecond
        s1_min_height = 0 * pe / microsecond
        s1_max_height = 10 * pe / microsecond
        s1_min_integral = 0 * pe
        s1_max_integral = 10 * pe
        s1_min_multiplicity = 0
        s1_max_multiplicity = 2
        s1_threshold = 0.04 * pe

        s2_min_count = 1
        s2_max_count = 1
        s2_min_width = 0 * microsecond
        s2_max_width = 1 * microsecond
        s2_min_height = 0 * pe / microsecond
        s2_max_height = 10 * pe / microsecond
        s2_min_integral = 0 * pe
        s2_max_integral = 10 * pe
        s2_min_multiplicity = 0
        s2_max_multiplicity = 2
        s2_threshold = 0.04 * pe
    """))
    sensors = pd.DataFrame({"SensorID": [1, 2],
                            "Plane": ["NP", "NP"],
                            "X": [0.0, 10.0],
                            "Y": [0.0, 20.0]})
    def load_sensor_position(detector, run):
        assert detector is Detector.pColina
        assert run == 0
        return sensors

    monkeypatch.setattr(dsts, "load_sensor_position", load_sensor_position)
    output_file = tmp_path / "dsts.h5"
    arguments = ["-i", str(input_file), "-o", str(output_file),
                 "-c", str(config_file), "-r", "0"]

    if not has_acquisition:
        original = b"existing output must survive missing acquisition metadata"
        output_file.write_bytes(original)
        with raises(tb.NoSuchNodeError, match="/Acquisition/config_json"):
            dsts.main(arguments)
        assert output_file.read_bytes() == original
        return

    dsts.main(arguments)

    with tb.open_file(output_file) as h5out:
        assert h5out.root.Acquisition._v_attrs.run_number == 0
        assert h5out.root.Acquisition.config_json.read() == payload
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
    assert pointlike.loc[0, "s1_integral"] == approx(5 * sampling_time)
    assert pointlike.loc[0, "s2_integral"] == approx(5 * sampling_time)
    assert pointlike.loc[0, "s1_multiplicity"] == 0
    assert pointlike.loc[0, "s2_multiplicity"] == 2
    assert pointlike.loc[0, "x"] == approx(5)
    assert pointlike.loc[0, "y"] == approx(10)


@mark.parametrize("metadata", ({}, dict(effective={})))
def test_main_rejects_missing_json_sampling_time_before_opening_output(tmp_path, metadata):
    input_file = tmp_path / "pmaps.h5"
    output_file = tmp_path / "dsts.h5"
    config = dict(detector="pCOLINA", **metadata)
    with tb.open_file(input_file, "w") as output:
        acquisition = output.create_group("/", "Acquisition")
        output.create_array(acquisition, "config_json", dumps(config).encode("utf-8"))
        output.root._v_attrs.sampling_time_microseconds = 0.016
    original = b"existing output must survive missing JSON sampling metadata"
    output_file.write_bytes(original)
    config_file = Path(__file__).resolve().parents[2] / "config" / "dsts_mc.conf"

    with raises(KeyError):
        dsts.main(["-i", str(input_file), "-o", str(output_file),
                   "-c", str(config_file), "-r", "0", "--quiet"])

    assert output_file.read_bytes() == original


def test_read_pmaps_yields_events_without_reading_sampling_metadata(tmp_path):
    input_file = tmp_path / "pmaps.h5"
    peaks = pd.DataFrame(dict(event=[42], peak=[0], time=[0.1], summed=[2.]))
    with tb.open_file(input_file, "w") as output:
        df_writer(output, pd.DataFrame(dict(event=[42, 7, 99])), "PMaps", "events")
        df_writer(output, peaks, "PMaps", "S1")
        df_writer(output, peaks.assign(event=7), "PMaps", "S2")

    events = list(read_pmaps(input_file))

    assert [data["event"] for data in events] == [42, 7, 99]
    assert all(set(data) == {"event", "s1s", "s2s"} for data in events)
    assert events[0]["s1s"].to_dict("records") == peaks.to_dict("records")
    assert events[1]["s2s"].to_dict("records") == peaks.assign(event=7).to_dict("records")
    assert events[0]["s2s"].empty
    assert events[1]["s1s"].empty
    assert events[2]["s1s"].empty
    assert events[2]["s2s"].empty


@mark.parametrize("sampling_time", (0.016, 0.032, 0.5))
@mark.parametrize("amplitudes", ((5.0,), (2.0, 3.0)))
def test_sensor_charges_integrates_single_and_multiple_samples(sampling_time, amplitudes):
    peak = pd.DataFrame({"sensor_001": amplitudes})

    ids, charges = sensor_charges(peak, sampling_time)
    summary = summarize_peak(peak.assign(time=np.arange(len(peak)) * sampling_time,
                                         summed=peak.sensor_001),
                              threshold=0, sensor_ids=[1], sampling_time=sampling_time)

    assert ids.tolist() == [1]
    assert charges[0] == approx(sum(amplitudes) * sampling_time)
    assert summary.integral == approx(charges[0])


@mark.parametrize(("sampling_time", "rejected"),
                  ((0.4, False), (0.5, False), (0.6, True), (1.0, True)))
def test_integral_cuts_use_time_weighted_charge(sampling_time, rejected):
    peak = pd.DataFrame(dict(peak=[0, 0], time=[0.0, sampling_time],
                             summed=[2.0, 3.0], sensor_001=[2.0, 3.0]))
    config = peak_properties_config(integral=Interval(2, 3))

    blame = blame_peaks(peak, config, sensor_ids=[1], sampling_time=sampling_time)

    assert blame.blame.tolist() == (["integral"] if rejected else [])


@mark.parametrize("nsamples", (1, 4, 8))
def test_equivalent_pulses_have_sampling_independent_charge_and_position(nsamples):
    sampling_time = 1 / nsamples
    peak = pd.DataFrame(dict(time=np.arange(nsamples) * sampling_time,
                             summed=np.full(nsamples, 6.0),
                             sensor_001=np.full(nsamples, 2.0),
                             sensor_002=np.full(nsamples, 4.0)))
    sensors = pd.DataFrame(dict(SensorID=[1, 2], X=[0.0, 10.0], Y=[0.0, 20.0]))

    summary = summarize_peak(peak, threshold=2, sensor_ids=[1, 2], sampling_time=sampling_time)
    x, y = peak_barycenter(peak, sensors, threshold=2, sampling_time=sampling_time)

    assert summary.integral == approx(6)
    assert summary.multiplicity == 2
    assert x == approx(40 / 6)
    assert y == approx(80 / 6)


@mark.parametrize(("sampling_time", "expected_count", "expected_x"),
                  ((0.25, 0, np.nan), (0.5, 1, 10.0), (1.0, 2, 20 / 3)))
def test_charge_thresholds_control_multiplicity_and_barycenter(sampling_time, expected_count, expected_x):
    peak = pd.DataFrame(dict(sensor_001=[1.0], sensor_002=[2.0]))
    sensors = pd.DataFrame(dict(SensorID=[1, 2], X=[0.0, 10.0], Y=[0.0, 0.0]))

    count = sensor_multiplicity(peak, threshold=1, sensors=[1, 2], sampling_time=sampling_time)
    x, y = peak_barycenter(peak, sensors, threshold=1, sampling_time=sampling_time)

    assert count == expected_count
    if expected_count == 0:
        assert np.isnan(x)
        assert np.isnan(y)
    else:
        assert x == approx(expected_x)
        assert y == 0


def test_single_photoelectron_area_calibrates_to_one_pe():
    sampling_time = 0.1
    amplitudes = np.array([[4.0, 6.0]])
    gain = amplitudes.sum() * sampling_time  # one-pe pulse area, in ADC * us / pe
    waveforms = Waveforms(np.arange(2) * sampling_time, np.array([1]), amplitudes)
    calibrated = calibrate(waveforms, [gain])
    peak = pd.DataFrame(dict(time=calibrated.time, summed=calibrated.sum_sensors(),
                             sensor_001=calibrated.amplitudes[0]))

    summary = summarize_peak(peak, threshold=1, sensor_ids=[1], sampling_time=sampling_time)
    _, charges = sensor_charges(peak, sampling_time)

    assert summary.integral == approx(1)
    assert charges[0] == approx(1)
    assert summary.multiplicity == 1


def test_dummy_peak_keeps_undefined_summary_and_position_after_integration():
    sensors = pd.DataFrame(dict(SensorID=[1], X=[0.0], Y=[0.0]))
    peak = empty_peak(sensors)

    summary = summarize_peak(peak, threshold=1, sensor_ids=[1], sampling_time=0.032)
    x, y = peak_barycenter(peak, sensors, threshold=1, sampling_time=0.032)

    assert np.isnan(summary.integral)
    assert summary.multiplicity == 0
    assert np.isnan(x)
    assert np.isnan(y)
