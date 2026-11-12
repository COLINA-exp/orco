#!/usr/bin/env python3
"""
Build calibrated waveform slices containing signal (PMaps) from raw HDF5 waveforms.

Each event is baseline-subtracted, polarity-inverted and gain-corrected before
independent S1/S2 searches in the summed waveform. Output tables retain padded
peak samples and sensor identifiers; sampling time is stored in microseconds as
root metadata. The detector and sampling interval come from acquisition JSON;
the full ``/Acquisition`` group is copied to the output.
See ``docs/configuration.md`` and ``docs/formats.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

import argparse

import warnings

import numpy  as np
import tables as tb
import pandas as pd

from typing import Any, Iterator, Mapping

from ..database.io      import load_channel_mapping
from ..database.io      import load_sensor_calibration
from ..internal.config  import config_from_scope
from ..internal.symbols import Detector
from ..internal.symbols import all_symbols
from ..internal.types   import IndexRange
from ..internal.types   import Waveforms
from ..internal.units   import Amplitude
from ..internal.units   import Time
from ..internal.units   import unit_scope
from ..io.hdf5          import copy_daq_config
from ..io.hdf5          import df_writer
from ..io.hdf5          import get_sampling_time
from ..io.hdf5          import read_daq_config
from ..reco.waveforms   import calibrate
from ..reco.waveforms   import find_windows
from ..reco.waveforms   import invert_polarity
from ..reco.waveforms   import remove_baseline
from ..reco.waveforms   import slice_waveforms


@dataclass
class PeakConfig:
    min_samples: int
    max_samples: int
    min_time   : Time
    max_time   : Time
    threshold  : Amplitude
    holdon     : int
    padding    : int

    def __post_init__(self):
        if self.min_samples < 1 or self.max_samples < self.min_samples:
            raise ValueError("invalid peak sample limits")
        if self.min_time < 0 or self.max_time < self.min_time:
            raise ValueError("invalid peak time limits")
        if self.holdon < 0 or self.padding < 0:
            raise ValueError("holdon and padding must be non-negative")


@dataclass
class PMapConfig:
    s1           : PeakConfig
    s2           : PeakConfig

    @classmethod
    def load(cls, filename: Path) -> PMapConfig:
        scope = unit_scope()
        scope.update(all_symbols())
        with open(filename) as file:
            exec(file.read(), scope)

        return config_from_scope(cls, scope,
                                 s1=config_from_scope(PeakConfig, scope, prefix="s1_"),
                                 s2=config_from_scope(PeakConfig, scope, prefix="s2_"))


def get_sensor_ids(channels       : np.ndarray,
                   channel_mapping: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Translate the channel identifiers of an input file into sensor identifiers.

    Channels present in the input but absent from the mapping are ignored and
    reported with a warning; channels expected by the mapping but absent from
    the input raise an error.

    Args:
        channels: One-dimensional array of channel identifiers found in
            the input data.
        channel_mapping: Sensor-to-channel mapping valid for the run.

    Returns:
        A tuple ``(sensor_ids, keep)`` with the sensor identifiers of the
        mapped input channels and the boolean mask selecting those channels
        among ``channels``.

    Warns:
        If the input data contains channels not present in the mapping.

    Raises:
        ValueError: If the input data is missing channels expected by the
            mapping.
    """
    expected = channel_mapping.ChannelID.values

    exist_in_db = np.isin(channels, expected)
    if not exist_in_db.all():
        warnings.warn(f"Input data has channels unknown to the mapping; "
                      f"they will be ignored: {channels[~exist_in_db].tolist()}")

    exist_in_file = np.isin(expected, channels)
    if not np.all(exist_in_file):
        missing = expected[~exist_in_file].tolist()
        raise ValueError(f"Input data is missing expected channels: {missing}")

    sensor_ids = (channel_mapping.set_index("ChannelID")
                                 .SensorID
                                 .loc[channels[exist_in_db]]
                                 .values)
    return sensor_ids, exist_in_db


def read_wfms(input_file     : Path,
              channel_mapping: pd.DataFrame,
              sampling_time  : float) -> Iterator[Mapping[str, Any]]:
    """
    Yield raw event waveforms in mapped sensor order.

    Args:
        input_file: HDF5 file with ``/Waveforms/adcs`` in
            ``(event, channel, sample)`` order, plus one-dimensional
            ``channel_id`` and ``event_id`` arrays in that group.
        channel_mapping: Run-valid dataframe with ``ChannelID`` and ``SensorID``.
        sampling_time: Input metadata's sample interval, sample zero has time zero.

    Yields:
        Mappings with ``event`` and ``wfms`` (a ``Waveforms`` instance). Input
        channels absent from the mapping are dropped with a warning. The input
        file remains open while the iterator is consumed.

    Raises:
        RuntimeError: If waveform axes do not match the identifier arrays.
        ValueError: If a channel required by the mapping is missing.
    """
    with tb.open_file(input_file) as infile:
        all_wfms = infile.root.Waveforms.adcs
        channels = infile.root.Waveforms.channel_id.read()
        event_id = infile.root.Waveforms.event_id.read()

        if len(all_wfms.shape) != 3:
            raise RuntimeError("Waveforms must have event, channel, and sample axes")
        if (all_wfms.shape[0], all_wfms.shape[1]) != (event_id.size, channels.size):
            raise RuntimeError("Invalid shape of waveform array")

        sensor_ids, keep = get_sensor_ids(channels, channel_mapping)
        time = np.arange(all_wfms.shape[2]) * sampling_time
        for amplitudes, event in zip(all_wfms, event_id):
            amplitudes = amplitudes[keep]
            yield dict(wfms=Waveforms(time, sensor_ids, amplitudes), event=event)


def select_windows(windows: list[IndexRange],
                   min_time: float,
                   max_time: float,
                   min_samples: int,
                   max_samples: int,
                   sampling_time: float) -> list[IndexRange]:
    """
    Keep unpadded windows within inclusive length and time limits.

    ``min_time`` and ``max_time`` bound the first and last sample times in
    microseconds. For ``[lower, upper)`` those times are ``lower * sampling_time``
    and ``(upper - 1) * sampling_time``. Sample limits apply to ``len(window)``.
    """
    def ok(window):
        start_time = window.lower       * sampling_time
        end_time   = (window.upper - 1) * sampling_time
        nsamples   = len(window)
        return (min_time    <= start_time and
                max_time    >= end_time   and
                min_samples <= nsamples   and
                max_samples >= nsamples)
    return list(filter(ok, windows))


def peak_to_df(peak: Waveforms) -> pd.DataFrame:
    """
    Convert one waveform slice to per-sample ``time``, ``summed`` and sensor data.

    Sensor columns are named ``sensor_<id>`` with a minimum of three digits.
    Event and peak identifiers are added by the caller.
    """
    data = dict(time=peak.time, summed=peak.sum_sensors())
    for c, wf in zip(peak.sensors, peak.amplitudes):
        data[f"sensor_{c:>03}"] = wf
    return pd.DataFrame(data)


def empty_peak_df(sensors: np.ndarray) -> pd.DataFrame:
    """
    Return a typed, zero-row peak table for the supplied sensor identifiers.

    Columns are ``peak``, ``time``, ``summed`` and ``sensor_<id>``. This keeps
    the same schema as a nonempty peak search, without an ``event`` column.
    """
    # in order to keep the type of each column, we give it a dummy value that we
    # later remove
    data = dict(peak=[0], time=[0.], summed=[0.])
    for c in sensors:
        data[f"sensor_{c:>03}"] = [0.]

    return pd.DataFrame(data).iloc[:0]


def find_peaks(wfms         : Waveforms,
               config       : PeakConfig,
               sampling_time: float) -> pd.DataFrame:
    """
    Find, select and pad windows in the sum of all waveform sensors.

    Threshold and hold-on define unpadded windows; time and sample limits select
    them before padding is applied. Returned rows include all padded samples,
    even if padding causes neighboring peaks to overlap. Peak numbers start at
    zero; no matches produce a zero-row table with the same columns.
    """
    windows = find_windows(wfms.sum_sensors(), config.threshold, config.holdon)
    windows = select_windows(windows,
                             config.min_time   , config.max_time,
                             config.min_samples, config.max_samples,
                             sampling_time)
    peaks   = slice_waveforms(wfms, windows, config.padding)
    peaks   = [peak_to_df(peak) for peak in peaks]
    if not peaks:
        return empty_peak_df(wfms.sensors)

    df = pd.concat(peaks, keys=np.arange(len(peaks)))
    df.reset_index(drop=False, inplace=True, level=0, names="peak")
    df.reset_index(drop=True , inplace=True)
    return df


def create_pmap(event: int,
                wfms : Waveforms,
                gains: np.ndarray,
                config: PMapConfig,
                sampling_time: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstruct one event into S1 and S2 sample tables.

    Raw negative-going pulses are median-baseline-subtracted per sensor,
    inverted, then divided by gains ordered to match ``wfms.sensors``. S1/S2
    searches are independent. Both returned tables include the event identifier,
    per-type peak numbers, sample times and calibrated sensor rates in pe/us
    when gains are defined in ADC * us / pe.
    ``sampling_time`` is the input file's sample interval in microseconds.
    """
    wfms = remove_baseline(wfms)
    wfms = invert_polarity(wfms)
    wfms = calibrate(wfms, gains)

    s1s = find_peaks(wfms, config.s1, sampling_time)
    s2s = find_peaks(wfms, config.s2, sampling_time)

    s1s.insert(0, "event", event)
    s2s.insert(0, "event", event)

    return s1s, s2s


def calibration_gains(detector: Detector, run_number: int, sensors: np.ndarray) -> np.ndarray:
    """
    Return run-valid gains ordered according to the requested sensor IDs.

    ``SensorCalibration.SensorID`` is the lookup key, not electronics ChannelID.
    Gains are interpreted as ADC * us / pe and returned as a float array.
    A missing sensor raises ``ValueError``.
    """
    calibration = load_sensor_calibration(detector, run_number).set_index("SensorID")

    sensors = np.asarray(sensors)
    missing = sensors[~np.isin(sensors, calibration.index)]
    if missing.size:
        raise ValueError(f"Missing calibration gains for sensors {missing}")

    return calibration.loc[sensors, "Gain"].to_numpy(dtype=float)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="orco pmaps", description="Generate PMAPs from COLINA data")
    parser.add_argument("-i", "--input_file" , type=Path, required=True, help="input file")
    parser.add_argument("-o", "--output_file", type=Path, required=True, help="output file")
    parser.add_argument("-c", "--config_file", type=Path, required=True, help="configuration file")
    parser.add_argument("-r", "--run-number" , type=int , required=True, help="run number")
    parser.add_argument("-n", "--n-events"   , type=int , default =-1  , help="number of events to process")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)

    if args.n_events < -1:
        parser.error("--n-events must be -1 or non-negative")

    if args.input_file.resolve() == args.output_file.resolve():
        parser.error("input and output files must be different")

    config = PMapConfig.load(args.config_file)
    daq_config    = read_daq_config(args.input_file)
    detector      = Detector.get(daq_config["detector"])
    sampling_time = get_sampling_time(daq_config)
    mapping       = load_channel_mapping(detector, args.run_number)

    with tb.open_file(args.output_file, "w", filters=tb.Filters(complib="zlib", complevel=4)) as outfile:
        copy_daq_config(args.input_file, outfile)
        outfile.root._v_attrs.sampling_time_microseconds = sampling_time

        for i, data in enumerate(read_wfms(args.input_file, mapping, sampling_time)):
            if not args.quiet:
                print(f"\rProcessing event {i:>05}", end="", flush=True)

            gains    = calibration_gains(detector, args.run_number, data["wfms"].sensors)
            s1s, s2s = create_pmap(data["event"], data["wfms"], gains, config, sampling_time)

            events = pd.DataFrame(dict(event=[data["event"]]))
            df_writer(outfile, events, "PMaps", "events")
            df_writer(outfile, s1s, "PMaps", "S1")
            df_writer(outfile, s2s, "PMaps", "S2")

            if args.n_events >= 0 and i+1 >= args.n_events:
                break


    if not args.quiet:
        print(f"\rFinished processing {i+1} events" + " "*20)

if __name__ == "__main__":
    main()
