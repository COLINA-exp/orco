#!/usr/bin/env python3
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
from ..internal.symbols import Detector
from ..internal.symbols import all_symbols
from ..internal.types   import IndexRange
from ..internal.types   import Waveforms
from ..internal.units   import unit_scope
from ..internal.units   import units
from ..io.hdf5          import df_writer
from ..reco.waveforms   import calibrate
from ..reco.waveforms   import find_windows
from ..reco.waveforms   import invert_polarity
from ..reco.waveforms   import remove_baseline
from ..reco.waveforms   import slice_waveforms


@dataclass
class PeakConfig:
    min_samples: int
    max_samples: int
    min_time   : float
    max_time   : float
    threshold  : float
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
    sampling_time: float
    detector     : Detector

    def __post_init__(self):
        if self.sampling_time <= 0:
            raise ValueError("sampling time must be positive")

    @classmethod
    def load(cls, filename: Path) -> PMapConfig:
        scope = unit_scope()
        scope.update(all_symbols())
        with open(filename) as file:
            exec(file.read(), scope)

        s1_vars = { key[3:] : value for key, value in scope.items() if key.startswith("s1_") }
        s2_vars = { key[3:] : value for key, value in scope.items() if key.startswith("s2_") }

        s1_config = PeakConfig(**s1_vars)
        s2_config = PeakConfig(**s2_vars)

        return cls(s1_config, s2_config, scope["sampling_time"], scope["detector"])


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
    with tb.open_file(input_file) as infile:
        all_wfms = infile.root.Waveforms.adcs
        channels = infile.root.Waveforms.channel_id.read()
        event_id = infile.root.Waveforms.event_id.read()

        if len(all_wfms.shape) != 3:
            raise RuntimeError("Waveforms must have event, channel, and sample axes")
        if (all_wfms.shape[0], all_wfms.shape[1]) != (event_id.size, channels.size):
            raise RuntimeError("Invalid shape of waveform array")

        sensor_ids, keep = get_sensor_ids(channels, channel_mapping)
        sampling_time = sampling_time.to(units.microsecond).magnitude
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
    data = dict(time=peak.time, summed=peak.sum_sensors())
    for c, wf in zip(peak.sensors, peak.amplitudes):
        data[f"sensor_{c:>03}"] = wf
    return pd.DataFrame(data)


def empty_peak_df(sensors: np.ndarray) -> pd.DataFrame:
    # in order to keep the type of each column, we give it a dummy value that we
    # later remove
    data = dict(peak=[0], time=[0.], summed=[0.])
    for c in sensors:
        data[f"sensor_{c:>03}"] = [0.]

    return pd.DataFrame(data).iloc[:0]


def find_peaks(wfms         : Waveforms,
               config       : PeakConfig,
               sampling_time: float) -> pd.DataFrame:
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
                config: PMapConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    wfms = remove_baseline(wfms)
    wfms = invert_polarity(wfms)
    wfms = calibrate(wfms, gains)

    s1s = find_peaks(wfms, config.s1, config.sampling_time)
    s2s = find_peaks(wfms, config.s2, config.sampling_time)

    s1s.insert(0, "event", event)
    s2s.insert(0, "event", event)

    return s1s, s2s


def calibration_gains(detector: Detector, run_number: int, sensors: np.ndarray) -> np.ndarray:
    """
    Return gains ordered according to ``sensors``.
    """
    calibration = load_sensor_calibration(detector, run_number).set_index("ChannelID")

    sensors = np.asarray(sensors)
    missing = sensors[~np.isin(sensors, calibration.index)]
    if missing.size:
        raise ValueError(f"Missing calibration gains for sensors {missing}")

    return calibration.loc[sensors, "Gain"].to_numpy(dtype=float)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate PMAPs from COLINA data")
    parser.add_argument("-i", "--input_file" , type=Path, required=True, help="input file")
    parser.add_argument("-o", "--output_file", type=Path, required=True, help="output file")
    parser.add_argument("-c", "--config_file", type=Path, required=True, help="configuration file")
    parser.add_argument("-r", "--run-number" , type=int , required=True, help="run number")
    parser.add_argument("-n", "--n-events"   , type=int , default =-1  , help="number of events to process")
    args = parser.parse_args(argv)

    if args.n_events < -1:
        parser.error("--n-events must be -1 or non-negative")

    config = PMapConfig.load(args.config_file)
    mapping = load_channel_mapping(config.detector, args.run_number)
    with tb.open_file(args.output_file, "w", filters=tb.Filters(complib="zlib", complevel=4)) as outfile:
        for i, data in enumerate(read_wfms(args.input_file, mapping, config.sampling_time)):
            if args.n_events >= 0 and i >= args.n_events:
                break

            gains    = calibration_gains(config.detector, args.run_number, data["wfms"].sensors)
            s1s, s2s = create_pmap(data["event"], data["wfms"], gains, config)

            df_writer(outfile, s1s, "PMaps", "S1")
            df_writer(outfile, s2s, "PMaps", "S2")


if __name__ == "__main__":
    main()
