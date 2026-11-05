#!/usr/bin/env python3
"""
Generate point-like data-summary tables (DSTs) from PMAP files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib     import Path
from typing      import Any, Iterator, Mapping, Sequence

import numpy  as np
import pandas as pd
import tables as tb

from ..database.io      import load_sensor_position
from ..internal.symbols import Detector
from ..internal.symbols import all_symbols
from ..internal.types   import ClosedInterval
from ..internal.types   import Interval
from ..internal.units   import reduce_to_internal
from ..internal.units   import unit_scope
from ..io.hdf5          import df_writer


@dataclass
class PeakPropertiesConfig:
    count       : ClosedInterval
    width       :       Interval
    height      :       Interval
    integral    :       Interval
    multiplicity: ClosedInterval
    threshold   : float

    @classmethod
    def from_dict(cls, d) -> PeakPropertiesConfig:
        count        = ClosedInterval(d["min_count"       ], d["max_count"       ])
        width        =       Interval(d["min_width"       ], d["max_width"       ])
        height       =       Interval(d["min_height"      ], d["max_height"      ])
        integral     =       Interval(d["min_integral"    ], d["max_integral"    ])
        multiplicity = ClosedInterval(d["min_multiplicity"], d["max_multiplicity"])
        threshold    =                d["threshold"       ]
        return cls(count, width, height, integral, multiplicity, threshold)


@dataclass
class DstConfig:
    s1_props: PeakPropertiesConfig
    s2_props: PeakPropertiesConfig
    detector: Detector

    @classmethod
    def load(cls, filename: Path) -> DstConfig:
        scope = unit_scope()
        scope.update(all_symbols())
        with open(filename) as file:
            exec(file.read(), scope)

        s1_props = { key[3:] : reduce_to_internal(value) for key, value in scope.items() if key.startswith("s1_") }
        s2_props = { key[3:] : reduce_to_internal(value) for key, value in scope.items() if key.startswith("s2_") }

        return cls(PeakPropertiesConfig.from_dict(s1_props),
                   PeakPropertiesConfig.from_dict(s2_props),
                   scope["detector"])


def read_pmaps(input_file: Path) -> Iterator[Mapping[str, Any]]:
    """
    Yield the sampling interval, event numbers and their S1 and S2 peak tables.
    """
    with tb.open_file(input_file) as infile:
        sampling_time = infile.root._v_attrs.sampling_time_microseconds

        events = infile.root.PMaps.events.read()["event"]
        s1s    = pd.DataFrame(infile.root.PMaps.S1.read(), columns=infile.root.PMaps.S1.colnames)
        s2s    = pd.DataFrame(infile.root.PMaps.S2.read(), columns=infile.root.PMaps.S2.colnames)

    for event in events:
        this_evt = lambda df: df.event == event
        yield dict(event=event,
                   s1s=s1s.loc[this_evt],
                   s2s=s2s.loc[this_evt],
                   sampling_time=sampling_time)


def sensor_charges(peak: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the sensor identifiers and total charges of a peak.

    The charge of a sensor is the sum of its amplitudes over the peak samples.
    """
    sensors    = peak.loc[:, peak.columns.str.startswith("sensor_")]
    sensor_ids = sensors.columns.str.removeprefix("sensor_").astype(int).values
    charges    = sensors.values.sum(axis=0)
    return sensor_ids, charges


def sensor_multiplicity(peak      : pd.DataFrame,
                        threshold : float,
                        sensors_np: np.ndarray) -> int:
    """
    Number of near-plane sensors of ``peak`` with charge at or above ``threshold``.
    """
    sensor_ids, charges = sensor_charges(peak)
    in_np = np.isin(sensor_ids, sensors_np)
    return int(np.count_nonzero(charges[in_np] >= threshold))


def summarize_peak(peak     : pd.DataFrame,
                   threshold: float,
                   ids_np   : Sequence,
                   sampling_time: float) -> pd.Series:
    """
    Summarize a single peak into a series of selection properties.

    Peaks with no net charge are summarized to ``nan`` values so that they can
    still be matched to the other plane without contributing a position.
    """
    ok = np.nansum(peak.summed) > 0
    return pd.Series(dict(
        width        = len(peak) * sampling_time                    if ok else np.nan,
        height       = peak.summed.max()                            if ok else np.nan,
        time         = peak.time.loc[peak.summed.idxmax()]          if ok else np.nan,
        integral     = peak.summed.sum()                            if ok else np.nan,
        multiplicity = sensor_multiplicity(peak, threshold, ids_np) if ok else 0,
    ))


def blame_peaks(peaks : pd.DataFrame,
                config: PeakPropertiesConfig,
                ids_np: Sequence,
                sampling_time: float) -> pd.DataFrame:
    """
    Return a table of peak numbers rejected by ``config`` with their blame.

    The returned table has the columns ``peak`` and ``blame``; ``blame`` lists
    the comma-separated properties failing the selection. Peaks satisfying the
    selection simply do not appear in the table.
    """
    blame_records = []
    for pkno, peak in peaks.groupby("peak"):
        blame = []
        summary = summarize_peak(peak, config.threshold, ids_np, sampling_time)

        if summary.width        not in config.width       : blame.append("width")
        if summary.height       not in config.height      : blame.append("height")
        if summary.integral     not in config.integral    : blame.append("integral")
        if summary.multiplicity not in config.multiplicity: blame.append("multiplicity")

        if blame:
            blame_records.append(dict(peak=pkno, blame=", ".join(blame)))

    return pd.DataFrame(blame_records, columns=["peak", "blame"]).astype(dict(peak=int, blame=str))


def filter_peaks(peaks: pd.DataFrame, blame: pd.DataFrame) -> pd.DataFrame:
    """
    Return the peaks of ``peaks`` not listed in the ``blame`` table.
    """
    return peaks.loc[~peaks.peak.isin(blame.peak)]


def count_peaks(event   : int,
                s1s     : pd.DataFrame,
                s2s     : pd.DataFrame,
                s1_range: ClosedInterval,
                s2_range: ClosedInterval) -> pd.Series:
    ns1 = s1s.peak.nunique()
    ns2 = s2s.peak.nunique()
    blame = []
    if ns1 not in s1_range: blame.append("S1")
    if ns2 not in s2_range: blame.append("S2")
    return pd.Series(dict(
        event    = event,
        s1_count = ns1,
        s2_count = ns2,
        blame    = ", ".join(blame),
    ))


def barycenter(x        : np.ndarray,
               y        : np.ndarray,
               q        : np.ndarray,
               threshold: float) -> tuple[float, float]:
    """
    Charge-weighted barycenter of sensors with charge at or above ``threshold``.
    """
    sel   = q >= threshold
    x, y  = x[sel], y[sel]
    q     = q[sel]
    total = q.sum()
    if total == 0:
        return np.nan, np.nan
    return float(np.sum(x * q) / total), float(np.sum(y * q) / total)


def peak_barycenter(peak      : pd.DataFrame,
                    sensors_np: pd.DataFrame,
                    threshold : float) -> tuple[float, float]:
    """
    Barycenter of ``peak`` using the positions of its near-plane sensors.
    """
    sensor_ids, charges = sensor_charges(peak)
    keep   = np.isin(sensor_ids, sensors_np.SensorID.values)
    ids    = sensor_ids[keep]
    q      = charges[keep]
    pos    = sensors_np.set_index("SensorID").loc[ids]
    return barycenter(pos.X.to_numpy(), pos.Y.to_numpy(), q, threshold)


def empty_peak(sensors: pd.DataFrame) -> pd.DataFrame:
    """
    Single dummy peak for planes without peaks, keeping the column schema.

    The dummy row carries ``peak = -1`` and ``nan`` amplitudes so that the
    plane still contributes a row to the DST with undefined properties.
    """
    per_sensor_data = { f"sensor_{sensor:>03}": np.nan for sensor in sensors.SensorID }
    return pd.DataFrame(dict(
        peak   = -1,
        time   = -1,
        summed = np.nan,
        **per_sensor_data,
    ), index=[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate point-like DSTs from PMAPs")
    parser.add_argument("-i", "--input_file" , type=Path, required=True, help="input PMAP file")
    parser.add_argument("-o", "--output_file", type=Path, required=True, help="output file")
    parser.add_argument("-c", "--config_file", type=Path, required=True, help="configuration file")
    parser.add_argument("-r", "--run-number" , type=int , required=True, help="run number")
    parser.add_argument("-n", "--n-events"   , type=int , default =-1  , help="number of events to process")
    args = parser.parse_args(argv)

    if args.n_events < -1:
        parser.error("--n-events must be -1 or non-negative")

    config     = DstConfig.load(args.config_file)
    sensors    = load_sensor_position(config.detector, args.run_number)
    sensors_np = sensors.loc[sensors.Plane == "NP"]
    ids_np     = sensors_np.SensorID.values

    with tb.open_file(args.output_file, "w", filters=tb.Filters(complib="zlib", complevel=4)) as outfile:
        for i, data in enumerate(read_pmaps(args.input_file)):
            if args.n_events >= 0 and i >= args.n_events:
                break

            blame_s1s = blame_peaks(data["s1s"], config.s1_props, ids_np, data["sampling_time"])
            blame_s2s = blame_peaks(data["s2s"], config.s2_props, ids_np, data["sampling_time"])
            blame_s1s.insert(0, "event", data["event"])
            blame_s2s.insert(0, "event", data["event"])
            df_writer(outfile, blame_s1s, "DST", "S1Blame")
            df_writer(outfile, blame_s2s, "DST", "S2Blame")

            s1s = filter_peaks(data["s1s"], blame_s1s)
            s2s = filter_peaks(data["s2s"], blame_s2s)

            peak_counts = count_peaks(data["event"], s1s, s2s, config.s1_props.count, config.s2_props.count)
            df_writer(outfile, pd.DataFrame([peak_counts.to_dict()]), "DST", "EventPeakCount")

            if peak_counts.blame != "": continue

            if s1s.empty: s1s = empty_peak(sensors)
            if s2s.empty: s2s = empty_peak(sensors)

            rows = []
            for s1no, s1 in s1s.groupby("peak"):
                for s2no, s2 in s2s.groupby("peak"):
                    x, y = peak_barycenter(s2, sensors_np, config.s2_props.threshold)

                    s1_summary = summarize_peak(s1, config.s1_props.threshold, ids_np, data["sampling_time"]).add_prefix("s1_")
                    s2_summary = summarize_peak(s2, config.s2_props.threshold, ids_np, data["sampling_time"]).add_prefix("s2_")

                    row = pd.concat([s1_summary.to_frame().T, s2_summary.to_frame().T], axis=1)
                    row.insert(0, "event", data["event"])
                    row.insert(1, "s1no" , s1no)
                    row.insert(2, "s2no" , s2no)
                    row["drift_time"] = row.s2_time - row.s1_time
                    row["x"] = x
                    row["y"] = y
                    rows.append(row)

            if rows:
                dsts = pd.concat(rows, ignore_index=True)
                df_writer(outfile, dsts, "DST", "Pointlike")


if __name__ == "__main__":
    main()
