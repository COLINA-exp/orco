#!/usr/bin/env python3
"""
Generate point-like data-summary tables (DSTs) from PMAP files.
"""

from __future__ import annotations

from argparse    import ArgumentParser
from dataclasses import dataclass
from pathlib     import Path

import numpy  as np
import pandas as pd
import tables as tb
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from ..database.io      import load_sensor_position
from ..internal.symbols import Detector
from ..internal.symbols import all_symbols
from ..internal.types   import ClosedInterval
from ..internal.types   import Interval
from ..internal.units   import reduce_to_internal
from ..internal.units   import unit_scope
from ..io.hdf5          import df_writer
from ..io.structures    import dst_types

from typing import Any
from typing import Iterator
from typing import Mapping
from typing import Sequence



@dataclass
class PeakPropertiesConfig:
    """
    Peak-property and event-count limits used by the DST selection.

    Attributes:
        count: Inclusive range for the number of selected peaks per event.
        width: Half-open range for a peak's duration, in microseconds.
        height: Half-open range for the peak's maximum summed amplitude.
        integral: Half-open range for the sum of the peak's summed amplitudes.
        multiplicity: Inclusive range for the number of near-plane sensors
            whose charge reaches ``threshold``.
        threshold: Inclusive minimum per-sensor charge for multiplicity.
    """

    count       : ClosedInterval
    width       :       Interval
    height      :       Interval
    integral    :       Interval
    multiplicity: ClosedInterval
    threshold   : float

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PeakPropertiesConfig:
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
        """
        Load a DST configuration and reduce quantities to internal units.

        The config file must define ``detector`` and the `PeakPropertiesConfig`
        fields with the corresponding ``s1_``/``s2_`` prefixes.

        Args:
            filename: Python config file defining the DST settings.

        Returns:
            The detector and the parsed S1/S2 selection limits.
        """
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
    Yield each PMAP event with its S1 and S2 peak tables.

    The sampling interval is read from the root ``sampling_time_microseconds``
    HDF5 attribute written by the PMAP transformer. Peak tables retain their
    stored columns, including ``event`` and ``peak``.

    Args:
        input_file: PMAP HDF5 file to read.

    Yields:
        One mapping per event with ``event``, ``s1s``, ``s2s`` and
        ``sampling_time`` (in microseconds).
    """
    with tb.open_file(input_file) as infile:
        sampling_time = infile.root._v_attrs.sampling_time_microseconds

        events = infile.root.PMaps.events.read()["event"]
        s1s    = pd.DataFrame(infile.root.PMaps.S1.read(), columns=infile.root.PMaps.S1.colnames)
        s2s    = pd.DataFrame(infile.root.PMaps.S2.read(), columns=infile.root.PMaps.S2.colnames)

    for event in events:
        this_evt = lambda df: df.event == event
        yield dict(event         = event,
                   s1s           = s1s.loc[this_evt],
                   s2s           = s2s.loc[this_evt],
                   sampling_time = sampling_time)


def sensor_charges(peak: pd.DataFrame) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Return sensor identifiers and each sensor's total charge for one peak.

    Sensor amplitudes are read from columns named ``sensor_<id>`` and summed
    over all rows in ``peak``.

    Args:
        peak: Peak rows with one ``sensor_<id>`` column per sensor.

    Returns:
        A pair of arrays containing the sensor identifiers and their summed
        charges, in matching column order.
    """
    sensors    = peak.loc[:, peak.columns.str.startswith("sensor_")]
    sensor_ids = sensors.columns.str.removeprefix("sensor_").to_numpy(dtype=np.int64)
    charges    = sensors.to_numpy(dtype=np.float64).sum(axis=0)
    return sensor_ids, charges


def sensor_multiplicity(peak     : pd.DataFrame,
                        threshold: float,
                        sensors  : ArrayLike) -> int:
    """
    Count sensors whose total peak charge reaches ``threshold``.

    Args:
        peak: Rows of one peak, including sensor-amplitude columns.
        threshold: Inclusive minimum charge for a sensor to count.
        sensors: Sensor to consider. Tipically either near-plane (S2) or far-plane (S1).

    Returns:
        Number of near-plane sensors whose charge is at least ``threshold``.
    """
    sensor_ids, charges = sensor_charges(peak)
    sensor_selection    = np.isin(sensor_ids, sensors)
    return int(np.count_nonzero(charges[sensor_selection] >= threshold))


def summarize_peak(peak         : pd.DataFrame,
                   threshold    : float,
                   sensor_ids   : ArrayLike,
                   sampling_time: float) -> pd.Series:
    """
    Calculate selection properties for one peak.

    - width: the number of samples times ``sampling_time``
    - height the maximum summed waveform amplitude
    - integral: the unweighted sum of that waveform over samples (it is not multiplied by ``sampling_time``)
    - multiplicity counts near-plane sensors at or above the charge threshold.

    Peaks with no positive net charge receive ``nan`` for numeric shape
    properties and zero multiplicity, allowing them to be paired without
    contributing a position.

    Args:
        peak: Rows belonging to a single peak, with ``time``, ``summed`` and
            ``sensor_<id>`` columns.
        threshold: Inclusive per-sensor charge threshold for multiplicity.
        ids_np: Near-plane sensor identifiers.
        sampling_time: Duration of one sample, in microseconds.

    Returns:
        A Series with ``width``, ``height``, ``time``, ``integral`` and
        ``multiplicity`` values.
    """
    ok = np.nansum(peak.summed) > 0
    return pd.Series(dict(
        width        = len(peak) * sampling_time                        if ok else np.nan,
        height       = peak.summed.max()                                if ok else np.nan,
        time         = peak.time.loc[peak.summed.idxmax()]              if ok else np.nan,
        integral     = peak.summed.sum()                                if ok else np.nan,
        multiplicity = sensor_multiplicity(peak, threshold, sensor_ids) if ok else 0,
    ))


def blame_peaks(peaks        : pd.DataFrame,
                config       : PeakPropertiesConfig,
                sensor_ids   : ArrayLike,
                sampling_time: float) -> pd.DataFrame:
    """
    Return a table of peak numbers rejected by ``config`` with their blame.

    The returned table has the columns ``peak`` and ``blame``; ``blame`` lists
    the comma-separated properties failing the selection. Peaks satisfying the
    selection simply do not appear in the table. Width, height and integral
    limits use half-open intervals; multiplicity uses a closed interval.

    Args:
        peaks: Peak rows with ``peak``, ``time``, ``summed`` and sensor
            amplitude columns.
        config: S1 or S2 selection limits to apply.
        ids_np: Near-plane sensor identifiers used for multiplicity.
        sampling_time: Duration of one sample, in microseconds.

    Returns:
        A dataframe with integer ``peak`` and string ``blame`` columns. It has
        the same schema when no peaks are rejected.
    """
    blame_records = []
    for pkno, peak in peaks.groupby("peak"):
        blame = []
        summary = summarize_peak(peak, config.threshold, sensor_ids, sampling_time)
        if summary.width        not in config.width       : blame.append("width")
        if summary.height       not in config.height      : blame.append("height")
        if summary.integral     not in config.integral    : blame.append("integral")
        if summary.multiplicity not in config.multiplicity: blame.append("multiplicity")

        if blame:
            blame_records.append(dict(peak=pkno, blame=", ".join(blame)))

    return pd.DataFrame(blame_records, columns=["peak", "blame"]).astype(dict(peak=int, blame=str))


def filter_peaks(peaks: pd.DataFrame, blame: pd.DataFrame) -> pd.DataFrame:
    """
    Return peak rows whose peak number does not appear in ``blame``; i.e. they
    satisfy all selection criteria.

    Args:
        peaks: Peak rows containing a ``peak`` column.
        blame: Rejected peak records containing a ``peak`` column.

    Returns:
        The selected rows from ``peaks``, preserving their original index and
        columns.
    """
    return peaks.loc[~peaks.peak.isin(blame.peak)]


def count_peaks(s1s     : pd.DataFrame,
                s2s     : pd.DataFrame,
                s1_range: ClosedInterval,
                s2_range: ClosedInterval) -> pd.Series:
    """
    Count selected S1 and S2 peaks and report count-range failures.

    Counts are the number of distinct peak identifiers. Each range is closed,
    so both configured endpoints are accepted.

    Args:
        s1s: Selected S1 peak rows.
        s2s: Selected S2 peak rows.
        s1_range: Allowed inclusive range of S1 peak counts.
        s2_range: Allowed inclusive range of S2 peak counts.

    Returns:
        A Series with ``event``, ``s1_count``, ``s2_count`` and a comma-joined
        ``blame`` string (empty when both counts are allowed).
    """
    ns1 = s1s.peak.nunique()
    ns2 = s2s.peak.nunique()
    blame = []
    if ns1 not in s1_range: blame.append("S1")
    if ns2 not in s2_range: blame.append("S2")
    return pd.Series(dict(
        s1_count = ns1,
        s2_count = ns2,
        blame    = ", ".join(blame),
    ))


def barycenter(x        : NDArray[np.float64],
               y        : NDArray[np.float64],
               q        : NDArray[np.float64],
               threshold: float) -> tuple[float, float]:
    """
    Calculate the charge-weighted position of sensors reaching a threshold.

    Args:
        x: One-dimensional sensor x coordinates.
        y: One-dimensional sensor y coordinates.
        q: Corresponding sensor charges.
        threshold: Inclusive minimum charge for a sensor to contribute.

    Returns:
        The weighted ``(x, y)`` position. Returns ``(nan, nan)`` when no
        sensor reaches the threshold or the selected total charge is zero.
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
    Calculate a peak's near-plane position from its sensor charges.

    Args:
        peak: Rows of one peak, with ``sensor_<id>`` amplitude columns.
        sensors_np: Near-plane sensor positions with ``SensorID``, ``X`` and
            ``Y`` columns.
        threshold: Inclusive minimum total charge for a sensor to contribute.

    Returns:
        Charge-weighted ``(x, y)`` coordinates, or ``(nan, nan)`` if no sensor
        reaches the threshold.
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

    Args:
        sensors: Sensor table containing a ``SensorID`` column.

    Returns:
        One row with ``peak`` set to -1, ``time`` set to -1, and ``summed`` and
        per-sensor amplitudes set to ``nan``.
    """
    per_sensor_data = { f"sensor_{sensor:>03}": np.nan for sensor in sensors.SensorID }
    return pd.DataFrame(dict(
        peak   = -1,
        time   = -1,
        summed = np.nan,
        **per_sensor_data,
    ), index=[0])


def main(argv: Sequence[str] | None = None) -> None:
    """
    Run the DST transformer.

    Reads PMAP events, applies the configured S1/S2 peak selections, and
    writes blame, event-count and point-like tables to an HDF5 output file.

    Args:
        argv: Optional command-line arguments, excluding the program name.
            When omitted, arguments are read from the process command line.
    """
    parser = ArgumentParser(description="Generate point-like DSTs from PMAPs")
    parser.add_argument("-i", "--input_file" , type=Path, required=True, help="input PMAP file")
    parser.add_argument("-o", "--output_file", type=Path, required=True, help="output file")
    parser.add_argument("-c", "--config_file", type=Path, required=True, help="configuration file")
    parser.add_argument("-r", "--run-number" , type=int , required=True, help="run number")
    parser.add_argument("-n", "--n-events"   , type=int , default =-1  , help="number of events to process")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.n_events < -1:
        parser.error("--n-events must be -1 or non-negative")

    config     = DstConfig.load(args.config_file)
    sensors    = load_sensor_position(config.detector, args.run_number)
    sensors_np = sensors.loc[sensors.Plane == "NP"]
    sensors_fp = sensors.loc[sensors.Plane == "FP"]
    ids_np     = sensors_np.SensorID.to_numpy()
    ids_fp     = sensors_fp.SensorID.to_numpy()

    with tb.open_file(args.output_file, "w", filters=tb.Filters(complib="zlib", complevel=4)) as outfile:
        for i, data in enumerate(read_pmaps(args.input_file)):
            if not args.quiet:
                print(f"\rProcessing event {i:>05}", end="", flush=True)

            blame_s1s = blame_peaks(data["s1s"], config.s1_props, ids_fp, data["sampling_time"])
            blame_s2s = blame_peaks(data["s2s"], config.s2_props, ids_np, data["sampling_time"])
            blame_s1s.insert(0, "event", data["event"])
            blame_s2s.insert(0, "event", data["event"])
            df_writer(outfile, blame_s1s, "DST", "S1Blame")
            df_writer(outfile, blame_s2s, "DST", "S2Blame")

            s1s = filter_peaks(data["s1s"], blame_s1s)
            s2s = filter_peaks(data["s2s"], blame_s2s)

            peak_counts = count_peaks(s1s, s2s, config.s1_props.count, config.s2_props.count)
            peak_counts = pd.concat([pd.Series([data["event"]], index=["event"]), peak_counts])
            df_writer(outfile, pd.DataFrame([peak_counts.to_dict()]), "DST", "EventPeakCount")

            if peak_counts.blame != "": continue

            if s1s.empty: s1s = empty_peak(sensors)
            if s2s.empty: s2s = empty_peak(sensors)

            rows = []
            for s1no, s1 in s1s.groupby("peak"):
                for s2no, s2 in s2s.groupby("peak"):
                    x, y = peak_barycenter(s2, sensors_np, config.s2_props.threshold)

                    s1_summary = summarize_peak(s1, config.s1_props.threshold, ids_fp, data["sampling_time"]).add_prefix("s1_")
                    s2_summary = summarize_peak(s2, config.s2_props.threshold, ids_np, data["sampling_time"]).add_prefix("s2_")

                    row = pd.concat([s1_summary.to_frame().T, s2_summary.to_frame().T], axis=1)
                    row.insert(0, "event", data["event"])
                    row.insert(1, "s1no" , s1no)
                    row.insert(2, "s2no" , s2no)
                    row["drift_time"] = row.s2_time - row.s1_time
                    row["x"] = x
                    row["y"] = y
                    rows.append(row.astype(dst_types))

            if rows:
                dsts = pd.concat(rows, ignore_index=True)
                df_writer(outfile, dsts, "DST", "Pointlike")

            if args.n_events >= 0 and i+1 >= args.n_events:
                break

    if not args.quiet:
        print(f"\rFinished processing {i+1} events" + " "*20)

if __name__ == "__main__":
    main()
