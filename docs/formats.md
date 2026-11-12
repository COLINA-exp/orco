# File formats and reconstruction

ORCO reads and writes HDF5 files through PyTables. Paths below are HDF5 node
paths, not filesystem paths. Transformer outputs use zlib compression.

## Raw waveform input

The PMAP transformer expects:

| Node | Shape | Contents |
| --- | --- | --- |
| `/Waveforms/adcs` | `(events, channels, samples)` | Raw waveform amplitudes. |
| `/Waveforms/channel_id` | `(channels,)` | Electronics channel identifiers in the second-axis order. |
| `/Waveforms/event_id` | `(events,)` | Event identifiers in the first-axis order. |

The JSON object stored in `/Acquisition/config_json` must contain the numeric
field `effective.sample_period_ns`, the finite positive interval between
samples in nanoseconds. For example, `16` means 16 ns and becomes `0.016`
microseconds internally. This JSON field is the source of sampling time;
the `/Waveforms` attribute `sample_period_ns` and configuration file are not
used. Files without the JSON field must have their acquisition metadata supplied
before PMAP reconstruction; no default sampling interval is assumed.

The top-level JSON `detector` string selects the database, ignoring case (for
example, `"pCOLINA"` selects `pColina`). The run number still comes from `-r`;
neither transformer uses a config-file detector or the JSON's run number.
The channel mapping for that detector/run translates channel IDs into sensor
IDs. Unknown input channels are dropped with a warning; missing channels
required by the mapping raise an error. PMAP columns use sensor IDs, not raw
channel IDs. Treat channel and event identifiers as unique within their axes.

For each event the transformer subtracts each sensor's median baseline, reverses
polarity, and divides by its gain. Raw pulses are therefore expected to be
negative-going. Gains are looked up by `SensorID` and ordered to match the
waveforms. Zero and NaN gains are rejected; infinite gains are allowed.

The gain convention is integrated single-photoelectron area in ADC * us / pe.
Dividing an ADC sample by gain produces a rate in pe/us, not charge per sample.
The PMAP sensor and `summed` columns retain those rates; integration in the DST
stage multiplies sample sums by the sampling interval in microseconds.

S1/S2 windows are found independently in the summed calibrated waveform, selected
by time and sample length, then padded. Time is `sample_index * sampling_time` in
microseconds. No timestamp array is required in the raw file.

## Acquisition configuration metadata

Both transformers copy the full `/Acquisition` group from input to output,
preserving its attributes and all nested groups, arrays and tables.
This carries acquisition metadata through raw waveform, PMAP and DST files
without modifying it. The copy is independent of event limits and peak
selections. Both transformers require `/Acquisition/config_json` with a known
`detector` and the JSON sampling period described above. Missing metadata is
not replaced by defaults.

`/Acquisition/config_json` is a scalar string array holding the full run
configuration as JSON. A standalone reader is available:

```python
from orco.io.hdf5 import read_daq_config

configuration = read_daq_config("waveforms.h5")
```

It returns a dictionary with nested JSON values, opens the file read-only, and
closes it before returning. The standard JSON decoder handles text encoding;
this is separate from the dataframe writer's ASCII-only string columns. Missing
nodes and malformed JSON raise the underlying PyTables/JSON errors.

The reader does not apply or validate settings. Both transformers extract
`detector` and `effective.sample_period_ns` from this dictionary. The shared
`orco.io.hdf5.get_sampling_time` helper converts the period to microseconds.
`/Acquisition/Configuration/config_json`, when present, is a different
acquisition-only subsection and is not used as a fallback.

## PMAP output

Sampling time is preserved in the copied DAQ JSON as `effective.sample_period_ns`.
No separate sampling-time root attribute is written. Both transformers convert
the JSON value to internal microseconds with the shared unit helper.

| Table | Columns |
| --- | --- |
| `/PMaps/events` | `event`: one row for each processed event, even if it has no peaks. |
| `/PMaps/S1` | `event`, `peak`, `time`, `summed`, `sensor_<id>` columns. |
| `/PMaps/S2` | Same schema as S1. |

Each peak table contains one row per stored sample. Peak numbering starts at zero
for each event and each peak type. `time` is in microseconds; `summed` is the sum
of all retained sensor amplitudes. Sensor column names have a minimum of three
digits, for example `sensor_000`, `sensor_024` and `sensor_1000`.

An event with no S1 or S2 peaks has no rows in that peak table, but remains in
`/PMaps/events`. Padding is included in these tables and can repeat samples in
neighboring peaks.

## DST output

| Table | Columns and purpose |
| --- | --- |
| `/DST/S1Blame`, `/DST/S2Blame` | `event`, `peak`, `blame`; one row per rejected peak. Reasons are comma-separated property names. |
| `/DST/EventPeakCount` | `event`, `s1_count`, `s2_count`, `blame`; counts after peak filtering. Empty blame means the event passes. Otherwise it lists `S1`, `S2`, or both. |
| `/DST/Pointlike` | One row per pair of selected S1/S2 peaks for events passing count cuts. |

Point-like rows contain:

- `event`, `s1no`, `s2no`: event and per-type peak identifiers.
- `s1_width`, `s1_height`, `s1_time`, `s1_integral`, `s1_multiplicity`, and the
  corresponding `s2_` fields.
- `drift_time = s2_time - s1_time`, in microseconds.
- `x`, `y`: the S2 near-plane charge-weighted barycenter, in millimeters.

Width is stored sample count times sampling time. Height is the maximum summed
amplitude; time is the sample time of that maximum (the first maximum on a tie).
Height is in pe/us. Integral is `sum(summed) * sampling_time`, in pe.
Per-sensor charges use the same integration rule. Multiplicity counts sensors
whose integrated charge is at least the configured threshold in pe, using FP
sensors for S1 and NP sensors for S2. Barycenter weights are integrated charges
of NP sensors reaching the S2 threshold. The common time factor cancels in the
weighted-position ratio, but matters for threshold selection. If selected total
charge is zero, both coordinates are NaN.

Every S1 is paired with every S2: there is no best-match selection, drift-velocity
conversion to `z`, or temporal ordering cut. Negative drift times are possible.
When zero peaks are allowed, a missing peak type contributes one dummy peak with
identifier `-1`, NaN width/height/time/integral and zero multiplicity. A missing
S2 also yields NaN position. Peaks with no positive net summed charge receive
the same undefined summary properties.

`Pointlike` uses the dtypes in [structures.py](../orco/io/structures.py): `uint32`
events, signed `int16` peak IDs, `uint16` multiplicities, and `float32` properties
and coordinates. Blame columns are fixed-width ASCII bytes (64 bytes for peak
blame, 32 for event-count blame); decode them as ASCII when reading with pandas.
There are no Pointlike rows for rejected events. The table itself is currently
created only when at least one row is written.

## Operational limits

Use unique output paths: existing outputs are replaced, and failed processing
can leave partial results. The CLI rejects matching resolved input/output paths
but does not detect distinct hard links to the same file.

Use `-n -1` for all events or a positive event limit. Currently `-n 0` processes
the first event. Empty input can fail in the final progress message unless
`--quiet` is used, and an empty output is not guaranteed to contain all tables.

The DST reader materializes both complete PMAP tables in memory and filters them
for each event. Large files may need substantial memory and processing time.
PMAPs carry sampling time and the original `/Acquisition` metadata, including
the detector used by both transformers. The transformers do not record the
reconstruction run number or configuration settings. Supply a consistent `-r`
at both stages yourself; it is not checked against the acquisition run number.

The generic dataframe writer fixes a table's schema on its first write. Numeric
compatibility uses NumPy's `same_kind` rule, which permits narrowing and does not
check value ranges. Oversized strings warn and are truncated. Validate ranges
and choose string widths when using this utility outside the transformers.
