# Configuration

Both transformers require a configuration file through `-c`. Files are executed
as Python code with detector symbols and Pint unit names available directly.
Only use trusted files: configuration evaluation is not sandboxed. For example:

```python
s1_min_time = 1 * microsecond
```

Both transformers read the top-level `detector` string from the input's
`/Acquisition/config_json`, not from the reconstruction configuration. Names
are case-insensitive: `"pCOLINA"` resolves to `Detector.pColina`. `Colina` is
also recognized, but no database for it is currently bundled. Missing detector
metadata or unknown names fail before opening the output file. There is no
configuration fallback and no fallback for other required configuration keys.

## Installed examples

The Nix package ships the `.conf` files from `config/` under
`share/orco/examples`, normally `~/.nix-profile/share/orco/examples` after a
profile installation. They are read-only reference files, not active
configuration: neither the dispatcher nor the transformers discover, load or
use them unless you explicitly pass a file with `-c`.

Copy an example to your working directory, make the copy writable, and review
its settings before using it. See the [README](../README.md#run-a-reconstruction)
for commands. No example is selected when `-c` is omitted; the CLI reports a
missing required argument.

## Units

Quantities are reduced to ORCO's internal units when the configuration is loaded.
Time is stored in **microseconds** and length in **millimeters**. The remaining
SI base dimensions use kilogram, ampere, kelvin, mole and candela. Physical config
fields require explicit quantities, even for zero values. Dimensionless ratios
are converted to their
dimensionless magnitude, including any scale factor.

Photoelectrons have their own independent dimension, with `photoelectron` as
the internal base unit and `pe` as an alias. They are neither dimensionless
counts nor coulombs. Charges reduce to numeric pe and rates to numeric pe/us:

```python
s1_min_integral = 100 * pe
s1_min_height = 10 * pe / microsecond
s1_threshold = 1 * photoelectron
```

Ratios such as `pe / pe` are dimensionless. Pint rejects conversion from pe to
an unrelated dimension such as coulombs. Unit-bearing arithmetic can therefore
distinguish these quantities before reduction. The conversion helper still
returns plain numbers: units are not carried into waveform arrays or stored
data. Bare numeric values are rejected for physical configuration fields.

ADC amplitude has a separate independent dimension with internal base `adc`.
It cannot be converted directly to pe, coulombs or dimensionless counts. Gain
expressions can use `adc * microsecond / pe`; dividing an amplitude in adc by
such a gain gives pe/us, and multiplying that rate by time gives pe. For example,
`20 * adc * nanosecond / pe` reduces to `0.02` in internal gain units.
Ratios such as `adc / adc` are dimensionless.

Unit names, including `adc`, `pe` and `photoelectron`, need no import or `units.`
prefix.
For example, `200 * ns` becomes
`0.2` microseconds, while `1 * microsecond` becomes `1`. Counts, hold-on and
padding are dimensionless integers, not time durations. Fractions and booleans
are rejected for those fields. A dimensionless Pint quantity is accepted only
if its reduced magnitude is an integer.

Config loaders validate each field against its expected unit before reducing
it. For example, `s1_min_height = 5 * pe` is invalid because height requires a
rate; use `5 * pe / microsecond`. A bare `5` is also invalid. Use `0 * microsecond`
for a zero time bound, not `0`. Errors name the field and expected unit.
Configurations are executed independently for each transformer. The general
`reduce_to_internal` helper still supports arbitrary registered dimensions;
loaders use the annotation-driven reader in `orco/internal/config.py`, which
calls the stricter `config_value(value, expected_unit, name=...)` helper.
The registry, unit aliases and general conversion remain in
`orco/internal/units.py`. Transformer-specific config dataclasses stay beside
their transformer code.

Expected units are declared beside dataclass fields, not in separate unit maps.
`Time`, `Amplitude` and `Charge` are shared `Annotated[float, unit]` aliases for
microseconds, pe/us and pe respectively. `Amplitude` describes a calibrated
waveform's instantaneous rate, not its integrated charge. For example:

```python
@dataclass
class ExampleConfig:
    min_time: Time
    threshold: Amplitude
    padding: int
```

The shared `config_from_scope` reader resolves these annotations with metadata
retained. An `Annotated[Interval, unit]` or `Annotated[ClosedInterval, unit]`
field applies the same unit to its external `min_` and `max_` bounds. Plain `int`
fields are dimensionless counts. S1/S2 prefixes remain part of the file syntax
and error messages. Unannotated float fields are rejected, so adding a physical
field cannot silently bypass validation.

`Annotated` itself performs no runtime checking and introduces no quantity
objects into the numerical core. Validation belongs to config loading; direct
dataclass construction continues to accept already converted internal values.

## PMAP settings

See [pmaps_mc.conf](../config/pmaps_mc.conf) and
[pmaps_run_8.conf](../config/pmaps_run_8.conf) for complete examples.

The acquisition JSON's `detector` selects the database. The sampling interval
comes from the numeric
`effective.sample_period_ns` field in the input's `/Acquisition/config_json`,
in nanoseconds. PMAPs convert it to internal microseconds and assign sample zero
time zero. There is no waveform-attribute or config fallback or override, and
`sampling_time` is no longer a config field. Missing metadata is rejected before
opening the output file; the acquisition is responsible for providing a valid
sample period. Every following key is required twice, with `s1_` and `s2_`
prefixes:

| Suffix | Meaning |
| --- | --- |
| `min_samples`, `max_samples` | Inclusive limits on the unpadded window length; minimum is at least 1. |
| `min_time`, `max_time` | Inclusive limits containing the first and last unpadded sample times. |
| `threshold` | Minimum summed calibrated rate in pe/us for a sample to cross threshold. Equality counts. |
| `holdon` | Maximum number of below-threshold samples between crossings that are merged. Must be nonnegative. |
| `padding` | Nonnegative number of samples added at both ends after window selection, clipped to waveform bounds. |

For example, `s1_holdon = 5` joins crossings separated by up to five
below-threshold samples. A peak is represented by a half-open index range
`[first, last + 1)`, so slicing includes the last crossing sample.

S1 and S2 searches are independent and use the sum of **all retained sensors**.
Padding can make neighboring peaks overlap; overlapping samples are not merged
or deduplicated. DST properties are computed from the stored, padded samples,
not the original unpadded window.

## DST settings

See [dsts_mc.conf](../config/dsts_mc.conf) for a complete example. The detector
and `effective.sample_period_ns` come from the acquisition JSON copied into
the PMAP file, not from DST configuration or the PMAP root attribute. The shared
unit helper converts the JSON period from nanoseconds to internal microseconds.
Every following key is required with both `s1_` and `s2_` prefixes:

| Suffix | Meaning and boundary convention |
| --- | --- |
| `min_count`, `max_count` | Inclusive allowed number of selected peaks per event. |
| `min_width`, `max_width` | Half-open `[min, max)` range of stored sample count times sampling time. |
| `min_height`, `max_height` | Half-open range of the maximum `summed` rate, in pe/us. |
| `min_integral`, `max_integral` | Half-open range of `sum(summed) * sampling_time`, in pe. |
| `min_multiplicity`, `max_multiplicity` | Inclusive range of the number of sensors reaching the charge threshold. |
| `threshold` | Inclusive per-sensor integrated-charge threshold in pe for multiplicity; also used for the S2 barycenter. |

Gains are defined in ADC * us / pe, so PMAP amplitudes are rates in pe/us.
Both peak integral and per-sensor charges multiply their sample sums by sampling
time in microseconds, yielding pe. S1 multiplicity uses far-plane (`FP`) sensors;
S2 multiplicity and position use near-plane (`NP`) sensors. Height and integral
use the stored waveform sum across all sensors, including padding.

Integral bounds and DST charge thresholds must explicitly include `pe`; height
bounds and PMAP thresholds must include `pe / microsecond` or a compatible rate.
The examples retain their numeric cut values, now with explicit units. Cuts are
not automatically rescaled to preserve older sample-sum selections.
To preserve an old sample-sum selection at a particular sampling interval,
multiply its integral bounds and charge thresholds by that interval in
microseconds. For example, an old threshold of `1` at 16 ns corresponds to
`0.016` pe. PMAP thresholds and DST height cuts remain rates in pe/us.

Individual peaks are filtered first. Event-count cuts are then applied to the
remaining peaks. Allowing zero peaks permits dummy peaks with undefined
properties in the output; it does not make a missing peak pass individual cuts.
See [the DST format](formats.md#dst-output) for pairing and dummy-row semantics.

Choose compatible cuts across the two stages, accounting for sampling time and
padding. The example files are editable templates, not validated selections for
every run.
