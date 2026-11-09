# Configuration

Both transformers require a configuration file through `-c`. Files are executed
as Python code with detector symbols and Pint unit names available directly.
Only use trusted files: configuration evaluation is not sandboxed. For example:

```python
detector = pColina
sampling_time = 16 * nanosecond
```

Use the unquoted symbol `pColina`, not the string `"pColina"`. `Colina` is also
defined as a detector symbol, but no database for it is currently bundled.
There are no fallback values for required configuration keys.

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
SI base dimensions use kilogram, ampere, kelvin, mole and candela. Plain numbers
are already in internal units. Dimensionless ratios are converted to their
dimensionless magnitude, including any scale factor.

Unit names need no import or `units.` prefix. For example, `200 * ns` becomes
`0.2` microseconds, while `1 * microsecond` becomes `1`. Counts, hold-on and
padding are sample counts, not time durations. Thresholds are numeric values in
the corresponding calibrated amplitude or charge scale.

Reduction does not validate a field's physical dimension: use time quantities
for time fields and integer counts for sample/count fields. Configurations are
executed independently for each transformer.

## PMAP settings

See [pmaps_mc.conf](../config/pmaps_mc.conf) and
[pmaps_run_8.conf](../config/pmaps_run_8.conf) for complete examples.

`detector` chooses the database. `sampling_time` is the positive interval between
raw waveform samples; sample zero is assigned time zero. Every following key is
required twice, with `s1_` and `s2_` prefixes:

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

See [dsts_mc.conf](../config/dsts_mc.conf) for a complete example. `detector` is
required; sampling time comes from the PMAP file, not from DST configuration.
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

Existing numeric integral bounds and DST charge thresholds are unchanged and
now mean pe; they are not automatically rescaled to preserve old selections.
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
