# ORCO: Offline Reconstruction for COlina

Two data transformers are available:

- `pmaps`: raw sensor waveforms to calibrated signal slices (aka, peak maps or PMAPs).
- `dsts`: PMAPs to data summary tables (DSTs) with combined information of each S1-S2 pair.

## Install with Nix

Nix must have the `nix-command` and `flakes` features enabled. Install the command
from GitHub; no local clone is required for installation:

```sh
nix profile install github:COLINA-exp/recolina#orco
```

This will install the `orco` command, which accepts a transformer name:

```sh
orco --help
orco pmaps --help
orco dsts --help
```

The flake currently supports `x86_64-linux`. It supplies Python and the runtime
dependencies, and bundles the SQLite database. The installed database is
read-only; changes require rebuilding the package. Example configurations are
installed under `share/orco/examples`. They are not loaded automatically and
are never used as defaults; both transformers still require an explicit `-c`.

## Run a reconstruction

Example config files can be obtained from the installation. Copy them and modify
them according to your needs. For instance, with the default Nix profile:

```sh
cp ~/.nix-profile/share/orco/examples/pmaps_mc.conf pmaps.conf
cp ~/.nix-profile/share/orco/examples/dsts_mc.conf dsts.conf
chmod u+w pmaps.conf dsts.conf
```

Physical settings require explicit units, for example `16 * nanosecond`,
`100 * pe` or `10 * pe / microsecond`. Bare numbers are accepted only for
dimensionless integer fields such as counts, hold-on and padding. See the
[configuration guide](docs/configuration.md#units) for the expected units.

To run the transformers:

```sh
orco pmaps -i waveforms.h5 -o pmaps.h5 -c pmaps.conf -r -1
orco dsts  -i pmaps.h5     -o dsts.h5  -c dsts.conf  -r -1
```

`-r -1` selects the bundled simulation calibration and geometry. For measured
data, supply the actual run number and appropriate configurations. Both stages
read the detector from `/Acquisition/config_json` (case-insensitive), which is
copied to each output. Both stages also read `effective.sample_period_ns` from
that JSON. Neither value comes from the reconstruction config. Use the same `-r`
for both stages; it is not checked against the acquisition's run number.

Both transformers require `-i`, `-o`, `-c` and `-r`. `-n N` limits processing to
the first `N` input events; the default `-n -1` processes all events. `--quiet`
suppresses progress messages.

Both transformers accept `-db FILE` (or `--database FILE`) to use a separate
read-only SQLite database. Without it, ORCO selects the bundled database using
the DAQ detector.
Explicit filenames need not match the detector name. Use matching calibration
and geometry data at both stages; see the [database guide](docs/database.md).

## Reference

- [Configuration](docs/configuration.md): required settings, units and cut
  boundaries.
- [File formats and reconstruction](docs/formats.md): input shapes, PMAP/DST
  schemas, peak pairing and current limitations.
- [Calibration database](docs/database.md): tables, run ranges, sensor IDs and
  read-only inspection.

## Development

The first time you enter your copy of the repo, allow `direnv` to read the
`.envrc` file (`direnv allow`), let it install everything and you are all set.

Runtime modules live in `orco/reco`, `orco/io`, `orco/database` and
`orco/internal`. Transformer entry points live in `orco/transformers`; each
exposes `main(argv=None)` and uses package-relative imports. The dispatcher
discovers transformer modules dynamically, excluding private and test modules.
When adding a runtime dependency, update `runtimePackages` in `flake.nix` so the
installed command and isolated test environment receive it too; adding it only
to the development shell is not sufficient.

## Tests

Run the test suite in the development environment:

```sh
pytest
```

`nix flake check` also runs the test suite in an isolated Nix build.

The test files are located next to their modules and named `*_test.py`. Tests use
pytest's `tmp_path` fixture for temporary files rather than modifying the bundled
database.

See [LICENSE](LICENSE) for the GNU General Public License, version 3.
