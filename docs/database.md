# Calibration database

ORCO bundles `orco/database/pColina.sqlite3`. By default, database files are
discovered in the installed `orco/database` directory by their `.sqlite3`
extension. Their filename stems must identify a `Detector` symbol, ignoring
case; the detector comes from DAQ JSON.

Both transformers accept `-db FILE` (or `--database FILE`) to override bundled
discovery.
Explicit files may have any filename or extension, must use the existing SQLite
schema, and are opened read-only. An invalid explicit file never falls back to
a bundled database. The selected path must be an existing regular file; SQLite
contents and required tables are checked when queried. There is no detector
metadata in the database schema to verify against DAQ JSON.

```sh
orco pmaps -i waveforms.h5 -o pmaps.h5 -c pmaps.conf -r 3 --database calibration.sqlite3
orco dsts -i pmaps.h5 -o dsts.h5 -c dsts.conf -r 3 --database calibration.sqlite3
```

The Nix package copies the database beside its Python loader. Readers use
SQLite's read-only mode; they do not modify calibration data. The installed
Nix-store copy is immutable. Use `--database` to work with a separate database
without reinstalling. To change bundled data, edit the source database carefully,
run the tests, and rebuild/reinstall the package.

## Tables and identifiers

All tables include `FirstRun` and nullable `LastRun` validity columns.

| Table | Remaining columns |
| --- | --- |
| `SensorCalibration` | `SensorID`, `Gain`, `GainUncertainty`, `Sigma1Pe`, `Sigma1PeUncertainty` |
| `ChannelMapping` | `SensorID`, `ChannelID` |
| `SensorPosition` | `SensorID`, `Plane`, `X`, `Y`, `Z`, `Type` |
| `SensorCount` | `NSensorsNear`, `NSensorsFar`, `NSensorTotal` |

`ChannelID` identifies an input electronics channel. `SensorID` identifies the
physical sensor and is used for calibrations, geometry and PMAP sensor columns.
Position coordinates are in millimeters; `Plane` is `NP` or `FP`. Gains are
integrated single-photoelectron areas in ADC * us / pe and divide the
baseline-subtracted, polarity-corrected ADC waveforms to give rates in pe/us.
Multiplying a calibrated sample sum by sampling time in microseconds gives pe.
The database stores bare numeric magnitudes, not unit-bearing quantities.
The Pint registry defines independent `adc` and `pe` dimensions, so the gain
unit can be expressed as `adc * microsecond / pe` when doing unit-aware arithmetic.
Uncertainty and
single-photoelectron sigma columns are loaded but not currently used by PMAP or
DST reconstruction. The count loader is available separately; transformers use
the mapping/position tables rather than `SensorCount` to select sensors.

The per-sensor tables use `(FirstRun, SensorID)` as their primary key;
`SensorCount` uses `FirstRun`. The current bundled gains are all unity. They are
initial constants, not a guarantee of measured, run-specific gain calibration.

## Run selection

A row is valid when:

```sql
FirstRun <= :run_number
AND (LastRun IS NULL OR :run_number <= LastRun)
```

Both endpoints are inclusive. `NULL` means an open-ended interval. `-1` is an
ordinary numeric run value, not a wildcard or a substitute for `NULL`.

The current bundled data has two validity ranges:

| Run range | Sensors | Mapping |
| --- | --- | --- |
| `[-1, -1]` | 25 NP + 100 FP = 125 total | Channels `0..124`; sensor IDs `1000*i + j` for `i=0..4`, `j=0..24`. |
| `[0, NULL]` | 4 NP + 0 FP = 4 total | Channels `1..4` map to sensor IDs `0..3`. |

Run `-1` is the convention for the bundled simulation geometry. Nonnegative run
numbers select the four-sensor data in this database. Future calibrations can
add other ranges. Loaders return all matching rows; they do not resolve overlaps
or choose the latest entry. Avoid overlapping ranges for the same sensor or
channel when maintaining these tables.

## Read with Python

In a checkout's development environment:

```sh
nix develop
python
```

```python
from pathlib import Path
from orco.database.io import get_database, load_sensor_calibration, load_sensor_position
from orco.internal.symbols import Detector

database_file = get_database(Detector.pColina)
# Or: get_database(Detector.pColina, Path("custom-calibration.sqlite3"))
calibration = load_sensor_calibration(database_file, -1)
positions = load_sensor_position(database_file, -1)
```

The loaders return pandas dataframes. Calibration, mapping and position loaders
cache results by database path/run in the current process. Resolve paths with
`get_database` before calling any loader. Treat cached dataframes as
read-only, or use `.copy()` before editing them. Restart the process or call the
relevant loader's `.cache_clear()` after changing a source database.

## Inspect with SQLite

From the checkout, open a read-only session:

```sh
nix shell nixpkgs#sqlite -c sqlite3 -readonly orco/database/pColina.sqlite3
```

Then inspect the schema and values:

```sql
.tables
.schema SensorCalibration
.headers on
.mode column
SELECT * FROM SensorCalibration WHERE FirstRun = -1 LIMIT 10;
PRAGMA integrity_check;
.quit
```

If `sqlite3` is already installed, invoke it directly with the same `-readonly`
option. Keep run validity, SensorIDs, mapping and geometry consistent when
maintaining the source database; run the tests before rebuilding.
