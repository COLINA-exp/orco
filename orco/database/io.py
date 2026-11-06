"""
Functions for reading ORCO calibration data from SQLite databases.

Calibration rows are selected by an inclusive run-validity interval. A row is
valid when ``FirstRun <= run_number <= LastRun``. A null ``LastRun`` denotes an
open-ended interval, in which case only the lower bound is checked.

Databases are discovered beside this module using detector names and the
``.sqlite3`` extension, and are opened read-only. Calibrations and positions use
SensorID; ChannelID is only the electronics-channel key in ChannelMapping.

Calibration, mapping and position dataframes are cached by detector/run. Callers
should copy them before mutation. All valid rows are returned; overlapping run
ranges are not resolved by selecting a latest version. See ``docs/database.md``.
"""

from contextlib import closing
from pathlib   import Path
from functools import lru_cache

import sqlite3

import pandas as pd

from ..internal.symbols import Detector


DATABASE_EXTENSION = ".sqlite3"
DATABASE_DIRECTORY = Path(__file__).parent


def find_databases() -> dict[Detector, Path]:
    """
    Find SQLite databases in ``DATABASE_DIRECTORY`` beside this module.

    Files are recognized by the ``.sqlite3`` extension. Each filename stem
    must name a detector, ignoring case.

    Returns:
        Mapping between detectors and their database paths.

    Raises:
        ValueError: If a filename does not identify a known detector.
    """
    return { Detector.get(path.stem): path
             for path in sorted(DATABASE_DIRECTORY.glob(f"*{DATABASE_EXTENSION}")) }



def get_database(detector: Detector) -> Path:
    """
    Return the database path for ``detector``.

    Raises:
        FileNotFoundError: If no database is available for ``detector``.
    """
    databases = find_databases()
    if detector not in databases:
        raise FileNotFoundError(f"No SQLite database found for {detector.name} in {DATABASE_DIRECTORY}")
    return databases[detector]


@lru_cache()
def load_sensor_calibration(detector: Detector, run_number: int) -> pd.DataFrame:
    """
    Load sensor calibrations valid for a run into a dataframe.

    Rows are ordered by sensor identifier. The returned dataframe contains
    every column from ``SensorCalibration``.

    Args:
        detector: Detector whose calibration constants are required.
        run_number: Run for which calibration constants are required.

    Returns:
        Calibration rows satisfying ``FirstRun <= run_number`` and either
        ``run_number <= LastRun`` or ``LastRun IS NULL``.

    Raises:
        FileNotFoundError: If no database is available for ``detector``.
        ValueError: If a database filename does not identify a known detector.
        sqlite3.Error: If the selected database cannot be opened or queried.
    """
    database_file = get_database(detector)
    uri = f"{database_file.resolve().as_uri()}?mode=ro"
    query = """
        SELECT
            FirstRun,
            LastRun,
            SensorID,
            Gain,
            GainUncertainty,
            Sigma1Pe,
            Sigma1PeUncertainty
        FROM SensorCalibration
        WHERE FirstRun <= :run_number
          AND (LastRun IS NULL OR :run_number <= LastRun)
        ORDER BY SensorID
    """

    with closing(sqlite3.connect(uri, uri=True)) as connection:
        return pd.read_sql_query(query,
                                 connection,
                                 params={"run_number": run_number})


def load_sensor_count(detector: Detector, run_number: int) -> pd.DataFrame:
    """
    Load the sensor counts valid for a run into a dataframe.

    Columns are ``FirstRun``, ``LastRun``, ``NSensorsNear``, ``NSensorsFar`` and
    ``NSensorTotal``. Results are ordered by FirstRun and are not cached.

    Args:
        detector: Detector whose sensor counts are required.
        run_number: Run for which sensor counts are required.

    Returns:
        Rows satisfying ``FirstRun <= run_number`` and either
        ``run_number <= LastRun`` or ``LastRun IS NULL``.

    Raises:
        FileNotFoundError: If no database is available for ``detector``.
        ValueError: If a database filename does not identify a known detector.
        sqlite3.Error: If the selected database cannot be opened or queried.
    """
    database_file = get_database(detector)
    uri = f"{database_file.resolve().as_uri()}?mode=ro"
    query = """
        SELECT
            FirstRun,
            LastRun,
            NSensorsNear,
            NSensorsFar,
            NSensorTotal
        FROM SensorCount
        WHERE FirstRun <= :run_number
          AND (LastRun IS NULL OR :run_number <= LastRun)
        ORDER BY FirstRun
    """

    with closing(sqlite3.connect(uri, uri=True)) as connection:
        return pd.read_sql_query(query,
                                 connection,
                                 params={"run_number": run_number})


@lru_cache()
def load_channel_mapping(detector: Detector, run_number: int) -> pd.DataFrame:
    """
    Load the sensor-to-channel mappings valid for a run into a dataframe.

    Rows are ordered by channel identifier.

    Args:
        detector: Detector whose sensor-to-channel mappings are required.
        run_number: Run for which the mappings are required.

    Returns:
        Mapping rows satisfying ``FirstRun <= run_number`` and either
        ``run_number <= LastRun`` or ``LastRun IS NULL``.

    Raises:
        FileNotFoundError: If no database is available for ``detector``.
        ValueError: If a database filename does not identify a known detector.
        sqlite3.Error: If the selected database cannot be opened or queried.
    """
    database_file = get_database(detector)
    uri = f"{database_file.resolve().as_uri()}?mode=ro"
    query = """
        SELECT
            FirstRun,
            LastRun,
            SensorID,
            ChannelID
        FROM ChannelMapping
        WHERE FirstRun <= :run_number
          AND (LastRun IS NULL OR :run_number <= LastRun)
        ORDER BY ChannelID
    """

    with closing(sqlite3.connect(uri, uri=True)) as connection:
        return pd.read_sql_query(query,
                                 connection,
                                 params={"run_number": run_number})


@lru_cache()
def load_sensor_position(detector: Detector, run_number: int) -> pd.DataFrame:
    """
    Load the sensor positions valid for a run into a dataframe.

    Rows are ordered by sensor identifier. ``Plane`` is NP or FP, coordinates
    ``X``, ``Y``, ``Z`` are in millimeters, and ``Type`` identifies sensor type.

    Args:
        detector: Detector whose sensor positions are required.
        run_number: Run for which the positions are required.

    Returns:
        Position rows satisfying ``FirstRun <= run_number`` and either
        ``run_number <= LastRun`` or ``LastRun IS NULL``.

    Raises:
        FileNotFoundError: If no database is available for ``detector``.
        ValueError: If a database filename does not identify a known detector.
        sqlite3.Error: If the selected database cannot be opened or queried.
    """
    database_file = get_database(detector)
    uri = f"{database_file.resolve().as_uri()}?mode=ro"
    query = """
        SELECT
            FirstRun,
            LastRun,
            SensorID,
            Plane,
            X,
            Y,
            Z,
            Type
        FROM SensorPosition
        WHERE FirstRun <= :run_number
          AND (LastRun IS NULL OR :run_number <= LastRun)
        ORDER BY SensorID
    """

    with closing(sqlite3.connect(uri, uri=True)) as connection:
        return pd.read_sql_query(query,
                                 connection,
                                 params={"run_number": run_number})
