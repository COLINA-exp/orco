from contextlib import closing

import sqlite3

from numpy.testing import assert_array_equal
from pytest import raises

from ..internal.symbols import Detector
from . import io
from .io import find_databases
from .io import get_database
from .io import load_channel_mapping
from .io import load_sensor_calibration
from .io import load_sensor_count
from .io import load_sensor_position


def create_database(filename):
    with closing(sqlite3.connect(filename)) as connection:
        with connection:
            connection.execute("""
                CREATE TABLE SensorCalibration (
                    FirstRun             INT,
                    LastRun              INT,
                    ChannelID            INT,
                    Gain                 DOUBLE,
                    GainUncertainty      DOUBLE,
                    Sigma1Pe             DOUBLE,
                    Sigma1PeUncertainty  DOUBLE
                )
            """)
            connection.executemany("""
                INSERT INTO SensorCalibration VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ((-1,   -1, 3, 1.0, 0.1, 0.2, 0.3),
                  ( 0,   10, 2, 2.0, 0.1, 0.2, 0.3),
                  ( 5,    5, 1, 3.0, 0.1, 0.2, 0.3),
                  ( 0,    4, 0, 4.0, 0.1, 0.2, 0.3),
                  ( 0, None, 4, 5.0, 0.1, 0.2, 0.3)))

            connection.execute("""
                CREATE TABLE SensorCount (
                    FirstRun      INT,
                    LastRun       INT,
                    NSensorsNear  INT,
                    NSensorsFar   INT,
                    NSensorTotal  INT
                )
            """)
            connection.executemany("""
                INSERT INTO SensorCount VALUES (?, ?, ?, ?, ?)
            """, ((-1,   -1, 25, 100, 125),
                  ( 0,   10, 10,  20,  30),
                  (11, None,  4,   0,   4)))

            connection.execute("""
                CREATE TABLE ChannelMapping (
                    FirstRun   INT,
                    LastRun    INT,
                    SensorID   INT,
                    ChannelID  INT
                )
            """)
            connection.executemany("""
                INSERT INTO ChannelMapping VALUES (?, ?, ?, ?)
            """, ((-1,   -1,  10, 0),
                  ( 0,   10,  11, 1),
                  ( 5,    5,  12, 2),
                  ( 0, None,  13, 3)))

            connection.execute("""
                CREATE TABLE SensorPosition (
                    FirstRun  INT,
                    LastRun   INT,
                    SensorID  INT,
                    Plane     TEXT,
                    X         DOUBLE,
                    Y         DOUBLE,
                    Z         DOUBLE,
                    Type      TEXT
                )
            """)
            connection.executemany("""
                INSERT INTO SensorPosition VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, ((-1,   -1, 10, "FP", 1.0, 2.0, 3.0, "S13370"),
                  ( 0,   10, 11, "FP", 4.0, 5.0, 6.0, "S13370"),
                  ( 5,    5, 12, "FP", 7.0, 8.0, 9.0, "S13370"),
                  ( 0, None, 13, "FP", 1.5, 2.5, 3.5, "S13370")))


def test_load_sensor_calibration_selects_valid_run_intervals(tmp_path, monkeypatch):
    filename = tmp_path / "pColina.sqlite3"
    create_database(filename)
    monkeypatch.setattr(io, "DATABASE_DIRECTORY", tmp_path)

    calibration = load_sensor_calibration(Detector.pColina, 5)

    assert calibration.ChannelID.tolist() == [1, 2, 4]
    assert calibration.Gain     .tolist() == [3.0, 2.0, 5.0]


def test_load_sensor_calibration_keeps_only_open_interval_after_last_run(tmp_path, monkeypatch):
    filename = tmp_path / "pColina.sqlite3"
    create_database(filename)
    monkeypatch.setattr(io, "DATABASE_DIRECTORY", tmp_path)

    calibration = load_sensor_calibration(Detector.pColina, 11)

    assert calibration.ChannelID.tolist() == [4]


def test_load_sensor_calibration_reads_bundled_database():
    calibration = load_sensor_calibration(Detector.pColina, -1)

    assert len(calibration) == 125
    assert_array_equal(calibration.ChannelID, [n * 1000 + k for n in range(5) for k in range(25)])
    assert_array_equal(calibration.FirstRun, -1)
    assert_array_equal(calibration.LastRun , -1)
    assert_array_equal(calibration.Gain    ,  1)


def test_load_sensor_count_selects_valid_run_intervals(tmp_path, monkeypatch):
    filename = tmp_path / "pColina.sqlite3"
    create_database(filename)
    monkeypatch.setattr(io, "DATABASE_DIRECTORY", tmp_path)

    sensor_count = load_sensor_count(Detector.pColina, 5)

    assert sensor_count.NSensorsNear.tolist() == [10]
    assert sensor_count.NSensorsFar .tolist() == [20]
    assert sensor_count.NSensorTotal.tolist() == [30]


def test_load_sensor_count_keeps_open_interval_after_last_run(tmp_path, monkeypatch):
    filename = tmp_path / "pColina.sqlite3"
    create_database(filename)
    monkeypatch.setattr(io, "DATABASE_DIRECTORY", tmp_path)

    sensor_count = load_sensor_count(Detector.pColina, 11)

    assert sensor_count.NSensorsNear.tolist() == [4]
    assert sensor_count.NSensorsFar .tolist() == [0]
    assert sensor_count.NSensorTotal.tolist() == [4]


def test_load_sensor_count_reads_bundled_database():
    sensor_count = load_sensor_count(Detector.pColina, 0)

    assert len(sensor_count) == 1
    assert sensor_count.iloc[0].FirstRun     == 0
    assert sensor_count.iloc[0].NSensorsNear == 4
    assert sensor_count.iloc[0].NSensorsFar  == 0
    assert sensor_count.iloc[0].NSensorTotal == 4


def test_load_channel_mapping_selects_valid_run_intervals(tmp_path, monkeypatch):
    filename = tmp_path / "pColina.sqlite3"
    create_database(filename)
    monkeypatch.setattr(io, "DATABASE_DIRECTORY", tmp_path)

    mapping = load_channel_mapping(Detector.pColina, 5)

    assert mapping.ChannelID.tolist() == [1, 2, 3]
    assert mapping.SensorID .tolist() == [11, 12, 13]


def test_load_channel_mapping_keeps_open_interval_after_last_run(tmp_path, monkeypatch):
    filename = tmp_path / "pColina.sqlite3"
    create_database(filename)
    monkeypatch.setattr(io, "DATABASE_DIRECTORY", tmp_path)

    mapping = load_channel_mapping(Detector.pColina, 11)

    assert mapping.ChannelID.tolist() == [3]
    assert mapping.SensorID .tolist() == [13]


def test_load_channel_mapping_reads_bundled_database():
    mapping = load_channel_mapping(Detector.pColina, -1)

    assert len(mapping) == 125
    assert_array_equal(mapping.ChannelID, range(125))
    assert_array_equal(mapping.SensorID,
                       [n * 1000 + k for n in range(5) for k in range(25)])
    assert_array_equal(mapping.FirstRun, -1)
    assert_array_equal(mapping.LastRun , -1)


def test_load_sensor_position_selects_valid_run_intervals(tmp_path, monkeypatch):
    filename = tmp_path / "pColina.sqlite3"
    create_database(filename)
    monkeypatch.setattr(io, "DATABASE_DIRECTORY", tmp_path)

    position = load_sensor_position(Detector.pColina, 5)

    assert position.SensorID.tolist() == [11, 12, 13]
    assert position.X.tolist() == [4.0, 7.0, 1.5]
    assert position.Y.tolist() == [5.0, 8.0, 2.5]
    assert position.Z.tolist() == [6.0, 9.0, 3.5]


def test_load_sensor_position_keeps_open_interval_after_last_run(tmp_path, monkeypatch):
    filename = tmp_path / "pColina.sqlite3"
    create_database(filename)
    monkeypatch.setattr(io, "DATABASE_DIRECTORY", tmp_path)

    position = load_sensor_position(Detector.pColina, 11)

    assert position.SensorID.tolist() == [13]
    assert position.X.tolist() == [1.5]
    assert position.Y.tolist() == [2.5]
    assert position.Z.tolist() == [3.5]


def test_load_sensor_position_reads_bundled_database():
    position = load_sensor_position(Detector.pColina, -1)

    assert len(position) == 125
    assert_array_equal(position.SensorID,
                       [n * 1000 + k for n in range(5) for k in range(25)])
    assert_array_equal(position.FirstRun, -1)
    assert_array_equal(position.LastRun , -1)
    assert (position.Type == "S13370").all()

    position = position.set_index("SensorID")
    assert position.loc[0   , ["X", "Y", "Z"]].tolist() == [-12.4, -12.4, -13.5]
    assert position.loc[24  , ["X", "Y", "Z"]].tolist() == [ 12.4,  12.4, -13.5]
    assert position.loc[1000, ["X", "Y", "Z"]].tolist() == [ -3.1, -27.9, 128.5]
    assert position.loc[4000, ["X", "Y", "Z"]].tolist() == [ 27.9,   3.1, 128.5]
    assert position.loc[4024, ["X", "Y", "Z"]].tolist() == [  3.1,  27.9, 128.5]

    assert (position.loc[position.Z < 0, "Plane"] == "NP").all()
    assert (position.loc[position.Z > 0, "Plane"] == "FP").all()


def test_load_sensor_position_reads_run_zero_positions():
    position = load_sensor_position(Detector.pColina, 0)

    assert_array_equal(position.SensorID, range(4))
    assert_array_equal(position.FirstRun, 0)
    assert (position.LastRun.isna()).all()
    assert (position.Type == "S13370").all()

    position = position.set_index("SensorID")
    assert position.loc[0, ["X", "Y", "Z"]].tolist() == [-3.1, -3.1, -14.0]
    assert position.loc[1, ["X", "Y", "Z"]].tolist() == [-3.1,  3.1, -14.0]
    assert position.loc[2, ["X", "Y", "Z"]].tolist() == [ 3.1, -3.1, -14.0]
    assert position.loc[3, ["X", "Y", "Z"]].tolist() == [ 3.1,  3.1, -14.0]

    assert (position.Plane == "NP").all()


def test_find_databases_recognizes_sqlite3_extension(tmp_path, monkeypatch):
    expected = tmp_path / "pColina.sqlite3"
    expected.touch()
    (tmp_path / "legacy.db").touch()
    (tmp_path / "legacy.sqlite").touch()
    (tmp_path / "notes.txt").touch()
    monkeypatch.setattr(io, "DATABASE_DIRECTORY", tmp_path)

    assert find_databases() == {Detector.pColina: expected}
    assert get_database(Detector.pColina) == expected


def test_load_sensor_calibration_rejects_missing_database(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "DATABASE_DIRECTORY", tmp_path)

    with raises(FileNotFoundError, match="No SQLite database"):
        load_sensor_calibration(Detector.pColina, 0)


def test_find_databases_rejects_unknown_detector(tmp_path, monkeypatch):
    (tmp_path / "unknown.sqlite3").touch()
    monkeypatch.setattr(io, "DATABASE_DIRECTORY", tmp_path)

    with raises(ValueError, match="Detector 'unknown' does not exist"):
        find_databases()
