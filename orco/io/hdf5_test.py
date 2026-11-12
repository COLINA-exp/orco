from json import dumps
from json import JSONDecodeError

import numpy  as np
import pandas as pd
import tables as tb

from numpy.testing import assert_array_equal
from pytest import raises
from pytest import mark
from pytest import warns

from ..internal.exceptions import TableMismatch
from . hdf5                import _decode_str_columns
from . hdf5                import df_writer
from . hdf5                import read_acquisition_config


@mark.parametrize("config", ({}, dict(run_number=3,
                                     effective=dict(sample_period_ns=16),
                                     channels=[0, 1, 2], enabled=True,
                                     optional=None, label="calibraci\u00f3n")))
@mark.parametrize("numpy_scalar", (False, True))
def test_read_acquisition_config_returns_nested_json_without_modifying_input(tmp_path, config, numpy_scalar):
    filename = tmp_path / "waveforms.h5"
    payload = dumps(config, ensure_ascii=False).encode("utf-8")
    if numpy_scalar:
        payload = np.asarray(payload)
    with tb.open_file(filename, "w") as output:
        acquisition = output.create_group("/", "Acquisition")
        output.create_array(acquisition, "config_json", payload)
        subsection = output.create_group(acquisition, "Configuration")
        output.create_array(subsection, "config_json", b'{"different": true}')
    original = filename.read_bytes()

    result = read_acquisition_config(filename)

    assert result == config
    assert filename.read_bytes() == original
    # The returned data remains usable after the input handle has closed.
    with tb.open_file(filename, "a") as output:
        assert output.isopen


@mark.parametrize("has_acquisition", (False, True))
def test_read_acquisition_config_reports_missing_node(tmp_path, has_acquisition):
    filename = tmp_path / "waveforms.h5"
    with tb.open_file(filename, "w") as output:
        if has_acquisition:
            output.create_group("/", "Acquisition")

    with raises(tb.NoSuchNodeError):
        read_acquisition_config(filename)


def test_read_acquisition_config_reports_invalid_json(tmp_path):
    filename = tmp_path / "waveforms.h5"
    with tb.open_file(filename, "w") as output:
        acquisition = output.create_group("/", "Acquisition")
        output.create_array(acquisition, "config_json", b"{invalid JSON}")

    with raises(JSONDecodeError):
        read_acquisition_config(filename)


def test_decode_str_columns_preserves_numeric_columns():
    source = pd.DataFrame({"label": [b"one", b"two"], "value": [1, 2]})

    decoded = _decode_str_columns(source)

    assert decoded.label.tolist() == ["one", "two"]
    assert decoded.value.tolist() == [1, 2]
    assert decoded.value.dtype == source.value.dtype


def test_df_writer_creates_group_and_appends_rows(tmp_path):
    filename = tmp_path / "data.h5"
    first    = pd.DataFrame({"event": [1], "energy": [2.5]})
    second   = pd.DataFrame({"event": [2], "energy": [3.5]})

    with tb.open_file(filename, "w") as h5out:
        df_writer(h5out, first, "Events", "summary", "Event summary")
        df_writer(h5out, second, "Events", "summary")

    with tb.open_file(filename) as h5in:
        table = h5in.root.Events.summary
        assert table.nrows == 2
        assert table.attrs.TITLE == "Event summary"
        assert_array_equal(table.col("event"), [1, 2])
        assert_array_equal(table.col("energy"), [2.5, 3.5])


def test_df_writer_creates_table_for_empty_dataframe(tmp_path):
    filename = tmp_path / "empty.h5"
    empty = pd.DataFrame({"event": pd.Series(dtype=int),
                          "energy": pd.Series(dtype=float)})

    with tb.open_file(filename, "w") as h5out:
        df_writer(h5out, empty, "Events", "summary")

    with tb.open_file(filename) as h5in:
        assert h5in.root.Events.summary.nrows == 0
        assert h5in.root.Events.summary.colnames == ["event", "energy"]


def test_df_writer_rejects_different_columns(tmp_path):
    filename = tmp_path / "mismatch.h5"

    with tb.open_file(filename, "w") as h5out:
        df_writer(h5out, pd.DataFrame({"event": [1]}), "Events", "summary")
        with raises(TableMismatch, match="table structure"):
            df_writer(h5out, pd.DataFrame({"other": [1]}), "Events", "summary")


def test_df_writer_warns_when_string_is_too_long(tmp_path):
    filename = tmp_path / "strings.h5"

    with tb.open_file(filename, "w") as h5out:
        df_writer(h5out, pd.DataFrame({"label": ["one"]}),
                  "Events", "labels", str_col_length=3)
        with warns(UserWarning, match="strings longer"):
            df_writer(h5out, pd.DataFrame({"label": ["lengthy"]}),
                      "Events", "labels")


def test_df_writer_rejects_non_ascii_strings(tmp_path):
    filename = tmp_path / "strings.h5"

    with tb.open_file(filename, "w") as h5out:
        with raises(ValueError, match="non-ASCII"):
            df_writer(h5out, pd.DataFrame({"label": ["España"]}),
                      "Events", "labels")

        assert "Events" not in h5out.root
