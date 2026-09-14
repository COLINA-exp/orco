import pandas as pd
import tables as tb

from numpy.testing import assert_array_equal
from pytest import raises
from pytest import warns

from ..internal.exceptions import TableMismatch
from . hdf5                import _decode_str_columns
from . hdf5                import df_writer


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
