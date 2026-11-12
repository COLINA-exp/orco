#!/usr/bin/env python3
"""
Utilities for HDF5 IO operations.

The first dataframe written to a group/table pair defines the persistent
table schema. Later writes are appended after their columns and numeric types
have been checked against that schema.

Dataframe string fields use fixed-width ASCII bytes, not UTF-8. Numeric checks
use NumPy's ``same_kind`` casting rule, not value-range validation; narrowing may
lose data. Acquisition JSON uses the standard JSON decoder independently of
these dataframe-writing conventions.
"""

import warnings
from json import loads
from pathlib import Path

import numpy  as np
import pandas as pd
import tables as tb

from typing import Any
from typing import Mapping

from .. internal.exceptions import TableMismatch
from .. internal.units      import reduce_to_internal
from .. internal.units      import units


def copy_acquisition(input_file: str | Path, output_file: tb.File) -> None:
    """
    Copy the full ``/Acquisition`` group to an open output file.

    Nested groups, arrays, tables and their attributes are copied without
    interpreting their contents. The source is opened read-only and closed
    before returning. If ``/Acquisition`` is absent, nothing is copied.

    Args:
        input_file: Input HDF5 filename.
        output_file: Open, writable HDF5 file receiving the group at its root.

    Raises:
        tables.NodeError: If the output already contains ``/Acquisition``;
            existing metadata is not overwritten.
    """
    with tb.open_file(input_file, "r") as infile:
        if "/Acquisition" in infile:
            infile.copy_node("/Acquisition", newparent=output_file.root, recursive=True)


def read_acquisition_config(input_file: str | Path) -> dict[str, Any]:
    """
    Read the full run configuration from ``/Acquisition/config_json``.

    The input node is a scalar HDF5 string array containing a JSON object,
    not a PyTables table. The separate ``/Acquisition/Configuration/config_json``
    subsection is not read. The file is opened read-only and closed before
    returning a plain dictionary, including nested JSON values. No configuration
    settings are applied or validated by this reader.

    Args:
        input_file: Input HDF5 filename.

    Returns:
        The JSON configuration decoded using Python's standard JSON decoder.

    Raises:
        tables.NoSuchNodeError: If the configuration node is missing.
        json.JSONDecodeError: If the stored text is not valid JSON.
    """
    with tb.open_file(input_file, "r") as infile:
        payload = np.asarray(infile.get_node("/Acquisition/config_json").read()).item()
    return loads(payload)


def get_sampling_time(config: Mapping[str, Any]) -> float:
    """
    Extract the DAQ sampling time from the acquisition JSON config.

    Return the period in internal units.

    Args:
        config: Acquisition configuration returned by ``read_acquisition_config``.

    Returns:
        sampling_time

    Raises:
        KeyError if the required fields are not present.
    """
    period = config["effective"]["sample_period_ns"]
    return reduce_to_internal(period * units.nanosecond)


def _decode_str_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with byte-valued object columns decoded as ASCII.
    """
    def to_string(col: pd.Series) -> pd.Series:
        if col.dtype != object:
            return col
        return col.astype(bytes).str.decode('ascii')
    return df.apply(to_string)


def _make_tabledef(column_types: np.dtype[Any], str_col_length: int) -> dict[str, tb.Col]:
    """
    Build a PyTables description from a structured NumPy dtype.

    Pandas represents string columns as objects, so those columns are mapped
    to fixed-width ASCII byte fields of ``str_col_length`` bytes.
    """
    tabledef = {}
    for index, colname in enumerate(column_types.names):
        coltype = column_types[colname].name
        if coltype == 'object':
            tabledef[colname] = tb.StringCol(str_col_length, pos=index)
        else:
            tabledef[colname] = tb.Col.from_type(coltype, pos=index)
    return tabledef


def _check_ascii_strings(arr: np.recarray) -> None:
    """
    Validate that every value in an object column is ASCII-compatible.

    Raises:
        ValueError: If a string or byte value contains non-ASCII data.
    """
    for name in arr.dtype.names:
        if arr.dtype[name].name != 'object':
            continue

        for value in arr[name]:
            try:
                if isinstance(value, bytes):
                    value.decode('ascii')
                elif isinstance(value, str):
                    value.encode('ascii')
            except UnicodeError as error:
                raise ValueError(f"column {name!r} contains non-ASCII string data") from error


def _check_castability(arr: np.recarray, table_types: np.dtype[Any]) -> None:
    """
    Validate that records can be appended to an existing table.

    Column order may differ because :func:`df_writer` reorders records before
    appending. Column names must match and numeric types must satisfy NumPy's
    ``same_kind`` casting rule. Narrowing within a kind is allowed without
    checking values for overflow or precision loss. Oversized strings produce
    a warning because they are truncated to the table's fixed width.

    Raises:
        TableMismatch: If columns or numeric types are incompatible.
    """
    arr_types = arr.dtype

    if set(arr_types.names) != set(table_types.names):
        raise TableMismatch('dataframe differs from existing table structure')

    for name in arr_types.names:
        if arr_types[name].name == 'object':
            max_str_length = max(map(len, arr[name]))
            if max_str_length > table_types[name].itemsize:
                warnings.warn('dataframe contains strings longer than allowed', UserWarning)

        elif not np.can_cast(arr_types[name], table_types[name], casting='same_kind'):
            raise TableMismatch('dataframe numeric types not consistent with the table existing ones')


def make_table( hdf5_file  : tb.File
              , group      : str
              ,  name      : str
              , fformat    : dict[str, tb.Col]
              , description: str = ""
              ) -> tb.Table:
    """
    Create and return a table, creating its root-level group if needed.
    """
    if group not in hdf5_file.root:
        hdf5_file.create_group(hdf5_file.root, group)

    table = hdf5_file.create_table( hdf5_file.root[group]
                                  , name
                                  , fformat
                                  , description
                                  )
    return table


def df_writer(h5out: tb.File,
              df: pd.DataFrame,
              group_name: str,
              table_name: str,
              descriptive_string: str = "",
              str_col_length: int = 32,
              ) -> None:
    """
    Append a dataframe to a table, creating the table when necessary.

    An empty dataframe still creates the group and table, allowing callers to
    produce a stable output schema even when no rows pass their selection.

    Object columns are expected to contain ASCII strings or bytes. Oversized
    strings warn and are truncated; choose their width on the first write.
    Numeric narrowing is allowed within a kind and is not range-checked.

    Args:
        h5out: Open PyTables file with write permission.
        df: Dataframe whose index is omitted from the stored records.
        group_name: Name of the root-level destination group.
        table_name: Name of the destination table within that group.
        descriptive_string: Human-readable table description.
        str_col_length: Fixed byte width used when creating string columns.

    Raises:
        TableMismatch: If ``df`` is incompatible with an existing table.
        ValueError: If a string column contains non-ASCII data.
    """
    arr = df.to_records(index=False)
    _check_ascii_strings(arr)

    if group_name not in h5out.root or table_name not in h5out.root[group_name]:
        tabledef = _make_tabledef(arr.dtype, str_col_length=str_col_length)
        table    =  make_table(h5out,
                               group       = group_name,
                               name        = table_name,
                               fformat     = tabledef,
                               description = descriptive_string)

    else:
        table = h5out.root[group_name][table_name]

    if len(df) == 0:
        return

    data_types = table.dtype
    _check_castability(arr, data_types)

    columns = list(data_types.names)
    arr = arr[columns].astype(data_types)

    table.append(arr)
    table.flush()
