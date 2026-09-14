#!/usr/bin/env python3

class TableMismatch(ValueError):
    """
    Raised when trying to append data to an existing table with a different
    schema.
    """

class InvalidIndexRange(ValueError):
    """
    Raised when index-range bounds cannot describe a valid range.
    """


class InvalidShape(ValueError):
    """
    Raised when array shapes are not valid or mutually consistent.
    """
