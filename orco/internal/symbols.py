#!/usr/bin/env python3

from __future__ import annotations

from enum import Enum
from enum import EnumMeta
from enum import auto
from functools import lru_cache

class AutoNameEnum(Enum):
    """
    Automatically generate Enum values from their names.

    Use this as a base class to make Enums with values which automatically
    match the member names:

    class Direction(AutoNameEnum):
    ...     LEFT  = auto()
    ...     RIGHT = auto()
    ...
    >>> list(Direction)
    [<Direction.LEFT: 'LEFT'>, <Direction.RIGHT: 'RIGHT'>]
    """
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name


class Detector(AutoNameEnum):
    pColina = auto()
    Colina  = auto()

    @classmethod
    def get(cls, label: str) -> Detector:
        """
        Return the detector whose name matches ``label``, ignoring case.
        """
        for detector in cls:
            if detector.name.lower() == label.lower():
                return detector

        raise ValueError(f"Detector {label!r} does not exist")


@lru_cache()
def all_symbols() -> dict:
    """Return all members of ORCO symbol enumerations, keyed by name."""
    out = {}
    for etype in globals().values():
        if not isinstance(etype, EnumMeta): continue
        if etype is AutoNameEnum          : continue

        for symbol in etype:
            out[symbol.name] = symbol
    return out
