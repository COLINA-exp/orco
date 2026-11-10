"""
Shared configuration validation and annotation-driven dataclass loading.

Physical fields require explicit compatible units, then become plain magnitudes
in ORCO's internal system. Unit definitions and aliases remain in units.py;
transformer-specific configuration dataclasses remain in their own modules.
"""

from dataclasses import fields
from dataclasses import is_dataclass
from enum        import Enum
from numbers     import Integral

from pint.errors import DimensionalityError

from typing import Annotated
from typing import Any
from typing import Mapping
from typing import TypeVar
from typing import get_args
from typing import get_origin
from typing import get_type_hints

from .types import ClosedInterval
from .types import Interval
from .units import reduce_to_internal
from .units import units


_Config = TypeVar("_Config")


def config_value(value, expected_unit, *, name: str):
    """
    Validate a config field's dimension and return its internal magnitude.

    Dimensional fields require explicit Pint quantities, including zero values.
    Dimensionless fields represent counts and must reduce to integers, not
    booleans or fractional values. Errors identify the original config key.
    ``expected_unit`` is mandatory; general unit reduction remains available
    separately through ``reduce_to_internal``.
    """
    if isinstance(value, units.Quantity):
        try:
            value = value.to(expected_unit)
        except DimensionalityError as error:
            raise ValueError(f"{name}: expected {expected_unit}, got {value.units}") from error
        value = reduce_to_internal(value)
    elif not expected_unit.dimensionless:
        raise ValueError(f"{name}: expected an explicit quantity in {expected_unit}")

    if expected_unit.dimensionless:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name}: expected a dimensionless integer")
        return int(value)
    return value


def config_from_scope(cls: type[_Config], scope: Mapping[str, Any], *,
                      prefix: str = "", **nested: Any) -> _Config:
    """
    Build a config dataclass from unit-annotated fields in a flat namespace.

    ``Annotated[float, unit]`` declares a physical scalar. Annotated ``Interval``
    or ``ClosedInterval`` fields read ``min_<field>`` and ``max_<field>`` with
    the same expected unit. Plain integers require dimensionless integer values.
    ``prefix`` precedes every external key, including min/max keys. Enum fields
    pass through as enum instances. Nested configs must be explicitly supplied
    as already constructed dataclasses via keyword arguments.

    Postponed annotations are resolved with metadata retained. Missing keys
    raise KeyError; invalid values retain their full external field names.
    Unsupported or unannotated float fields raise TypeError rather than bypass
    dimensional validation. Constructors retain their existing range checks.
    """

    hints = get_type_hints(cls, include_extras=True)
    values = {}
    consumed = set()
    consumed_nested = set()
    for field in fields(cls):
        if not field.init:
            continue
        annotation = hints[field.name]
        key = prefix + field.name
        if is_dataclass(annotation) and annotation not in (Interval, ClosedInterval):
            value = nested[field.name]
            if not isinstance(value, annotation):
                raise TypeError(f"{key}: expected a constructed {annotation.__name__}")
            values[field.name] = value
            consumed_nested.add(field.name)
            continue
        if get_origin(annotation) is Annotated:
            base, expected_unit = get_args(annotation)
            if not isinstance(expected_unit, units.Unit):
                raise TypeError(f"{key}: expected unit metadata")
            if base in (Interval, ClosedInterval):
                lower_key = prefix + "min_" + field.name
                upper_key = prefix + "max_" + field.name
                lower = config_value(scope[lower_key], expected_unit, name=lower_key)
                upper = config_value(scope[upper_key], expected_unit, name=upper_key)
                values[field.name] = base(lower, upper)
                consumed.update((lower_key, upper_key))
                continue
            if base is not float:
                raise TypeError(f"{key}: unsupported unit-annotated type {base}")
            values[field.name] = config_value(scope[key], expected_unit, name=key)
        elif annotation is int:
            values[field.name] = config_value(scope[key], units.dimensionless, name=key)
        elif isinstance(annotation, type) and issubclass(annotation, Enum):
            value = scope[key]
            if not isinstance(value, annotation):
                raise TypeError(f"{key}: expected {annotation.__name__}")
            values[field.name] = value
        else:
            raise TypeError(f"{key}: unsupported config annotation {annotation}")
        consumed.add(key)

    unknown_nested = nested.keys() - consumed_nested
    if unknown_nested:
        raise TypeError(f"Unexpected nested config fields: {sorted(unknown_nested)}")
    if prefix:
        unknown = {key for key in scope if key.startswith(prefix)} - consumed
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    return cls(**values)
