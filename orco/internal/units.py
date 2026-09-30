"""
Physical units used throughout ORCO.

Importing a shared registry ensures that quantities created in different modules
belong to the same Pint unit system.
"""

from pint        import UnitRegistry
from pint.errors import DimensionalityError
from pint.errors import UndefinedUnitError


units = UnitRegistry()

# --- internal system of units ------------------------------------------------
#
# The internal system extends to every quantity of the international system of
# units: it declares one base quantity per SI dimension, each defined as 1,
# and every other quantity is expressed relative to it:
#
#     SI quantity          internal base    value of 1
#     -------------------- ---------------- ------------------------------
#     time                 microsecond      1 us (the PMAP time column)
#     length               millimeter       1 mm (the sensor positions)
#     mass                 kilogram         1 kg
#     electric current     ampere           1 A
#     temperature          kelvin           1 K
#     amount of substance  mole             1 mol
#     luminous intensity   candela          1 cd
#     dimensionless        --               1 (ADC counts, thresholds, ...)
#
# Time and length use the units in which ORCO stores its data (the PMAP time
# column in microseconds, the sensor positions in millimeters); the remaining
# SI dimensions use their SI base units. Configurations may use any compatible
# unit, e.g. ``12 * nanosecond`` or ``5 * centimeter``: they are reduced to
# the corresponding internal base automatically. Every SI quantity -- including
# compound ones such as ``5 * volt`` or ``1 * joule`` -- reduces to the
# internal bases of its dimensions. A quantity of a dimension for which no
# base is declared is rejected, since the transformers only understand the
# internal units.

time_unit        = units.microsecond
length_unit      = units.millimeter
mass_unit        = units.kilogram
current_unit     = units.ampere
temperature_unit = units.kelvin
amount_unit      = units.mole
luminous_unit    = units.candela
count_unit       = 1   # the dimensionless base


def _dimension_name(unit):
    """Name (like ``[time]``) of the single dimension spanned by ``unit``."""
    return next(iter(unit.dimensionality.items()))[0]


_internal_base = {
    _dimension_name(unit): unit
    for unit in (time_unit, length_unit, mass_unit, current_unit,
                 temperature_unit, amount_unit, luminous_unit)
}


def _compose_target(dimensionality):
    """
    Internal unit matching ``dimensionality``, or ``None`` when one of its
    dimensions has no declared internal base.
    """
    target = None
    for name, exponent in dimensionality.items():
        base = _internal_base.get(name)
        if base is None:
            return None
        term = base ** exponent
        target = term if target is None else target * term
    return target


def reduce_to_internal(value):
    """
    Reduce a configuration value to the internal system of units.

    Plain numbers and dimensionless quantities are already in the system
    (their base is 1). Every other quantity is reduced to the internal bases
    of its SI dimensions: time to microseconds, length to millimeters, and
    mass, current, temperature, amount and luminous intensity to their SI
    base units. A quantity of a dimension without an internal base raises
    ``pint.errors.DimensionalityError``.

    Args:
        value: Configuration value, either a plain number or a Pint quantity.

    Returns:
        A plain number in the internal system of units.
    """
    if isinstance(value, units.Quantity):
        if value.dimensionless:
            return value.magnitude
        target = _compose_target(value.dimensionality)
        if target is None:
            raise DimensionalityError(value.units,
                                      "internal system of units",
                                      value.dimensionality,
                                      "undeclared dimension",
                                      "no internal unit is declared for this dimension")
        return value.to(target).magnitude
    return value


class _UnitScope(dict):
    def __missing__(self, name):
        try:
            unit = units.Unit(name)
        except UndefinedUnitError:
            raise KeyError(name) from None

        self[name] = unit
        return unit


def unit_scope() -> dict:
    """
    Return a namespace which resolves unknown names as Pint units.
    """
    return _UnitScope()
