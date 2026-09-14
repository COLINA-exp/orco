"""
Physical units used throughout ORCO.

Importing a shared registry ensures that quantities created in different modules
belong to the same Pint unit system.
"""

from pint        import UnitRegistry
from pint.errors import UndefinedUnitError


units = UnitRegistry()


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
