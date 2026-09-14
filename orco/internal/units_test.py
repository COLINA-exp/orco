from .units import unit_scope
from .units import units


def test_unit_scope_resolves_prefixed_unit_names():
    scope = unit_scope()

    assert scope["nanosecond"] == units.nanosecond
