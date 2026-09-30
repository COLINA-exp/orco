import pytest

from pint.errors import DimensionalityError

from .units import amount_unit
from .units import count_unit
from .units import current_unit
from .units import length_unit
from .units import luminous_unit
from .units import mass_unit
from .units import reduce_to_internal
from .units import temperature_unit
from .units import time_unit
from .units import unit_scope
from .units import units


def test_unit_scope_resolves_prefixed_unit_names():
    scope = unit_scope()

    assert scope["nanosecond"] == units.nanosecond


# -------------------------------------------------------------------- internal system of units


def test_time_base_is_defined_as_one_microsecond():
    assert time_unit == units.microsecond


def test_length_base_is_defined_as_one_millimeter():
    assert length_unit == units.millimeter


def test_mass_base_is_defined_as_one_kilogram():
    assert mass_unit == units.kilogram


def test_current_base_is_defined_as_one_ampere():
    assert current_unit == units.ampere


def test_temperature_base_is_defined_as_one_kelvin():
    assert temperature_unit == units.kelvin


def test_amount_base_is_defined_as_one_mole():
    assert amount_unit == units.mole


def test_luminous_base_is_defined_as_one_candela():
    assert luminous_unit == units.candela


def test_count_base_is_defined_as_one():
    assert count_unit == 1


def test_reduce_to_internal_keeps_plain_numbers():
    assert reduce_to_internal(12) == 12
    assert reduce_to_internal(0.5) == 0.5


def test_reduce_to_internal_defines_time_base_as_one():
    # one internal time unit is exactly one microsecond
    assert reduce_to_internal(eval("1 * microsecond", unit_scope())) == 1


def test_reduce_to_internal_converts_relative_time_units():
    scope = unit_scope()

    assert reduce_to_internal(eval("1000 * nanosecond", scope)) == 1
    assert reduce_to_internal(eval("10 * nanosecond"  , scope)) == 0.01
    assert reduce_to_internal(eval("2 * millisecond"  , scope)) == 2000.0


def test_reduce_to_internal_defines_length_base_as_one():
    # one internal length unit is exactly one millimeter
    assert reduce_to_internal(eval("1 * millimeter", unit_scope())) == 1


def test_reduce_to_internal_converts_relative_length_units():
    scope = unit_scope()

    assert reduce_to_internal(eval("5 * centimeter", scope)) == 50.0
    assert reduce_to_internal(eval("1 * meter"     , scope)) == 1000.0
    assert reduce_to_internal(eval("2500 * micrometer", scope)) == 2.5


def test_reduce_to_internal_defines_si_bases_as_one():
    scope = unit_scope()

    assert reduce_to_internal(eval("1 * kilogram", scope)) == 1
    assert reduce_to_internal(eval("1 * ampere"  , scope)) == 1
    assert reduce_to_internal(eval("1 * kelvin"  , scope)) == 1
    assert reduce_to_internal(eval("1 * mole"    , scope)) == 1
    assert reduce_to_internal(eval("1 * candela" , scope)) == 1


def test_reduce_to_internal_converts_relative_si_units():
    scope = unit_scope()

    assert reduce_to_internal(eval("500 * gram"        , scope)) == 0.5
    assert reduce_to_internal(eval("2 * milliampere"   , scope)) == 0.002
    assert reduce_to_internal(eval("2 * millimole"     , scope)) == 0.002
    assert reduce_to_internal(eval("3 * millicandela"  , scope)) == 0.003


def test_reduce_to_internal_accepts_compound_si_quantities():
    # compound quantities reduce to the internal bases of their dimensions
    assert reduce_to_internal(2 * units.watt) == pytest.approx(2e-12)
    assert reduce_to_internal(1 * units.volt) == pytest.approx(1e-12)
    assert reduce_to_internal(1 * units.joule) == pytest.approx(1e-6)
    assert reduce_to_internal(1 * units.hertz) == pytest.approx(1e-6)

    # two spellings of the same physical quantity agree
    assert reduce_to_internal(1 * units.joule) == pytest.approx(
        reduce_to_internal(1 * units.newton * units.meter))
    assert reduce_to_internal(1 * units.watt) == pytest.approx(
        reduce_to_internal(1 * units.volt * units.ampere))


def test_reduce_to_internal_accepts_dimensionless_quantities():
    # a dimensionless ratio is a plain number, its base is 1
    assert reduce_to_internal((3 * units.millisecond) / (1 * units.millisecond)) == 3


def test_reduce_to_internal_rejects_undeclared_dimensions():
    # a bespoke dimension has no internal base unit
    units.define("wibble = [undefined_quantity]")
    with pytest.raises(DimensionalityError):
        reduce_to_internal(5 * units.wibble)