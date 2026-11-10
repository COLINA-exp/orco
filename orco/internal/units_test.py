from pytest import approx
from pytest import mark
from pytest import raises

from pint.errors import DimensionalityError

from .units import adc_unit
from .units import amount_unit
from .units import count_unit
from .units import current_unit
from .units import length_unit
from .units import luminous_unit
from .units import mass_unit
from .units import photoelectron_unit
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


def test_photoelectron_is_an_independent_internal_base():
    assert photoelectron_unit == units.photoelectron
    assert units.pe == units.photoelectron
    assert dict(units.pe.dimensionality) == {"[photoelectron]": 1}
    assert not (1 * units.pe).dimensionless


@mark.parametrize("target", ("dimensionless", "coulomb", "mole"))
def test_photoelectron_cannot_convert_to_other_dimensions(target):
    with raises(DimensionalityError):
        (1 * units.pe).to(target)


@mark.parametrize(("expression", "expected"),
                  (("5 * pe", 5),
                   ("5 * photoelectron", 5),
                   ("2 * kilophotoelectron", 2000),
                   ("5 * pe / microsecond", 5),
                   ("1 * pe / nanosecond", 1000),
                   ("1 * pe / millisecond", 0.001),
                   ("(2 * pe / microsecond) * (500 * nanosecond)", 1),
                   ("(1 * kilophotoelectron) / (1 * pe)", 1000)))
def test_photoelectron_quantities_reduce_in_config_scope(expression, expected):
    assert reduce_to_internal(eval(expression, unit_scope())) == approx(expected)


def test_photoelectron_ratio_is_dimensionless():
    ratio = (3 * units.pe) / (2 * units.photoelectron)

    assert ratio.dimensionless
    assert reduce_to_internal(ratio) == approx(1.5)


def test_adc_is_an_independent_internal_base():
    assert adc_unit == units.adc
    assert dict(units.adc.dimensionality) == {"[adc]": 1}
    assert not (1 * units.adc).dimensionless


@mark.parametrize("target", ("dimensionless", "coulomb", "pe"))
def test_adc_cannot_convert_to_other_dimensions(target):
    with raises(DimensionalityError):
        (1 * units.adc).to(target)


@mark.parametrize(("expression", "expected"),
                  (("5 * adc", 5),
                   ("2 * kiloadc", 2000),
                   ("3 * adc * microsecond", 3),
                   ("20 * adc * nanosecond / pe", 0.02),
                   ("2 * adc * millisecond / photoelectron", 2000),
                   ("(2 * kiloadc) / (4 * adc)", 500)))
def test_adc_quantities_reduce_in_config_scope(expression, expected):
    assert reduce_to_internal(eval(expression, unit_scope())) == approx(expected)


def test_adc_gain_produces_photoelectron_rate_and_integrated_charge():
    amplitude = 10 * units.adc
    gain = 2 * units.adc * units.microsecond / units.pe
    rate = amplitude / gain
    charge = rate * (200 * units.nanosecond)

    assert rate.to(units.pe / units.microsecond).magnitude == approx(5)
    assert charge.to(units.pe).magnitude == approx(1)
    assert reduce_to_internal(rate) == approx(5)
    assert reduce_to_internal(charge) == approx(1)


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
    assert reduce_to_internal(2 * units.watt) == approx(2e-12)
    assert reduce_to_internal(1 * units.volt) == approx(1e-12)
    assert reduce_to_internal(1 * units.joule) == approx(1e-6)
    assert reduce_to_internal(1 * units.hertz) == approx(1e-6)

    # two spellings of the same physical quantity agree
    assert reduce_to_internal(1 * units.joule) == approx(
        reduce_to_internal(1 * units.newton * units.meter))
    assert reduce_to_internal(1 * units.watt) == approx(
        reduce_to_internal(1 * units.volt * units.ampere))


def test_reduce_to_internal_accepts_dimensionless_quantities():
    # dimensionless ratios are converted to canonical units before reading
    # their magnitude
    assert reduce_to_internal((3 * units.millisecond) / (1 * units.millisecond)) == 3
    assert reduce_to_internal((1 * units.millisecond) / (16 * units.nanosecond)) == approx(62500)


def test_reduce_to_internal_rejects_undeclared_dimensions():
    # a bespoke dimension has no internal base unit
    units.define("wibble = [undefined_quantity]")
    with raises(DimensionalityError):
        reduce_to_internal(5 * units.wibble)
