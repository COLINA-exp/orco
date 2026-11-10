from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated
from typing import get_args
from typing import get_type_hints

from pytest import approx
from pytest import mark
from pytest import raises

from .config import config_from_scope
from .config import config_value
from .exceptions import InvalidRange
from .symbols import Detector
from .types import ClosedInterval
from .types import Interval
from .units import Amplitude
from .units import Charge
from .units import Time
from .units import units


@dataclass
class ExampleConfig:
    time: Time
    amplitude: Amplitude
    charge: Charge
    count: int
    window: Annotated[Interval, units.microsecond]
    counts: Annotated[ClosedInterval, units.dimensionless]


@dataclass
class ParentConfig:
    child: ExampleConfig
    sampling_time: Time
    detector: Detector


def example_scope():
    return dict(s1_time=200 * units.nanosecond,
                s1_amplitude=2000 * units.pe / units.millisecond,
                s1_charge=3 * units.pe,
                s1_count=2,
                s1_min_window=100 * units.nanosecond,
                s1_max_window=1 * units.microsecond,
                s1_min_counts=1,
                s1_max_counts=3)


def test_aliases_resolve_from_postponed_annotations_with_unit_metadata():
    hints = get_type_hints(ExampleConfig, include_extras=True)

    assert get_args(hints["time"]) == (float, units.microsecond)
    assert get_args(hints["amplitude"]) == (float, units.pe / units.microsecond)
    assert get_args(hints["charge"]) == (float, units.pe)


def test_config_reader_converts_scalars_and_interval_bounds():
    config = config_from_scope(ExampleConfig, example_scope(), prefix="s1_")

    assert config.time == approx(0.2)
    assert config.amplitude == approx(2)
    assert config.charge == 3
    assert config.count == 2
    assert config.window.lower == approx(0.1)
    assert config.window.upper == 1
    assert config.counts == ClosedInterval(1, 3)
    assert not isinstance(config.time, units.Quantity)


def test_config_reader_assembles_nested_configs_and_validates_top_level_fields():
    child = config_from_scope(ExampleConfig, example_scope(), prefix="s1_")
    scope = dict(sampling_time=16 * units.nanosecond, detector=Detector.pColina)

    parent = config_from_scope(ParentConfig, scope, child=child)

    assert parent.child is child
    assert parent.detector is Detector.pColina
    assert parent.sampling_time == approx(0.016)

    with raises(TypeError, match="child: expected a constructed ExampleConfig"):
        config_from_scope(ParentConfig, scope, child=object())
    with raises(TypeError, match="Unexpected nested config fields"):
        config_from_scope(ParentConfig, scope, child=child, unused=child)


@mark.parametrize(("key", "value"),
                  (("s1_time", 0),
                   ("s1_amplitude", 2 * units.pe),
                   ("s1_charge", 3 * units.adc),
                   ("s1_count", 2 * units.microsecond),
                   ("s1_min_window", 0),
                   ("s1_max_window", 2 * units.pe),
                   ("s1_min_counts", 1.5),
                   ("s1_max_counts", True)))
def test_config_reader_reports_full_external_key_for_invalid_values(key, value):
    scope = example_scope()
    scope[key] = value

    with raises(ValueError, match=key):
        config_from_scope(ExampleConfig, scope, prefix="s1_")


def test_config_reader_reports_missing_interval_bound():
    scope = example_scope()
    del scope["s1_max_window"]

    with raises(KeyError, match="s1_max_window"):
        config_from_scope(ExampleConfig, scope, prefix="s1_")


def test_config_reader_preserves_interval_validation():
    scope = example_scope()
    scope["s1_min_window"] = 2 * units.microsecond

    with raises(InvalidRange):
        config_from_scope(ExampleConfig, scope, prefix="s1_")


def test_config_reader_does_not_silently_ignore_unknown_prefixed_fields():
    scope = example_scope()
    scope["s1_typo"] = 1

    with raises(ValueError, match="s1_typo"):
        config_from_scope(ExampleConfig, scope, prefix="s1_")


def test_config_reader_rejects_unannotated_float_fields():
    @dataclass
    class MissingUnit:
        height: float

    with raises(TypeError, match="height: unsupported config annotation"):
        config_from_scope(MissingUnit, dict(height=1 * units.pe))


@mark.parametrize(("value", "expected_unit", "expected"),
                  ((200 * units.nanosecond, units.microsecond, 0.2),
                   (2 * units.pe, units.pe, 2),
                   (2000 * units.pe / units.millisecond, units.pe / units.microsecond, 2),
                   (3, units.dimensionless, 3),
                   (3 * units.dimensionless, units.dimensionless, 3)))
def test_config_value_validates_units_before_reduction(value, expected_unit, expected):
    assert config_value(value, expected_unit, name="setting") == approx(expected)


@mark.parametrize(("value", "expected_unit"),
                  ((0, units.microsecond),
                   (1, units.pe),
                   (2, units.pe / units.microsecond),
                   (2 * units.pe, units.pe / units.microsecond),
                   (2 * units.adc, units.pe),
                   (5 * units.microsecond, units.dimensionless),
                   (1.5, units.dimensionless),
                   (True, units.dimensionless)))
def test_config_value_rejects_missing_or_incorrect_units(value, expected_unit):
    with raises(ValueError, match="setting: expected"):
        config_value(value, expected_unit, name="setting")
