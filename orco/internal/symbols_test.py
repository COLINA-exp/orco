from pytest import raises

from .symbols import all_symbols
from .symbols import Detector


def test_detector_values_match_names():
    assert Detector.pColina.value == "pColina"
    assert Detector.Colina.value  == "Colina"


def test_detector_get_ignores_case():
    assert Detector.get("PCOLINA") is Detector.pColina
    assert Detector.get("colina")  is Detector.Colina


def test_detector_get_rejects_unknown_detector():
    with raises(ValueError, match="Detector 'unknown' does not exist"):
        Detector.get("unknown")


def test_all_symbols_contains_enum_members():
    symbols = all_symbols()

    assert symbols["pColina"] is Detector.pColina
    assert symbols["Colina"]  is Detector.Colina
