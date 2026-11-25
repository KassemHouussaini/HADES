"""Province ambient temperatures (ECCC 1991-2020 normals) in the LCA presets."""
import importlib.util
import os

import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def lcx():
    spec = importlib.util.spec_from_file_location("lcx_ambient", os.path.join(_HERE, "lca_contexts.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_values_from_csv(lcx):
    assert lcx.get_preset("QC")["T_amb_C"] == 7.0
    assert lcx.get_preset("ON")["T_amb_C"] == 8.6
    assert lcx.get_preset("AB")["T_amb_C"] == 4.5
    assert set(lcx._AMBIENT) == {k for k in lcx.PRESETS if k != "QC_PAPER1"}


def test_paper1_keeps_default(lcx):
    assert lcx.ambient_overrides(lcx.get_preset("QC_PAPER1")) == {}


def test_provinces_inside_hi_range(lcx):
    spec = importlib.util.spec_from_file_location("hi_ambient", os.path.join(_HERE, "heat_integration.py"))
    hi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hi)
    lo, up = hi.T_AMB_RANGE_C
    for k in ("QC", "ON", "MB", "SK", "AB", "BC", "NB", "NS", "PE", "NL", "YT"):
        assert lo <= lcx.ambient_overrides(lcx.get_preset(k))["T_AMB_C"] <= up


def test_below_freezing_refused(lcx):
    for k in ("NT", "NU"):
        with pytest.raises(ValueError):
            lcx.ambient_overrides(lcx.get_preset(k))
