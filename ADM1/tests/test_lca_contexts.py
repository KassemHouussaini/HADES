"""lca_contexts: regional presets (location priority + displaced heat mix)."""
import os
import pytest
from conftest import _load


@pytest.fixture(scope="module")
def lcx():
    return _load("p2_lcx_test", "lca_contexts.py")


def test_paper1_preset_unchanged(lcx):
    p = lcx.get_preset()
    assert p["key"] == "QC_PAPER1"
    assert p["location_priority"] == ["CA-QC", "CA", "RoW", "GLO"]   # Paper 1 LOC
    assert p["heat_mix_shares"] is None
    assert lcx.heat_mix_shares(p) == lcx.hce.QC_HEAT_MIX
    assert lcx.ctx_kwargs(p)["location_code"] == "CA-QC"


def test_province_presets(lcx):
    for code in ("QC", "ON", "MB", "SK", "AB", "BC", "NB", "NS", "PE", "NL", "YT", "NT", "NU"):
        p = lcx.get_preset(code)
        assert p["location_priority"][0] == p["location_code"]
        assert p["location_priority"][1:] == ["CA", "RoW", "GLO"]
        shares = lcx.heat_mix_shares(p)
        assert sum(shares.values()) == pytest.approx(1.0, abs=0.002)
        assert set(shares) <= set(lcx.hce.HEAT_MIX_DATASETS)
        assert p["biomethane_use"] == "heat"
    assert lcx.get_preset("CA-ON")["key"] == "ON"
    assert lcx.get_preset("CA-YK")["location_code"] == "CA-YK"
    assert lcx.heat_mix_shares(lcx.get_preset("PE")) == lcx.hce.REGIONAL_HEAT_MIX["ATL"]
    assert "Atlantic" in lcx.get_preset("NS")["note"]
    with pytest.raises(KeyError):
        lcx.get_preset("XX")


def test_preset_is_a_copy(lcx):
    p = lcx.get_preset("ON"); p["location_priority"].append("X")
    assert lcx.get_preset("ON")["location_priority"] == ["CA-ON", "CA", "RoW", "GLO"]


def test_ctx_kwargs_are_context_fields(lcx, sb):
    import dataclasses
    fields = {f.name for f in dataclasses.fields(sb.SystemContext)}
    assert set(lcx.ctx_kwargs(lcx.get_preset("AB"))) <= fields
    assert sb.resolve_heat_mix_shares(lcx.get_preset("AB")["heat_mix_shares"]) == lcx.hce.REGIONAL_HEAT_MIX["AB"]
