"""Heat supply of the HTC reactor, dryer and digester (system_builder)."""
import pytest


class _A(dict):
    pass


def _acts():
    elec = _A(name="electricity", unit="kilowatt hour")
    ng = _A(name="heat, natural gas", unit="megajoule")
    bio = _A(name="heat_biomethane", unit="megajoule")
    return elec, ng, bio


def test_electric_htc_matches_paper1_convention(sb):
    elec, ng, bio = _acts()
    plan, used, short = sb.plan_heat_supply({"ad": ng, "htc": elec, "dryer": elec},
                                            {"ad": 50.0, "htc": 324.0, "dryer": 36.0}, 0.0, bio, ng)
    assert plan["htc"] == [(elec, pytest.approx(324.0 / sb.hi.ETA_HTC_UTILITY / 3.6), "kilowatt hour")]
    assert plan["dryer"] == [(elec, pytest.approx(10.0), "kilowatt hour")]
    assert plan["ad"] == [(ng, 50.0, "megajoule")] and used == 0.0 and short == {}


def test_own_biogas_order_and_shortfall(sb):
    elec, ng, bio = _acts()
    src = {"ad": "own_biogas", "htc": "own_biogas", "dryer": "own_biogas"}
    plan, used, short = sb.plan_heat_supply(src, {"ad": 70.0, "htc": 320.0, "dryer": 160.0}, 250.0, bio, ng)
    assert plan["ad"] == [(bio, 70.0, "megajoule")]                        # digester first
    assert plan["htc"] == [(bio, 180.0, "megajoule"), (ng, 140.0, "megajoule")]
    assert plan["dryer"] == [(ng, 160.0, "megajoule")]
    assert used == pytest.approx(250.0)
    assert short == {"htc": pytest.approx(140.0), "dryer": pytest.approx(160.0)}


def test_own_biogas_surplus(sb):
    elec, ng, bio = _acts()
    plan, used, short = sb.plan_heat_supply({"ad": "own_biogas", "htc": ng, "dryer": elec},
                                            {"ad": 190.0, "htc": 115.0, "dryer": 68.0}, 1200.0, bio, ng)
    assert used == pytest.approx(190.0) and short == {}


def test_resolve_defaults_and_errors(sb):
    from types import SimpleNamespace
    elec, ng, bio = _acts()
    ctx = SimpleNamespace(elec=elec, drying_heat=elec, ad_heat=ng, heat_supply=None)
    assert sb.resolve_heat_sources(ctx) == {"htc": elec, "dryer": elec, "ad": ng}
    ctx.heat_supply = {"htc": "electricity", "ad": "own_biogas"}
    assert sb.resolve_heat_sources(ctx)["ad"] == "own_biogas"
    for bad in ({"boiler": ng}, {"htc": _A(name="x", unit="kilogram")}):
        ctx.heat_supply = bad
        with pytest.raises(ValueError):
            sb.resolve_heat_sources(ctx)


def test_own_biogas_heat_efficiency(sb):
    # 0.9 MJ heat per MJ methane: 100 MJ CH4 cover 90 MJ, burned fuel 100 MJ
    elec, ng, bio = _acts()
    plan, used, short = sb.plan_heat_supply({"ad": "own_biogas", "htc": ng, "dryer": elec},
                                            {"ad": 120.0, "htc": 0.0, "dryer": 0.0}, 100.0, bio, ng,
                                            heat_per_MJ_CH4=0.9)
    assert plan["ad"][0] == (bio, pytest.approx(100.0), "megajoule")
    assert plan["ad"][1] == (ng, pytest.approx(30.0), "megajoule")
    assert used == pytest.approx(100.0) and short == {"ad": pytest.approx(30.0)}
    for bad in (0.0, 1.2, float("nan")):
        with pytest.raises(ValueError):
            sb.plan_heat_supply({"ad": "own_biogas", "htc": ng, "dryer": elec},
                                {"ad": 1.0}, 10.0, bio, ng, heat_per_MJ_CH4=bad)
