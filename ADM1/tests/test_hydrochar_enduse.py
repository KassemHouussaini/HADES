"""hydrochar_enduse: heat delivered and stack emissions of the hydrochar boiler."""
import math
import pytest

INV = {"C_char (%)": 55.0, "H_char (%)": 5.5, "O_char (%)": 25.0, "N_char (%)": 3.0,
       "S_char (%)": 0.3, "Ash_char (%)": 11.2, "HHV_char (MJ/kg)": 23.0}


def test_hhv_coal_reference(hce):
    # ecoinvent 3.12 CA-QC hard coal furnace input: "heating value, net" 28.9 MJ/kg;
    # net/gross 0.95 (IEA/OECD/Eurostat 2005, Energy Statistics Manual)
    assert hce.HHV_COAL_MJ_PER_KG == pytest.approx(28.9 / 0.95, rel=1e-12)
    assert hce.HHV_COAL_MJ_PER_KG == pytest.approx(30.42, abs=0.01)


def test_qc_heat_mix_shares(hce):
    # Quebec industrial combustion heat mix, electricity excluded: rounded shares sum to 0.999
    assert sum(hce.QC_HEAT_MIX.values()) == pytest.approx(0.999, abs=1e-9)
    assert set(hce.QC_HEAT_MIX) <= set(hce.HEAT_MIX_DATASETS)


def test_regional_heat_mixes(hce):
    # NRCan CEUD 2023, Industrial (aggregated), Table 1, PJ; electricity, "Other",
    # still gas and petroleum coke excluded by default
    for region, mix in hce.REGIONAL_HEAT_MIX.items():
        assert set(mix) <= set(hce.HEAT_MIX_DATASETS), region
        assert sum(mix.values()) == pytest.approx(1.0, abs=1e-12), region
        assert min(mix.values()) > 0.0, region
    qc = hce.regional_heat_mix("CA-QC")      # 301.5 PJ: sum of the fuel rows
    assert qc["natural_gas"] == pytest.approx(163.6 / 301.5, rel=1e-9)
    assert qc["bioenergy"] == pytest.approx(82.0 / 301.5, rel=1e-9)
    assert qc["petroleum"] == pytest.approx((33.7 + 6.3) / 301.5, rel=1e-9)
    assert qc["coal_coke"] == pytest.approx((8.4 + 4.6) / 301.5, rel=1e-9)
    assert qc["lpg_ngl"] == pytest.approx(2.9 / 301.5, rel=1e-9)
    # with refinery fuels: reproduces 48.9 / 24.5 / 21.8 / 3.9 / 0.9 (QC) and Canada 2952 PJ
    qi = hce.regional_heat_mix("QC", include_refinery_fuels=True)
    assert [round(100 * qi[k], 1) for k in ("natural_gas", "bioenergy", "petroleum", "coal_coke", "lpg_ngl")] \
        == [48.9, 24.5, 21.8, 3.9, 0.9]
    ca = hce.NRCAN_TABLE1_PJ["CA"]
    assert ca["total"] - ca["electricity"] - ca["other"] == pytest.approx(2951.6, abs=0.05)
    # rows of the stored tables add up to the printed totals (rounding <= 0.3 PJ)
    for r, v in hce.NRCAN_TABLE1_PJ.items():
        if r != "CA":
            assert sum(x for k, x in v.items() if k != "total") == pytest.approx(v["total"], abs=0.3), r
    # grouped regions, zero groups dropped, regional dataset weights
    assert hce.regional_heat_mix("NS") == hce.REGIONAL_HEAT_MIX["ATL"]
    assert hce.regional_heat_mix("bc") == hce.REGIONAL_HEAT_MIX["BC_TER"]
    assert "coal_coke" not in hce.regional_heat_mix("SK")
    ds = hce.regional_heat_mix_datasets("ON")
    assert [w for _, _, w in ds["coal_coke"]] == [9.3, 87.4]
    assert "propane" in ds["lpg_ngl"][0][0]
    with pytest.raises(KeyError):
        hce.regional_heat_mix("XX")


def test_mj_hand_calculation(hce):
    q = hce.compute_enduse_quantities(INV, 100.0)
    assert q["MJ_hydrochar"] == pytest.approx(100.0 * 23.0 / (28.9 / 0.95) / 0.04325, rel=1e-12)
    assert q["SubstitutionFactor_HC_COAL"] == pytest.approx(23.0 / (28.9 / 0.95))


def test_carbon_and_sulfur_conservation(hce):
    q = hce.compute_enduse_quantities(INV, 100.0)
    assert q["CO2_bio_kg"] * 12.0 / 44.0 == pytest.approx(55.0, rel=1e-12)   # all char C -> CO2
    assert q["SO2_kg"] * 32.0 / 64.0 == pytest.approx(0.3, rel=1e-12)        # all S -> SO2 (uncapped)
    assert q["ash_kg"] == pytest.approx(11.2)


def test_flue_gas_pure_carbon_analytic(hce):
    # 1 kg C: 83.26 mol CO2 + 3.762 x 83.26 mol N2 = 8.887 Nm3 dry stoichiometric,
    # x 21/(21-6) at 6 % O2 (Directive (EU) 2015/2193 Annex IV)
    n = 1000.0 / 12.0
    v0 = (n + 79.0 / 21.0 * n) * 0.022414
    assert hce.dry_flue_gas_Nm3_per_kg(100, 0, 0, 0, 0) == pytest.approx(v0 * 21 / 15, rel=1e-12)
    assert v0 == pytest.approx(8.89, abs=0.01)


def test_caps_bound_emissions(hce):
    q = hce.compute_enduse_quantities(INV, 100.0)
    V = q["V_fluegas_Nm3"]
    assert q["NOx_comb_kg"] <= hce.ELV_KG_PER_NM3["NOx"] * V * (1 + 1e-12)
    assert q["SO2_comb_kg"] <= hce.ELV_KG_PER_NM3["SO2"] * V * (1 + 1e-12)
    assert q["PM_comb_kg"] <= hce.ELV_KG_PER_NM3["dust"] * V * (1 + 1e-12)
    assert q["NOx_comb_kg"] <= q["NOx_kg"] and q["SO2_comb_kg"] <= q["SO2_kg"]
    assert q["NOx_per_MJ"] * q["MJ_hydrochar"] == pytest.approx(q["NOx_comb_kg"])


def test_caps_disabled_returns_composition_estimate(hce, monkeypatch):
    monkeypatch.setattr(hce, "ELV_CAP_ENABLED", False)
    q = hce.compute_enduse_quantities(INV, 100.0)
    assert q["NOx_comb_kg"] == q["NOx_kg"] and q["SO2_comb_kg"] == q["SO2_kg"]
    assert q["PM_scale"] == pytest.approx(11.2 / 10.0)


def test_zero_char_is_zero(hce):
    q = hce.compute_enduse_quantities(INV, 0.0)
    assert q["MJ_hydrochar"] == 0.0 and q["NOx_per_MJ"] == 0.0 and q["PM_scale"] == 0.0


@pytest.mark.parametrize("bad", [
    {"HHV_char (MJ/kg)": float("nan")}, {"HHV_char (MJ/kg)": 45.0},
    {"C_char (%)": 120.0}, {"Ash_char (%)": -1.0}])
def test_invalid_properties_raise(hce, bad):
    with pytest.raises(ValueError):
        hce.compute_enduse_quantities({**INV, **bad}, 100.0)


def test_invalid_mass_and_nox_raise(hce):
    with pytest.raises(ValueError):
        hce.compute_enduse_quantities(INV, -1.0)
    with pytest.raises(ValueError):
        hce.compute_enduse_quantities(INV, 100.0, nox_conv=0.5)


class _Exc(dict):
    pass


class _Act(dict):
    def __init__(self, name):
        super().__init__(name=name); self.ex = []

    def new_exchange(self, **kw):
        e = _Exc(kw); e.save = lambda: self.ex.append(e); return e


def test_heat_mix_shares_switch(sb, hce):
    assert sb.resolve_heat_mix_shares(None) == hce.QC_HEAT_MIX          # Paper 1 default
    assert sb.resolve_heat_mix_shares("ON") == hce.REGIONAL_HEAT_MIX["ON"]
    with pytest.raises(ValueError):
        sb.resolve_heat_mix_shares({"natural_gas": 0.5, "lng_typo": 0.5})
    shares = sb.resolve_heat_mix_shares("QC")
    comps = {g: [(f"act_{g}", 1.0)] for g in shares}
    mix = _Act("mix")
    sb.build_heat_mix_exchanges(mix, comps, shares)
    amounts = {e["input"]: e["amount"] for e in mix.ex}
    assert sum(amounts.values()) == pytest.approx(1.0, rel=1e-12)       # renormalized to 1 MJ
    assert amounts["act_lpg_ngl"] == pytest.approx(2.9 / 301.5, rel=1e-9)


def test_removed_mass_is_uncontrolled_minus_emitted(hce, monkeypatch):
    q = hce.compute_enduse_quantities(INV, 100.0)
    assert q["SO2_removed_kg"] == pytest.approx(q["SO2_kg"] - q["SO2_comb_kg"])
    assert q["NOx_removed_kg"] == pytest.approx(q["NOx_kg"] - q["NOx_comb_kg"])
    assert q["PM_removed_kg"] == pytest.approx(q["PM_uncapped_kg"] - q["PM_comb_kg"])
    assert min(q["SO2_removed_kg"], q["NOx_removed_kg"], q["PM_removed_kg"]) >= 0.0
    # below the limits (a clean fuel) nothing is removed
    clean = dict(INV, **{"S_char (%)": 0.0, "N_char (%)": 0.0, "Ash_char (%)": 0.0})
    q0 = hce.compute_enduse_quantities(clean, 100.0, nox_conv=hce.NOX_CONV_RANGE[0])
    assert q0["SO2_removed_kg"] == 0.0 and q0["NOx_removed_kg"] == 0.0 and q0["PM_removed_kg"] == 0.0
    monkeypatch.setattr(hce, "ELV_CAP_ENABLED", False)
    q1 = hce.compute_enduse_quantities(INV, 100.0)
    assert q1["SO2_removed_kg"] == q1["NOx_removed_kg"] == q1["PM_removed_kg"] == 0.0
