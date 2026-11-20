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
    assert set(hce.QC_HEAT_MIX) == set(hce.HEAT_MIX_DATASETS)


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
