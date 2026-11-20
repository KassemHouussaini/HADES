"""digestate_simple: element conservation and limiting cases."""
import pytest

FEED = dict(C=57.4, H=8.2, O=64.5, N=5.5, S=0.75, ash_kg=13.65)


def _run(ac, CH4=28.09, CO2=60.04, NH4=2.03, name="food_waste"):
    return ac.digestate_simple(name, FEED["C"], FEED["H"], FEED["O"], FEED["N"], FEED["S"],
                               FEED["ash_kg"], CH4, CO2, 1875.0, NH4_N_kg=NH4)


def test_carbon_conservation(ac):
    r = _run(ac)
    C_biogas = 28.09 * 12.011 / 16.043 + 60.04 * 12.011 / 44.009
    assert r["_C_dig_kg"] + C_biogas + r["_dg_C_bicarbonate_kg"] == pytest.approx(FEED["C"], rel=1e-12)
    assert r["_dg_C_bicarbonate_kg"] == pytest.approx(2.03 * 12.011 / 14.007)


def test_nitrogen_sulfur_ash_conserved(ac):
    r = _run(ac)
    assert r["_N_dig_kg"] == pytest.approx(FEED["N"])      # organic N + NH4-N
    assert r["_S_dig_kg"] == pytest.approx(FEED["S"])
    assert r["_Ash_dig_kg"] == pytest.approx(FEED["ash_kg"])


def test_wet_mass_and_fractions(ac):
    r = _run(ac)
    assert r["_m_wet_kg"] == pytest.approx(1875.0 - 28.09 - 60.04)
    pct = sum(r[k] for k in ("C_feed", "H_feed", "O_feed", "N_feed", "S_feed", "Ash_feed"))
    assert pct == pytest.approx(100.0, abs=0.05)              # rounded to 2 decimals
    assert 0.0 < r["_dg_VS_destroyed"] < 1.0
    assert 0.0 < r["WC"] < 100.0


def test_measured_ratios_applied(ac):
    r = _run(ac)
    nC = r["_C_dig_kg"] / 12.011
    # Cao et al. 2019 (Waste Manag. 100, 138-150), Table 2, organic household waste
    # digestate: C 34.3, H 4.0, O 23.9 %db -> H/C = (4.0/1.008)/(34.3/12.011) = 1.39,
    # O/C = (23.9/15.999)/(34.3/12.011) = 0.523
    assert (r["_H_dig_kg"] / 1.008) / nC == pytest.approx(1.39)
    assert (r["_O_dig_kg"] / 15.999) / nC == pytest.approx(0.523)


def test_coefficients_match_cited_tables(ac):
    # Recompute each coefficient from the published table values it cites.
    p = ac.AD_CHARACTERIZATION
    hc = lambda C, H: (H / 1.008) / (C / 12.011)
    oc = lambda C, O: (O / 15.999) / (C / 12.011)
    # food waste: Cao et al. 2019 Table 2 (OHWD)
    assert p["food_waste"]["HC_molar"] == pytest.approx(hc(34.3, 4.0), abs=0.005)
    assert p["food_waste"]["OC_molar"] == pytest.approx(oc(34.3, 23.9), abs=0.005)
    assert p["food_waste"]["fc_ratio_central"] == pytest.approx(14.9 / 100, abs=1e-3)
    # food waste TAN/TKN: Zhang et al. 2012, fibre 23.6/54.7, liquor 65.1/112
    assert p["food_waste"]["f_NH4"] == pytest.approx(65.1 / 112, abs=0.005)
    assert p["food_waste"]["f_NH4_range"] == (0.40, 0.60)
    # dairy digestate: Belete et al. 2021 Table 1, digestate column
    assert p["dairy_manure"]["HC_molar"] == pytest.approx(hc(26.8, 3.4), abs=0.005)
    assert p["dairy_manure"]["OC_molar"] == pytest.approx(oc(26.8, 22.8), abs=0.005)
    # sewage digestate: Aragon-Briceno et al. 2020 feedstock
    assert p["sewage_sludge"]["HC_molar"] == pytest.approx(hc(33.3, 4.6), abs=0.01)
    assert p["sewage_sludge"]["OC_molar"] == pytest.approx(oc(33.3, 20.3), abs=0.005)
    # OFMSW reuses food waste coefficients
    assert p["ofmsw"]["HC_molar"] == p["food_waste"]["HC_molar"]


def test_food_waste_vs_destroyed_plausible(ac):
    # Banks et al. 2018 (IEA Bioenergy Task 37): food waste 81-85 % VS destroyed.
    # HADES checks against a wider 60-90 % band (one source; flagged if outside).
    assert 0.60 <= _run(ac)["_dg_VS_destroyed"] <= 0.90


def test_no_biogas_limit(ac):
    r = _run(ac, CH4=0.0, CO2=0.0, NH4=0.0)
    assert r["_C_dig_kg"] == pytest.approx(FEED["C"])
    assert r["_m_wet_kg"] == pytest.approx(1875.0)


def test_ammonium_above_feed_n_raises(ac):
    with pytest.raises(ValueError):
        _run(ac, NH4=FEED["N"] * 1.5)


def test_fallback_fraction_when_no_adm1(ac):
    r = _run(ac, NH4=None)
    assert r["_NH4_N_kg"] == pytest.approx(0.58 * FEED["N"])   # f_NH4 food waste [4]


def test_ofmsw_uses_food_waste_coefficients(ac):
    a, b = _run(ac), _run(ac, name="ofmsw")
    for k in ("C_feed", "H_feed", "O_feed", "N_feed", "FC_feed", "VM_feed"):
        assert a[k] == b[k]


def test_unknown_feedstock_raises(ac):
    with pytest.raises(KeyError):
        _run(ac, name="unknown_feed")
