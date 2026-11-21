"""Feedstock-specific Pathway B digester feed (ADM1.PATHWAY_B_FEED_PARAMS)."""
import pytest

FW = dict(C=57.4, H=8.2, O=64.5, N=5.5, S=0.75)       # kg, food waste base case order of magnitude


def test_legacy_fractions_unchanged_without_feedstock(adm1):
    fr = adm1.derive_adm1_fractions(**FW, substrate_type="feedstock")
    assert fr["X_I"] == 0.08 and fr["S_I"] == 0.02
    sol = fr["S_su"] + fr["S_aa"] + fr["S_fa"]
    assert sol / 0.90 == pytest.approx(0.27)


@pytest.mark.parametrize("name", sorted(["food_waste", "ofmsw", "dairy_manure", "wheat_straw", "sewage_sludge"]))
def test_feedstock_fractions(adm1, name):
    p = adm1.pathway_b_feed_params(name)
    fr = adm1.derive_adm1_fractions(**FW, substrate_type="feedstock", feed_params=p)
    assert fr["S_I"] == 0.0
    assert fr["X_I"] == pytest.approx(1 - p["fD"])
    sol = fr["S_su"] + fr["S_aa"] + fr["S_fa"]
    part = fr["X_ch"] + fr["X_pr"] + fr["X_li"]
    assert sol == pytest.approx(p["fD"] * p["f_sol"])
    assert part == pytest.approx(p["fD"] * (1 - p["f_sol"]))
    assert sum(fr.values()) == pytest.approx(1.0)


def test_values_as_specified(adm1):
    P = adm1.PATHWAY_B_FEED_PARAMS
    assert (P["food_waste"]["fD"], P["food_waste"]["f_sol"], P["food_waste"]["k_hyd"]) == (0.86, 0.44, 0.18)
    assert (P["dairy_manure"]["fD"], P["dairy_manure"]["f_sol"], P["dairy_manure"]["k_hyd"]) == (0.47, 0.03, 0.15)
    assert (P["wheat_straw"]["fD"], P["wheat_straw"]["f_sol"], P["wheat_straw"]["k_hyd"]) == (0.74, 0.17, 0.08)
    assert (P["sewage_sludge"]["fD"], P["sewage_sludge"]["f_sol"], P["sewage_sludge"]["k_hyd"]) == (0.62, 0.21, 0.36)
    assert P["ofmsw"]["fD"] == P["food_waste"]["fD"]


def test_translate_carries_k_hyd_and_overrides(adm1):
    kw = dict(C_feed=57.4, H_feed=8.2, O_feed=64.5, N_feed=5.5, S_feed=0.75,
              feedstock_wet_kg=1000.0, ash_kg=13.65, mode="continuous", HRT_days=30)
    legacy = adm1.translate_feedstock_to_adm1(**kw)
    assert "_k_hyd" not in legacy
    d = adm1.translate_feedstock_to_adm1(**kw, feedstock_name="dairy_manure", feed_overrides={"fD": 0.40})
    assert d["_k_hyd"] == 0.15 and d["_fD"] == 0.40
    assert d["_fractions"]["X_I"] == pytest.approx(0.60)


def test_unknown_feedstock_or_override_raises(adm1):
    with pytest.raises(ValueError):
        adm1.pathway_b_feed_params("garden_waste")
    with pytest.raises(ValueError):
        adm1.pathway_b_feed_params("food_waste", {"k_dis": 1.0})
