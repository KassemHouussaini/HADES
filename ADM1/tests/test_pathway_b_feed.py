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
    assert (P["ofmsw"]["fD"], P["ofmsw"]["fD_range"]) == (0.70, (0.60, 0.80))
    assert (P["ofmsw"]["f_sol"], P["ofmsw"]["k_hyd"]) == (P["food_waste"]["f_sol"], P["food_waste"]["k_hyd"])


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


MANURE_KW = dict(C_feed=40.0, H_feed=5.3, O_feed=36.0, N_feed=2.3, S_feed=0.4,
                 feedstock_wet_kg=1000.0, ash_kg=16.0, mode="continuous", HRT_days=30,
                 TS_target=0.10, feedstock_name="dairy_manure")    # 100 kg TS per tonne


def test_fisgativa_chemistry_values(adm1):
    """Manure at 10 % TS (100 kg TS per tonne): Fisgativa 2020 Table 2 (TIC
    1.2 gC/kgWW, TS 191.3 g/kgWW, NH4+ 0.8 of TKN 5.2, lipids 4.7 % COD) and
    Table 5 (S_cat 0.10, S_an 0.02 kmol/m3), scaled per kg TS."""
    d = adm1.translate_feedstock_to_adm1(**MANURE_KW, chemistry="fisgativa")
    V = d["_V_feed_m3"]
    assert d["S_IC"] == pytest.approx(100.0 * 1.2 / 191.3 / 12.011 / V, rel=1e-9)
    assert d["S_IN"] == pytest.approx(2.3 * 0.8 / 5.2 / adm1.MW_N / V, rel=1e-6)
    assert d["S_cation"] == pytest.approx(100.0 * 0.10 / 191.3 / V, rel=1e-9)
    assert d["S_anion"] == pytest.approx(100.0 * 0.02 / 191.3 / V, rel=1e-9)
    assert d["_feed_chemistry"] == "fisgativa"
    fr = d["_fractions"]
    lip = fr["S_fa"] + fr["X_li"]
    deg = sum(fr[k] for k in ("S_su", "S_aa", "S_fa", "X_ch", "X_pr", "X_li"))
    assert lip / deg == pytest.approx(0.047, rel=1e-6)


def test_fisgativa_at_their_ts_gives_their_inputs(adm1):
    # at the substrate's own TS the scaled inputs equal Table 5 (manure: S_ic 0.10, S_cat 0.10)
    kw = dict(MANURE_KW, TS_target=0.999, feedstock_wet_kg=100.0 / 0.1913)
    d = adm1.translate_feedstock_to_adm1(**kw, chemistry="fisgativa")
    assert d["S_cation"] == pytest.approx(0.10, rel=0.03) and d["S_IC"] == pytest.approx(0.10, rel=0.05)


def test_legacy_is_default_and_unchanged(adm1):
    a = adm1.translate_feedstock_to_adm1(**MANURE_KW)
    b = adm1.translate_feedstock_to_adm1(**MANURE_KW, chemistry="legacy")
    assert a["S_IC"] == b["S_IC"] == pytest.approx(0.01) and a["_feed_chemistry"] == "legacy"
    assert a["S_IN"] == b["S_IN"] and a["S_cation"] == b["S_cation"]
    with pytest.raises(ValueError):
        adm1.translate_feedstock_to_adm1(**MANURE_KW, chemistry="other")
