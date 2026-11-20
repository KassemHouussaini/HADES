"""
ad_characterization.py
======================
Elemental and proximate composition of the digestate leaving the Pathway B
digester, computed from the feed elements, the biogas predicted by ADM1 and
coefficients measured on real digestates. Pure Python, no Brightway.

Per functional unit:
  1. C_dig = C_feed - C in CH4 - C in CO2 - C in bicarbonate.
     Bicarbonate C = 12.011/14.007 x NH4-N: one HCO3- per NH4+ released by
     ammonification (Batstone et al., 2002).
  2. NH4-N from the ADM1 steady state (S_IN x inflow); organic N = N - NH4-N.
  3. H = C_dig x (H/C)_molar and O = C_dig x (O/C)_molar, with molar ratios
     measured on digestates (water consumed by hydrolysis prevents an
     independent H and O balance).
  4. Ash and S are fully retained in the solids.
  5. FC = FC_ratio x (100 - ash %), VM = 100 - ash % - FC.
  Mass: dry digestate = C + H + O + organic N + S + ash + NH4-N;
        wet digestate = wet slurry entering - CH4 - CO2.

References
----------
[1] Zhang, Y., Banks, C.J., Heaven, S. (2012). Anaerobic digestion of two
    biodegradable municipal waste streams. J. Environ. Manage. 104, 166-174.
    Food waste digestate TAN/TKN: 0.43 (fibre), 0.58 (liquor, 98.8 % of the
    digestate); "TAN was around 40-60 % of TKN".
[2] Di Costanzo, N. et al. (2021). Energies 14, 8149. Digested sewage sludge:
    TKN 52-77 and free ammonia 21-45 g/kg TS. No direct TAN/TKN ratio is
    reported, so the sewage value is indicative only (not used in the case study).
[3] Drosg, B. et al. (2015). Nutrient Recovery by Biogas Digestate
    Processing. IEA Bioenergy Task 37, Table 6 (NH4-N/TN = 0.70);
    Czekala, W. (2022). Water 14, 4067 (raw digestates 0.61-0.90).
[4] Molar H/C and O/C (and FC ratio where reported):
      sewage digestate   Aragon-Briceno et al. (2020) Renew. Energy 157,
                         959-973: C 33.3, H 4.6, O 20.3 %db -> 1.65, 0.458
      dairy digestate    Belete et al. (2021) Bioresour. Technol. 333,
                         125164, Table 1, digestate column: C 26.8, H 3.4,
                         O 22.8 %db -> 1.51, 0.638
      food waste         Cao et al. (2019) Waste Manag. 100, 138-150,
                         Table 2, organic household waste digestate:
                         C 34.3, H 4.0, O 23.9 %db -> 1.39, 0.523;
                         FC 14.9 %daf -> FC/(VM+FC) 0.149
[5] FC/(VM+FC): Tsai, C.-H. et al. (2025) Processes 13, 3380 (dairy
    digestate, 0.181); Pawlak-Kruczek, H. et al. (2020) Fuel 275, 117837,
    Table 2 (agricultural digestate 0.308, wet MSW digestate 0.206; FC by
    difference).

No OFMSW-specific digestate data were found; OFMSW uses the food waste
coefficients.
"""

import math

MW = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007}
MW_CH4 = 16.043
MW_CO2 = 44.009

# f_NH4         measured ammonium share of the digestate N, used only when no
#               ADM1 ammonium is passed (NH4_N_kg=None).
# f_NH4_range   measured range; an ADM1 ammonium share outside it is flagged.
# HC_molar,
# OC_molar      molar H/C and O/C of the digestate organic matter [4].
# fc_ratio_*    FC/(VM+FC) of the digestate organic matter [4][5].
AD_CHARACTERIZATION = {
    "sewage_sludge": dict(
        f_NH4=0.51, HC_molar=1.65, OC_molar=0.458, f_NH4_range=(0.27, 0.87),
        fc_ratio_central=0.153, fc_ratio_range=(0.149, 0.206),
        source="f_NH4 [2] (indicative). H/C, O/C, FC ratio: sewage digestate [4]."),
    "dairy_manure": dict(
        f_NH4=0.70, HC_molar=1.51, OC_molar=0.638, f_NH4_range=(0.61, 0.90),
        fc_ratio_central=0.245, fc_ratio_range=(0.181, 0.308),
        source="f_NH4 [3]. H/C, O/C: dairy digestate, Belete et al. 2021 [4]. FC ratio: dairy digestates [5]."),
    "food_waste": dict(
        f_NH4=0.58, HC_molar=1.39, OC_molar=0.523, f_NH4_range=(0.40, 0.60),
        fc_ratio_central=0.149, fc_ratio_range=(0.149, 0.206),
        source=("f_NH4 [1]. H/C, O/C, FC ratio: organic household waste "
                "digestate, Cao et al. 2019 Table 2 [4]; FC range upper "
                "bound: wet MSW digestate [5].")),
}
AD_CHARACTERIZATION["ofmsw"] = dict(AD_CHARACTERIZATION["food_waste"],
                                    source="= food_waste (no OFMSW-specific value found)")


def _position(rng, u):
    """Linear position u in [0, 1] inside rng; None or NaN returns None."""
    if u is None or (isinstance(u, float) and math.isnan(u)):
        return None
    lo, hi = rng
    return lo + min(max(float(u), 0.0), 1.0) * (hi - lo)


def resolve_fc_ratio(feedstock_name, u_fc=None):
    """FC/(VM+FC) of the digestate organic matter (u_fc=None gives the central value)."""
    cfg = AD_CHARACTERIZATION[feedstock_name]
    r = _position(cfg["fc_ratio_range"], u_fc)
    return cfg["fc_ratio_central"] if r is None else r


def proximate_from_fc_ratio(ash_pct, fc_ratio):
    """FC and VM (% of dry matter) from the ash content and FC/(VM+FC).
    Also used by system_builder.renormalize_digestate_after_N_removal."""
    fc = fc_ratio * (100.0 - ash_pct)
    return fc, 100.0 - ash_pct - fc


def adm1_ammonium_kg(adm1_out, V_feed_m3):
    """Ammonium-N leaving the digester per FU: steady-state S_IN (kmol N/m3)
    times the inflow volume per FU (equal to the outflow in a CSTR)."""
    s = (adm1_out or {}).get("ss_state") or {}
    if "S_IN" not in s:
        raise ValueError("adm1_ammonium_kg: ADM1 output has no steady-state S_IN")
    return float(s["S_IN"]) * float(V_feed_m3) * MW["N"]


def digestate_simple(feedstock_name, C, H, O, N, S, ash_kg,
                     CH4_kg, CO2_kg, m_wet_feed_kg, fD=None, u_fc=None,
                     NH4_N_kg=None, TAN_kg=0.0):
    """
    Digestate leaving the digester (steps 1-5 of the module docstring).

    C, H, O, N, S, ash_kg   feed entering the digester (kg/FU, dry basis)
    CH4_kg, CO2_kg          biogas leaving the digester (kg/FU)
    m_wet_feed_kg           wet slurry entering the digester (kg/FU)
    fD                      recorded in the diagnostics only
    u_fc                    position in the FC ratio range (None: central value)
    NH4_N_kg                ammonium-N from the ADM1 steady state
                            (adm1_ammonium_kg). A value outside [0, N] raises
                            an error. None: f_NH4 x N.
    TAN_kg                  ammonium-N carried by the wet feed in addition to
                            its dry-basis N (as ammonium bicarbonate); added to
                            the N and C entering the digester.

    Returns the composition in % of dry matter (C_feed ... FC_feed), the
    moisture WC, the element masses (_C_dig_kg ... _Ash_dig_kg), the wet and
    dry masses, _NH4_N_kg, _S_IN_equiv_kmol_m3, _fc_ratio and diagnostics
    (_dg_*). Warnings are collected in _dg_warnings.
    """
    p = AD_CHARACTERIZATION[feedstock_name]
    warn = []

    # 1. carbon left in the digestate organic matter
    C_biogas = CH4_kg * MW["C"] / MW_CH4 + CO2_kg * MW["C"] / MW_CO2
    if TAN_kg < 0:
        raise ValueError(f"[FEED AMMONIUM] TAN_kg={TAN_kg} < 0")
    _C_card, _N_card = C, N                                  # dry-basis feed values, for VS
    N = N + TAN_kg
    C = C + TAN_kg * MW["C"] / MW["N"]
    if NH4_N_kg is None:
        NH4_N = p["f_NH4"] * N
    else:
        NH4_N = float(NH4_N_kg)
        if not -1e-9 <= NH4_N <= N * (1.0 + 1e-6):
            raise ValueError(f"[FEED AMMONIUM] ADM1 ammonium {NH4_N:.3f} kg outside "
                             f"[0, feed N {N:.3f} kg]")
        NH4_N = min(max(NH4_N, 0.0), N)
        lo, hi = p.get("f_NH4_range", (0.0, 1.0))
        if N > 0 and not lo <= NH4_N / N <= hi:
            warn.append(f"NH4/N {NH4_N / N:.2f} outside measured {lo:.2f}-{hi:.2f}")
    C_bicarb = NH4_N * MW["C"] / MW["N"]
    C_dig = C - C_biogas - C_bicarb
    if C_dig <= 0:
        warn.append(f"C_dig {C_dig:.2f} kg <= 0 (biogas C {C_biogas:.1f} of feed C {C:.1f})")
        C_dig = max(C_dig, 1e-9)

    # 2. nitrogen
    N_org = N - NH4_N

    # 3. hydrogen and oxygen from the measured molar ratios
    n_C = C_dig / MW["C"]
    H_dig = n_C * p["HC_molar"] * MW["H"]
    O_dig = n_C * p["OC_molar"] * MW["O"]

    # 4. masses (ash and S fully retained)
    VS_feed = _C_card + H + O + _N_card + S
    VS_dig = C_dig + H_dig + O_dig + N_org + S
    if VS_dig >= VS_feed:
        warn.append("digestate VS >= feed VS: check H/C, O/C or the biogas")
    kg = dict(C=C_dig, H=H_dig, O=O_dig, N=N_org + NH4_N, S=S, Ash=ash_kg)
    m_dry = sum(kg.values())
    m_wet = max(m_wet_feed_kg - CH4_kg - CO2_kg, 0.0)
    water = max(m_wet - m_dry, 1e-9)

    # 5. proximate analysis
    fc_ratio = resolve_fc_ratio(feedstock_name, u_fc)
    ash_pct = ash_kg / m_dry * 100.0
    FC, VM = proximate_from_fc_ratio(ash_pct, fc_ratio)

    pct = {e: v / m_dry * 100.0 for e, v in kg.items()}
    WC_exact = water / m_wet * 100.0 if m_wet > 0 else 0.0
    return {
        "C_feed": round(pct["C"], 2), "H_feed": round(pct["H"], 2),
        "O_feed": round(pct["O"], 2), "N_feed": round(pct["N"], 2),
        "S_feed": round(pct["S"], 2), "Ash_feed": round(ash_pct, 2),
        "VM_feed": round(VM, 2), "FC_feed": round(FC, 2),
        "WC": round(WC_exact, 2),
        "_WC_exact": WC_exact,
        "_C_dig_kg": kg["C"], "_H_dig_kg": kg["H"], "_O_dig_kg": kg["O"],
        "_N_dig_kg": kg["N"], "_S_dig_kg": kg["S"], "_Ash_dig_kg": kg["Ash"],
        "_m_wet_kg": m_wet, "_m_dry_kg": m_dry,
        "_NH4_N_kg": NH4_N,
        # ammonium concentration in the digestate water (kmol N/m3); 14.0 is
        # the molar mass used by system_builder.split_centrifuge_N
        "_S_IN_equiv_kmol_m3": NH4_N / 14.0 / (water / 1000.0),
        "_fc_ratio": fc_ratio,
        "_dg_fD": fD,
        "_dg_VS_destroyed": 1.0 - VS_dig / VS_feed if VS_feed > 0 else None,
        "_dg_C_biogas_kg": C_biogas,
        "_dg_C_bicarbonate_kg": C_bicarb,
        "_dg_N_NH4_kg": NH4_N,
        "_dg_f_NH4": NH4_N / N if N > 0 else None,
        "_dg_HC_molar": p["HC_molar"],
        "_dg_OC_molar": p["OC_molar"],
        "_dg_fc_ratio": fc_ratio,
        "_dg_ash_solids_pct": ash_kg / (m_dry - NH4_N) * 100.0,
        "_dg_warnings": " | ".join(warn),
    }
