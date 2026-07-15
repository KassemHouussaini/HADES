"""
ADM1_fixed.py — Scientifically grounded ADM1 implementation
============================================================
Supports:
  Pathway A : HTC process water  → AD
  Pathway B : Raw feedstock      → AD  (digestate optionally → HTC)

Modes:
  mode="batch"      — BMP protocol (VDI 4630).
  mode="continuous" — CSTR.

PW-ADM1 extensions (v4):
  [PW-1] Six recalcitrant SI sub-pools replace the two lumped fast/slow pools:
          S_I_furan, S_I_alcohol, S_I_phenol, S_I_acid, S_I_ketone, S_I_humic
          Each leaks into S_su at a first-order rate k_leak derived from
          Zhou et al. (2024b) average removal at 25d HRT.
  [PW-2] Non-competitive inhibition of methanogenesis (I_10, I_11, I_12) by
          all six sub-pools using literature K_I values. Alcohol-like and
          carboxylic acid-like pools assigned K_I = 10 kg COD/m3 (low
          inhibition at PW concentrations; Monlau et al. 2014; Akunna et al. 1993).
  [PW-3] Sub-pool initial concentrations derived from temperature-dependent
          composition table (literature-reconciled, Section 2.X.5) scaled by
          total SI and fast/slow partition calibrated from UASB effluent data.
"""

import numpy as np
import pandas as pd
import scipy.integrate
import math

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, **kw):
            self._it = iterable; self.total = total
        def __iter__(self): return iter(self._it) if self._it else iter([])
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): pass


# ─────────────────────────────────────────────────────────────────────────────
# ATOMIC & MOLECULAR WEIGHT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MW_C   = 12.011
MW_H   =  1.008
MW_O   = 16.000
MW_N   = 14.007
MW_S   = 32.06

MW_CH4    = 16.0
MW_CO2    = 44.0
MW_H2_gas =  2.016
MW_NH3    = 17.031

COD_CH4 = 64.0
COD_ac  = 64.0
COD_pro = 112.0
COD_bu  = 160.0
COD_va  = 208.0

RHO_CH4 = 0.716
RHO_CO2 = 1.977


def compute_thod(C, H, O, N, S):
    thod = 32.0 * (C / MW_C
                   + H / (4.0 * MW_H)
                   - O / (2.0 * MW_O)
                   - 3.0 * N / (4.0 * MW_N)
                   + S / MW_S)
    return max(0.0, float(thod))


def compute_S_IN(N_kg, V_m3, substrate_type="feedstock",
                 T_HTC=180.0, RT_HTC=60.0):
    if substrate_type == "htc_pw":
        R0    = RT_HTC * math.exp((T_HTC - 100.0) / 14.75)
        logR0 = math.log10(max(R0, 1e-9))
        f_IN  = 0.30 + 0.35 / (1.0 + math.exp(-1.5 * (logR0 - 3.8)))
        f_IN  = float(np.clip(f_IN, 0.30, 0.65))
    elif substrate_type == "digestate":
        f_IN  = 0.70
    else:
        f_IN  = 0
    S_IN = max(float(np.clip((N_kg * f_IN) / max(V_m3, 1e-9) / MW_N, 0.0, 0.13)), 1e-4)
    print(f"[compute_S_IN | {substrate_type}] N_kg={N_kg:.4f}  "
          f"f_IN={f_IN:.3f}  S_IN={S_IN:.5f} kmolN/m³")
    return S_IN, f_IN


def compute_S_IC(pH=7.2, T_op_K=308.15, p_CO2_atm=0.0005):
    R     = 8.314e-3
    T_ref = 298.15
    KH    = 3.4e-2 * math.exp(19.5 / R * (1.0 / T_op_K - 1.0 / T_ref))
    Ka1   = 10**-6.35  * math.exp((7.646  / R) * (1.0 / T_ref - 1.0 / T_op_K))
    Ka2   = 10**-10.33 * math.exp((14.85  / R) * (1.0 / T_ref - 1.0 / T_op_K))
    h     = 10**(-pH)
    CO2_aq = KH * p_CO2_atm
    S_IC   = CO2_aq * (1.0 + Ka1/h + Ka1*Ka2/h**2)
    S_IC   = float(np.clip(S_IC, 0.01, 0.20))
    print(f"[compute_S_IC] pH={pH:.2f}  T={T_op_K-273.15:.1f}°C  "
          f"p_CO2={p_CO2_atm:.4f} atm  S_IC={S_IC:.5f} kmolC/m³")
    return S_IC


# ─────────────────────────────────────────────────────────────────────────────
# PW-ADM1 SUB-POOL PARAMETERS  [PW-1, PW-2]
# ─────────────────────────────────────────────────────────────────────────────

PW_ADM1_PARAMS = {
    # k_leak (d-1) — derived from Zhou et al. (2024b) average removal at 25d HRT
    # k_leak = -ln(1 - removal_efficiency) / 25
    "k_leak_furan":   0.1842,   # ~99% removal — fast, readily degraded
    "k_leak_alcohol": 0.1842,   # ~99% removal — fast, non-toxic
    "k_leak_phenol":  0.0848,   # ~88% removal — fast but slower than furans
    "k_leak_acid":    0.1063,   # ~99% removal — fast, non-toxic
    "k_leak_ketone":  0.0277,   # ~50% removal — slow
    "k_leak_humic":   0.004,   # ~55% removal — slow

    # K_I (kg COD/m3) — non-competitive inhibition of methanogenesis
    # Sources: Ghasimi et al. 2016 (furan); Chapleur et al. 2013 (phenol);
    #          Fedorak & Hrudey 1989, Liu et al. 2019 (ketone);
    #          Azman et al. 2018, Li et al. 2021 (humic);
    #          Monlau et al. 2014 (alcohol); Akunna et al. 1993 (acid)
    "K_I_furan":   1.5,
    "K_I_alcohol": 10.0,   # low inhibition at PW concentrations
    "K_I_phenol":  1.0,
    "K_I_acid":    10.0,   # low inhibition at PW concentrations
    "K_I_ketone":  1.9,
    "K_I_humic":   10.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# SI COMPOSITION TABLE  [PW-3]
# Fractional composition of SI across six sub-pools as function of T_HTC.
# Literature-reconciled from published compound-class concentrations,
# normalised to sum to 1.0 at each temperature node (Section 2.X.5).
# ─────────────────────────────────────────────────────────────────────────────

SI_TEMPS = [160, 180, 200, 220, 240, 260, 280]

SI_COMPOSITION = {
    #              160     180     200     220     240     260     280
    "furan":   [0.167,  0.214,  0.155,  0.095,  0.047,  0.024,  0.012],
    "alcohol": [0.238,  0.214,  0.167,  0.119,  0.071,  0.047,  0.035],
    "phenol":  [0.060,  0.071,  0.107,  0.155,  0.188,  0.212,  0.198],
    "acid":    [0.262,  0.238,  0.262,  0.286,  0.318,  0.329,  0.337],
    "ketone":  [0.095,  0.095,  0.119,  0.143,  0.165,  0.176,  0.186],
    "humic":   [0.179,  0.167,  0.190,  0.202,  0.212,  0.212,  0.233],
}

# Sub-pool rate classification (for fast/slow partitioning)
SI_FAST_POOLS = ["furan", "alcohol", "phenol", "acid"]
SI_SLOW_POOLS = ["ketone", "humic"]


def get_si_composition(T_HTC):
    """
    Linear interpolation of SI sub-pool fractions at T_HTC.
    Returns dict of fractions summing to 1.0.
    """
    T = float(np.clip(T_HTC, 160, 280))
    i = int(np.clip(np.searchsorted(SI_TEMPS, T, side='right') - 1,
                    0, len(SI_TEMPS) - 2))
    t0, t1 = SI_TEMPS[i], SI_TEMPS[i + 1]
    alpha = (T - t0) / (t1 - t0)
    fracs = {k: (1 - alpha) * v[i] + alpha * v[i + 1]
             for k, v in SI_COMPOSITION.items()}
    # Normalise to ensure exact closure at interpolated T
    total = sum(fracs.values())
    return {k: v / total for k, v in fracs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID ELEMENTAL → ADM1 FRACTION DERIVATION
# ─────────────────────────────────────────────────────────────────────────────

def derive_adm1_fractions(C, H, O, N, S, ash_kg=0.0,
                           substrate_type="feedstock",
                           T_HTC=180.0):
    m_org = C + H + O + N + S
    if m_org <= 0:
        raise ValueError("derive_adm1_fractions: all elemental masses are zero.")

    X_I_frac = 0.0 if substrate_type == "htc_pw" else 0.08

    if substrate_type == "htc_pw":
        OC_mol   = (O / MW_O) / max(C / MW_C, 1e-9)
        NC_mol   = (N / MW_N) / max(C / MW_C, 1e-9)
        #S_I_frac = float(np.clip(
         #   0.80 - 0.45 * OC_mol + 0.30 * NC_mol + 0.0015 * (T_HTC - 180),
          #  0.15, 0.90))

        S_I_frac = float(np.clip(0.25 + 0.10 * ((T_HTC - 180) / 60), 0.25, 0.50))
        #S_I_frac = 0.2
        print(f"[S_I_frac | htc_pw] OC={OC_mol:.3f}  NC={NC_mol:.3f}  "
              f"T={T_HTC}°C  →  S_I_frac={S_I_frac:.3f}  "
              f"predicted BMP={(1-S_I_frac)*350:.0f} mL/g COD")
    else:
        S_I_frac = 0.02

    biodeg_frac = 1.0 - S_I_frac - X_I_frac

    if substrate_type == "htc_pw":
        sol_frac  = 0.75
        vfa_frac  = 0.05
        sol_biodeg  = sol_frac * biodeg_frac
        part_biodeg = (1.0 - sol_frac) * biodeg_frac
        sol_vfa  = sol_biodeg * vfa_frac
        sol_mono = sol_biodeg * (1.0 - vfa_frac)

        OC_mol = (O / MW_O) / max(C / MW_C, 1e-9)
        f_aa = float(np.clip((N / m_org) / 0.16, 0.0, 0.55))
        f_fa_raw = float(np.clip((1.0 - OC_mol) / (1.0 - 0.1), 0.0, 0.40))
        f_fa = min(f_fa_raw, max(0.0, 0.80 - f_aa))
        f_su = max(1.0 - f_aa - f_fa, 0.05)

        mono_total = f_aa + f_su + f_fa
        f_aa /= mono_total; f_su /= mono_total; f_fa /= mono_total

        ac_vfa = 0.90; pro_vfa = 0.05; bu_vfa = 0.04; va_vfa = 0.01

        fractions = {
            "S_su":  sol_mono * f_su,
            "S_aa":  sol_mono * f_aa,
            "S_fa":  sol_mono * f_fa,
            "S_ac":  sol_vfa  * ac_vfa,
            "S_pro": sol_vfa  * pro_vfa,
            "S_bu":  sol_vfa  * bu_vfa,
            "S_va":  sol_vfa  * va_vfa,
            "S_I":   S_I_frac,
            "X_ch":  part_biodeg * f_su,
            "X_pr":  part_biodeg * f_aa,
            "X_li":  part_biodeg * f_fa,
            "X_xc":  0.0,
            "X_I":   X_I_frac,
        }
    else:
        protein_frac = float(np.clip((N / m_org) / 0.16, 0.0, 0.55))
        OC_mol = (O / MW_O) / max(C / MW_C, 1e-9)
        lipid_raw = float(np.clip((1.0 - OC_mol) / (1.0 - 0.1), 0.0, 0.40))
        lipid_frac = min(lipid_raw, max(0.0, 0.80 - protein_frac))
        carb_frac = max(1.0 - protein_frac - lipid_frac, 0.05)

        bio_total = protein_frac + lipid_frac + carb_frac
        protein_frac /= bio_total; lipid_frac /= bio_total; carb_frac /= bio_total

        sol_frac    = 0.27
        part_frac   = 1.0 - sol_frac
        sol_biodeg  = sol_frac  * biodeg_frac
        part_biodeg = part_frac * biodeg_frac

        fractions = {
            "S_su":  sol_biodeg  * carb_frac,
            "S_aa":  sol_biodeg  * protein_frac,
            "S_fa":  sol_biodeg  * lipid_frac,
            "S_ac":  0.0, "S_pro": 0.0, "S_bu":  0.0, "S_va":  0.0,
            "S_I":   S_I_frac,
            "X_ch":  part_biodeg * carb_frac,
            "X_pr":  part_biodeg * protein_frac,
            "X_li":  part_biodeg * lipid_frac,
            "X_xc":  0.0,
            "X_I":   X_I_frac,
        }

    frac_sum = sum(fractions.values())
    assert abs(frac_sum - 1.0) < 1e-6, \
        f"derive_adm1_fractions: fractions sum to {frac_sum:.6f}, expected 1.0"
    return fractions


def _print_fractions(fractions, substrate_type, C, H, O, N, S):
    m_org  = C + H + O + N + S
    HC_mol = (H / MW_H) / max(C / MW_C, 1e-9)
    OC_mol = (O / MW_O) / max(C / MW_C, 1e-9)
    print(f"[derive_fractions | {substrate_type}]")
    print(f"  Elemental ratios: H/C={HC_mol:.3f}  O/C={OC_mol:.3f}  "
          f"N/org={N/m_org*100:.1f}%")
    print(f"  Fractions: " +
          "  ".join(f"{k}={v:.3f}" for k, v in fractions.items() if v > 0.001))


# ─────────────────────────────────────────────────────────────────────────────
# BSM2 BIOMASS DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

BSM2_BIOMASS_FRACTIONS = {
    "X_su": 0.18, "X_aa": 0.14, "X_fa": 0.08,
    "X_c4": 0.16, "X_pro": 0.13, "X_ac": 0.26, "X_h2": 0.05,
}
BSM2_SS_BIOMASS        = 2.43
BSM2_COLD_SEED_BIOMASS = 3.0


# ─────────────────────────────────────────────────────────────────────────────
# CHARGE-BALANCE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def compute_charge_pair(S_IC, S_IN, S_va, S_bu, S_pro, S_ac,
                        pH_target=7.2, T_op=308.15):
    R = 0.083145; T_base = 298.15
    Ka_va  = 10**-4.86; Ka_bu  = 10**-4.82
    Ka_pro = 10**-4.88; Ka_ac  = 10**-4.76
    Ka_co2 = 10**-6.35 * np.exp((7646  / (100*R)) * (1/T_base - 1/T_op))
    Ka_IN  = 10**-9.25 * np.exp((51965 / (100*R)) * (1/T_base - 1/T_op))
    Kw     = 10**-14   * np.exp((55900 / (100*R)) * (1/T_base - 1/T_op))
    h = 10**(-pH_target)
    S_va_ion  = Ka_va  * S_va  / (Ka_va  + h)
    S_bu_ion  = Ka_bu  * S_bu  / (Ka_bu  + h)
    S_pro_ion = Ka_pro * S_pro / (Ka_pro + h)
    S_ac_ion  = Ka_ac  * S_ac  / (Ka_ac  + h)
    S_hco3    = Ka_co2 * S_IC  / (Ka_co2 + h)
    S_nh3     = Ka_IN  * S_IN  / (Ka_IN  + h)
    net = (S_hco3
           + S_ac_ion  / COD_ac + S_pro_ion / COD_pro
           + S_bu_ion  / COD_bu + S_va_ion  / COD_va
           + Kw / h - (S_IN - S_nh3) - h)
    net_capped = float(np.clip(net, -0.20, 0.20))
    base = 0.04
    S_cation = base + max(net_capped, 0.0)
    S_anion  = base - min(net_capped, 0.0)
    return S_cation, S_anion


# ─────────────────────────────────────────────────────────────────────────────
# BSM2 PHYSICO-CHEMICAL / KINETIC CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

def _build_constants(T_op=308.15):
    R = 0.083145; T_base = 298.15
    c = dict(
        R=R, T_base=T_base, T_op=T_op, p_atm=1.013,
        f_sI_xc=0.1, f_xI_xc=0.2, f_ch_xc=0.2, f_pr_xc=0.2, f_li_xc=0.3,
        N_xc=0.0376/14, N_I=0.06/14, N_aa=0.007, N_bac=0.08/14,
        C_xc=0.02786, C_sI=0.03,  C_ch=0.0313, C_pr=0.03,
        C_li=0.022,   C_xI=0.03,  C_su=0.0313, C_aa=0.03,
        C_fa=0.0217,  C_bu=0.025, C_pro=0.0268, C_ac=0.0313,
        C_bac=0.0313, C_va=0.024, C_ch4=0.0156,
        f_fa_li=0.95,
        f_h2_su=0.19, f_bu_su=0.13, f_pro_su=0.27, f_ac_su=0.41,
        f_h2_aa=0.06, f_va_aa=0.23, f_bu_aa=0.26, f_pro_aa=0.05, f_ac_aa=0.40,
        Y_su=0.10, Y_aa=0.08, Y_fa=0.06, Y_c4=0.06,
        Y_pro=0.04, Y_ac=0.05, Y_h2=0.06,
        k_A_B_va=1e10, k_A_B_bu=1e10, k_A_B_pro=1e10,
        k_A_B_ac=1e10, k_A_B_co2=1e10, k_A_B_IN=1e10,
        k_dis=0.5, k_hyd_ch=10.0, k_hyd_pr=10.0, k_hyd_li=10.0,
        K_S_IN=1e-4,
        k_m_su=30.0,  K_S_su=0.5,   pH_UL_aa=5.5, pH_LL_aa=4.0,
        k_m_aa=50.0,  K_S_aa=0.3,
        k_m_fa=6.0,   K_S_fa=0.4,   K_I_h2_fa=5e-6,
        k_m_c4=20.0,  K_S_c4=0.2,   K_I_h2_c4=1e-5,
        k_m_pro=13.0, K_S_pro=0.1,  K_I_h2_pro=3.5e-6,
        k_m_ac=8.0,   K_S_ac=0.15,  K_I_nh3=0.0018,
        pH_UL_ac=7.0, pH_LL_ac=6.0,
        k_m_h2=35.0,  K_S_h2=7e-6,  pH_UL_h2=6.0, pH_LL_h2=5.0,
        k_dec_X_su=0.02, k_dec_X_aa=0.02, k_dec_X_fa=0.02,
        k_dec_X_c4=0.02, k_dec_X_pro=0.02, k_dec_X_ac=0.02, k_dec_X_h2=0.02,
        k_L_a=200.0, k_p=5e4,
    )
    c["K_w"]     = 1e-14   * np.exp((55900 / (100*R)) * (1/T_base - 1/T_op))
    c["K_a_va"]  = 10**-4.86
    c["K_a_bu"]  = 10**-4.82
    c["K_a_pro"] = 10**-4.88
    c["K_a_ac"]  = 10**-4.76
    c["K_a_co2"] = 10**-6.35 * np.exp((7646  / (100*R)) * (1/T_base - 1/T_op))
    c["K_a_IN"]  = 10**-9.25 * np.exp((51965 / (100*R)) * (1/T_base - 1/T_op))
    c["p_gas_h2o"] = 0.0313  * np.exp(5290   * (1/T_base - 1/T_op))
    c["K_H_co2"] = 0.035     * np.exp((-19410/ (100*R)) * (1/T_base - 1/T_op))
    c["K_H_ch4"] = 0.0014    * np.exp((-14240/ (100*R)) * (1/T_base - 1/T_op))
    c["K_H_h2"]  = 7.8e-4   * np.exp((-4180 / (100*R)) * (1/T_base - 1/T_op))
    c["K_pH_aa"] = 10**-((c["pH_LL_aa"] + c["pH_UL_aa"]) / 2.0)
    c["nn_aa"]   = 3.0 / (c["pH_UL_aa"] - c["pH_LL_aa"])
    c["K_pH_ac"] = 10**-((c["pH_LL_ac"] + c["pH_UL_ac"]) / 2.0)
    c["n_ac"]    = 3.0 / (c["pH_UL_ac"] - c["pH_LL_ac"])
    c["K_pH_h2"] = 10**-((c["pH_LL_h2"] + c["pH_UL_h2"]) / 2.0)
    c["n_h2"]    = 3.0 / (c["pH_UL_h2"] - c["pH_LL_h2"])
    return c


# ─────────────────────────────────────────────────────────────────────────────
# DAE SOLVER
# ─────────────────────────────────────────────────────────────────────────────

def _dae_solve(sv, c):
    S_IC     = sv["S_IC"];      S_IN     = sv["S_IN"]
    S_H_ion  = sv["S_H_ion"];   S_h2     = sv["S_h2"]
    S_cation = sv["S_cation"];  S_anion  = sv["S_anion"]
    S_gas_h2 = sv["S_gas_h2"]

    Ka_va  = c["K_a_va"];  Ka_bu  = c["K_a_bu"]; Ka_pro = c["K_a_pro"]
    Ka_ac  = c["K_a_ac"];  Ka_co2 = c["K_a_co2"]; Ka_IN = c["K_a_IN"]
    Kw     = c["K_w"];     R      = c["R"];         T_op  = c["T_op"]
    K_pH_aa = c["K_pH_aa"]; nn_aa = c["nn_aa"]
    K_pH_h2 = c["K_pH_h2"]; n_h2  = c["n_h2"]
    K_S_IN  = c["K_S_IN"];  K_I_h2_fa  = c["K_I_h2_fa"]
    K_I_h2_c4 = c["K_I_h2_c4"]; K_I_h2_pro = c["K_I_h2_pro"]
    k_m_su  = c["k_m_su"];  K_S_su  = c["K_S_su"]
    k_m_aa  = c["k_m_aa"];  K_S_aa  = c["K_S_aa"]
    k_m_fa  = c["k_m_fa"];  K_S_fa  = c["K_S_fa"]
    k_m_c4  = c["k_m_c4"];  K_S_c4  = c["K_S_c4"]
    k_m_pro = c["k_m_pro"]; K_S_pro = c["K_S_pro"]
    k_m_h2  = c["k_m_h2"];  K_S_h2  = c["K_S_h2"]
    Y_su=c["Y_su"]; Y_aa=c["Y_aa"]; Y_fa=c["Y_fa"]; Y_c4=c["Y_c4"]
    Y_pro=c["Y_pro"]
    f_h2_su=c["f_h2_su"]; f_h2_aa=c["f_h2_aa"]
    k_L_a = c["k_L_a"]; K_H_h2 = c["K_H_h2"]

    tol = 1e-12; maxIter = 1000; eps = 1e-7
    S_va_ion = S_bu_ion = S_pro_ion = S_ac_ion = S_hco3_ion = S_nh3 = 0.0

    for _ in range(maxIter):
        S_va_ion   = Ka_va  * sv["S_va"]  / (Ka_va  + S_H_ion)
        S_bu_ion   = Ka_bu  * sv["S_bu"]  / (Ka_bu  + S_H_ion)
        S_pro_ion  = Ka_pro * sv["S_pro"] / (Ka_pro + S_H_ion)
        S_ac_ion   = Ka_ac  * sv["S_ac"]  / (Ka_ac  + S_H_ion)
        S_hco3_ion = Ka_co2 * S_IC        / (Ka_co2 + S_H_ion)
        S_nh3      = Ka_IN  * S_IN        / (Ka_IN  + S_H_ion)
        delta = (S_cation + (S_IN - S_nh3) + S_H_ion
                 - S_hco3_ion
                 - S_ac_ion  / COD_ac
                 - S_pro_ion / COD_pro
                 - S_bu_ion  / COD_bu
                 - S_va_ion  / COD_va
                 - Kw / S_H_ion
                 - S_anion)
        grad = (1.0
                + Ka_IN  * S_IN  / (Ka_IN  + S_H_ion)**2
                + Ka_co2 * S_IC  / (Ka_co2 + S_H_ion)**2
                + Ka_ac  * sv["S_ac"]  / (Ka_ac  + S_H_ion)**2 / COD_ac
                + Ka_pro * sv["S_pro"] / (Ka_pro + S_H_ion)**2 / COD_pro
                + Ka_bu  * sv["S_bu"]  / (Ka_bu  + S_H_ion)**2 / COD_bu
                + Ka_va  * sv["S_va"]  / (Ka_va  + S_H_ion)**2 / COD_va
                + Kw / S_H_ion**2)
        S_H_ion -= delta / grad
        S_H_ion = float(np.clip(S_H_ion, 1e-10, 1e-4))
        if abs(delta) <= tol:
            break

    pH = -np.log10(max(S_H_ion, 1e-14))
    I_pH_aa = (K_pH_aa**nn_aa) / (S_H_ion**nn_aa + K_pH_aa**nn_aa)
    I_pH_h2 = (K_pH_h2**n_h2)  / (S_H_ion**n_h2  + K_pH_h2**n_h2)
    I_IN    = 1.0 / (1.0 + K_S_IN / max(S_IN, 1e-12))

    Xsu = sv["X_su"]; Xaa = sv["X_aa"]; Xfa = sv["X_fa"]
    Xc4 = sv["X_c4"]; Xpro = sv["X_pro"]; Xh2 = sv["X_h2"]
    Ssu = sv["S_su"]; Saa = sv["S_aa"]; Sfa = sv["S_fa"]
    Sva = sv["S_va"]; Sbu = sv["S_bu"]; Spro = sv["S_pro"]

    for _ in range(maxIter):
        Ih2fa  = 1.0 / (1.0 + S_h2 / K_I_h2_fa)
        Ih2c4  = 1.0 / (1.0 + S_h2 / K_I_h2_c4)
        Ih2pro = 1.0 / (1.0 + S_h2 / K_I_h2_pro)
        R5  = k_m_su  * Ssu  / (K_S_su  + Ssu)  * Xsu  * I_pH_aa * I_IN
        R6  = k_m_aa  * Saa  / (K_S_aa  + Saa)  * Xaa  * I_pH_aa * I_IN
        R7  = k_m_fa  * Sfa  / (K_S_fa  + Sfa)  * Xfa  * I_pH_aa * I_IN * Ih2fa
        R8  = k_m_c4  * Sva  / (K_S_c4  + Sva)  * Xc4  * (Sva / (Sbu + Sva + eps)) * I_pH_aa * I_IN * Ih2c4
        R9  = k_m_c4  * Sbu  / (K_S_c4  + Sbu)  * Xc4  * (Sbu / (Sbu + Sva + eps)) * I_pH_aa * I_IN * Ih2c4
        R10 = k_m_pro * Spro / (K_S_pro + Spro) * Xpro * I_pH_aa * I_IN * Ih2pro
        R12 = k_m_h2  * S_h2 / (K_S_h2  + S_h2) * Xh2  * I_pH_h2 * I_IN
        p_h2 = S_gas_h2 * R * T_op / MW_H2_gas
        RT8  = k_L_a * (S_h2 - MW_H2_gas * K_H_h2 * p_h2)
        delta = ((1-Y_su)*f_h2_su*R5 + (1-Y_aa)*f_h2_aa*R6
                 + (1-Y_fa)*0.3*R7 + (1-Y_c4)*0.15*R8
                 + (1-Y_c4)*0.2*R9 + (1-Y_pro)*0.43*R10
                 - R12 - RT8)
        grad = (- k_m_h2 / (K_S_h2 + S_h2) * Xh2 * I_pH_h2 * I_IN
                + k_m_h2 * S_h2 / (K_S_h2 + S_h2)**2 * Xh2 * I_pH_h2 * I_IN
                - k_L_a
                - (1-Y_fa)*0.3  * k_m_fa  * Sfa  / (K_S_fa  + Sfa)  * Xfa  * I_pH_aa * I_IN / (1 + S_h2/K_I_h2_fa)**2  / K_I_h2_fa
                - (1-Y_c4)*0.15 * k_m_c4  * Sva  / (K_S_c4  + Sva)  * Xc4  * (Sva/(Sbu+Sva+eps)) * I_pH_aa * I_IN / (1 + S_h2/K_I_h2_c4)**2  / K_I_h2_c4
                - (1-Y_c4)*0.2  * k_m_c4  * Sbu  / (K_S_c4  + Sbu)  * Xc4  * (Sbu/(Sbu+Sva+eps)) * I_pH_aa * I_IN / (1 + S_h2/K_I_h2_c4)**2  / K_I_h2_c4
                - (1-Y_pro)*0.43* k_m_pro * Spro / (K_S_pro + Spro) * Xpro * I_pH_aa * I_IN / (1 + S_h2/K_I_h2_pro)**2 / K_I_h2_pro)
        S_h2 -= delta / (grad if abs(grad) > 1e-20 else 1e-20)
        if S_h2 <= 0:
            S_h2 = tol
        if abs(delta) <= tol:
            break

    return {
        "S_H_ion": S_H_ion, "S_h2": S_h2, "pH": pH,
        "S_va_ion": S_va_ion, "S_bu_ion": S_bu_ion,
        "S_pro_ion": S_pro_ion, "S_ac_ion": S_ac_ion,
        "S_hco3_ion": S_hco3_ion, "S_nh3": S_nh3,
        "S_nh4_ion": max(S_IN - S_nh3, 0.0),
        "S_co2":     max(S_IC - S_hco3_ion, 0.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STATE VECTOR DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

STATE_NAMES_BASE = [
    "S_su","S_aa","S_fa","S_va","S_bu","S_pro","S_ac","S_h2","S_ch4",
    "S_IC","S_IN","S_I",
    "X_xc","X_ch","X_pr","X_li",
    "X_su","X_aa","X_fa","X_c4","X_pro","X_ac","X_h2","X_I",
    "S_cation","S_anion",
    "S_H_ion","S_va_ion","S_bu_ion","S_pro_ion","S_ac_ion",
    "S_hco3_ion","S_co2","S_nh3","S_nh4_ion",
    "S_gas_h2","S_gas_ch4","S_gas_co2",
]

# [PW-1] Six SI sub-pools replace the two lumped fast/slow pools.
# All six are differential state variables (ODE).
PW_EXTRA_STATES = [
    "S_I_furan", "S_I_alcohol", "S_I_phenol",
    "S_I_acid",  "S_I_ketone",  "S_I_humic",
]

# All sub-pool names (convenient list for iteration)
SI_SUBPOOL_NAMES = ["furan", "alcohol", "phenol", "acid", "ketone", "humic"]

STATE_NAMES = STATE_NAMES_BASE + PW_EXTRA_STATES
IDX = {k: i for i, k in enumerate(STATE_NAMES)}

BIOMASS_KEYS   = ["X_su","X_aa","X_fa","X_c4","X_pro","X_ac","X_h2"]
SUBSTRATE_KEYS = ["S_su","S_aa","S_fa","S_va","S_bu","S_pro","S_ac",
                   "S_IC","S_IN","S_I","X_xc","X_ch","X_pr","X_li","X_I"]
PW_EXTRA_KEYS  = PW_EXTRA_STATES


# ─────────────────────────────────────────────────────────────────────────────
# ODE RIGHT-HAND SIDE
# ─────────────────────────────────────────────────────────────────────────────

def _build_ode(c, V_liq, V_gas, STATE_NAMES, idx_map, mode, influent, q_ad,
               pw_mode=False):
    """
    Return ODE callable ode(t, y).

    pw_mode=True activates PW-ADM1 extensions [PW-1, PW-2, PW-3]:
      [PW-1] Six SI sub-pools each leak into S_su at first-order rate k_leak.
      [PW-2] All six sub-pools exert non-competitive inhibition on
             methanogenesis (I_10, I_11, I_12) via multiplicative K_I terms.
             Alcohol and acid pools: K_I = 10 kg COD/m3 (minor inhibition).
      Acidogens (I_5 to I_9) are unaffected — consistent with literature
      showing HTC inhibitors primarily target methanogens.
    pw_mode=False (Pathway B) = standard BSM2 ODE, zero overhead.
    """
    I = idx_map.__getitem__

    R=c["R"]; T_op=c["T_op"]; p_atm=c["p_atm"]
    f_sI_xc=c["f_sI_xc"]; f_xI_xc=c["f_xI_xc"]; f_ch_xc=c["f_ch_xc"]
    f_pr_xc=c["f_pr_xc"]; f_li_xc=c["f_li_xc"]
    N_xc=c["N_xc"]; N_I=c["N_I"]; N_aa=c["N_aa"]; N_bac=c["N_bac"]
    C_xc=c["C_xc"]; C_sI=c["C_sI"]; C_ch=c["C_ch"]; C_pr=c["C_pr"]
    C_li=c["C_li"]; C_xI=c["C_xI"]; C_su=c["C_su"]; C_aa=c["C_aa"]
    C_fa=c["C_fa"]; C_bu=c["C_bu"]; C_pro=c["C_pro"]; C_ac=c["C_ac"]
    C_bac=c["C_bac"]; C_va=c["C_va"]; C_ch4=c["C_ch4"]; f_fa_li=c["f_fa_li"]
    f_h2_su=c["f_h2_su"]; f_bu_su=c["f_bu_su"]; f_pro_su=c["f_pro_su"]; f_ac_su=c["f_ac_su"]
    f_h2_aa=c["f_h2_aa"]; f_va_aa=c["f_va_aa"]; f_bu_aa=c["f_bu_aa"]
    f_pro_aa=c["f_pro_aa"]; f_ac_aa=c["f_ac_aa"]
    Y_su=c["Y_su"]; Y_aa=c["Y_aa"]; Y_fa=c["Y_fa"]; Y_c4=c["Y_c4"]
    Y_pro=c["Y_pro"]; Y_ac=c["Y_ac"]; Y_h2=c["Y_h2"]
    k_A_B_va=c["k_A_B_va"]; k_A_B_bu=c["k_A_B_bu"]; k_A_B_pro=c["k_A_B_pro"]
    k_A_B_ac=c["k_A_B_ac"]; k_A_B_co2=c["k_A_B_co2"]; k_A_B_IN=c["k_A_B_IN"]
    k_dis=c["k_dis"]; k_hyd_ch=c["k_hyd_ch"]; k_hyd_pr=c["k_hyd_pr"]
    k_hyd_li=c["k_hyd_li"]; K_S_IN=c["K_S_IN"]
    k_m_su=c["k_m_su"]; K_S_su=c["K_S_su"]; K_pH_aa=c["K_pH_aa"]; nn_aa=c["nn_aa"]
    k_m_aa=c["k_m_aa"]; K_S_aa=c["K_S_aa"]
    k_m_fa=c["k_m_fa"]; K_S_fa=c["K_S_fa"]; K_I_h2_fa=c["K_I_h2_fa"]
    k_m_c4=c["k_m_c4"]; K_S_c4=c["K_S_c4"]; K_I_h2_c4=c["K_I_h2_c4"]
    k_m_pro=c["k_m_pro"]; K_S_pro=c["K_S_pro"]; K_I_h2_pro=c["K_I_h2_pro"]
    k_m_ac=c["k_m_ac"]; K_S_ac=c["K_S_ac"]; K_I_nh3=c["K_I_nh3"]
    K_pH_ac=c["K_pH_ac"]; n_ac=c["n_ac"]
    k_m_h2=c["k_m_h2"]; K_S_h2=c["K_S_h2"]; K_pH_h2=c["K_pH_h2"]; n_h2=c["n_h2"]
    k_dec_X_su=c["k_dec_X_su"]; k_dec_X_aa=c["k_dec_X_aa"]; k_dec_X_fa=c["k_dec_X_fa"]
    k_dec_X_c4=c["k_dec_X_c4"]; k_dec_X_pro=c["k_dec_X_pro"]
    k_dec_X_ac=c["k_dec_X_ac"]; k_dec_X_h2=c["k_dec_X_h2"]
    k_L_a=c["k_L_a"]; K_H_co2=c["K_H_co2"]; K_H_ch4=c["K_H_ch4"]; K_H_h2=c["K_H_h2"]
    p_gas_h2o=c["p_gas_h2o"]; k_p=c["k_p"]
    Ka_va=c["K_a_va"]; Ka_bu=c["K_a_bu"]; Ka_pro=c["K_a_pro"]
    Ka_ac=c["K_a_ac"]; Ka_co2=c["K_a_co2"]; Ka_IN=c["K_a_IN"]

    # Pre-extract PW k_leak and K_I values once at ODE build time (not per call)
    if pw_mode:
        kl_furan   = PW_ADM1_PARAMS["k_leak_furan"]
        kl_alcohol = PW_ADM1_PARAMS["k_leak_alcohol"]
        kl_phenol  = PW_ADM1_PARAMS["k_leak_phenol"]
        kl_acid    = PW_ADM1_PARAMS["k_leak_acid"]
        kl_ketone  = PW_ADM1_PARAMS["k_leak_ketone"]
        kl_humic   = PW_ADM1_PARAMS["k_leak_humic"]
        KI_furan   = PW_ADM1_PARAMS["K_I_furan"]
        KI_alcohol = PW_ADM1_PARAMS["K_I_alcohol"]
        KI_phenol  = PW_ADM1_PARAMS["K_I_phenol"]
        KI_acid    = PW_ADM1_PARAMS["K_I_acid"]
        KI_ketone  = PW_ADM1_PARAMS["K_I_ketone"]
        KI_humic   = PW_ADM1_PARAMS["K_I_humic"]

    if mode == "continuous":
        D = q_ad / V_liq
        def _get(inf_key, fallback, default=0.0):
            if inf_key in influent: return influent[inf_key]
            return influent.get(fallback, default)
        in_su   = _get("_inf_S_su",   "S_su",   0.0)
        in_aa   = _get("_inf_S_aa",   "S_aa",   0.0)
        in_fa   = _get("_inf_S_fa",   "S_fa",   0.0)
        in_va   = _get("_inf_S_va",   "S_va",   0.0)
        in_bu   = _get("_inf_S_bu",   "S_bu",   0.0)
        in_pro  = _get("_inf_S_pro",  "S_pro",  0.0)
        in_ac   = _get("_inf_S_ac",   "S_ac",   0.0)
        in_h2   = _get("_inf_S_h2",   "S_h2",   1e-8)
        in_ch4  = _get("_inf_S_ch4",  "S_ch4",  1e-8)
        in_IC   = _get("_inf_S_IC",   "S_IC",   0.0)
        in_IN   = _get("_inf_S_IN",   "S_IN",   0.0)
        in_SI   = _get("_inf_S_I",    "S_I",    0.0)
        in_Xxc  = _get("_inf_X_xc",   "X_xc",   0.0)
        in_Xch  = _get("_inf_X_ch",   "X_ch",   0.0)
        in_Xpr  = _get("_inf_X_pr",   "X_pr",   0.0)
        in_Xli  = _get("_inf_X_li",   "X_li",   0.0)
        in_Xsu  = _get("_inf_X_su",   "X_su",   0.0)
        in_Xaa  = _get("_inf_X_aa",   "X_aa",   0.0)
        in_Xfa  = _get("_inf_X_fa",   "X_fa",   0.0)
        in_Xc4  = _get("_inf_X_c4",   "X_c4",   0.0)
        in_Xpro = _get("_inf_X_pro",  "X_pro",  0.0)
        in_Xac  = _get("_inf_X_ac",   "X_ac",   0.0)
        in_Xh2  = _get("_inf_X_h2",   "X_h2",   0.0)
        in_XI   = _get("_inf_X_I",    "X_I",    0.0)
        in_cat  = _get("_inf_S_cation","S_cation",0.06)
        in_an   = _get("_inf_S_anion", "S_anion", 0.04)
        # PW sub-pool influent concentrations
        if pw_mode:
            in_SI_furan   = _get("_inf_S_I_furan",   "S_I_furan",   0.0)
            in_SI_alcohol = _get("_inf_S_I_alcohol",  "S_I_alcohol", 0.0)
            in_SI_phenol  = _get("_inf_S_I_phenol",   "S_I_phenol",  0.0)
            in_SI_acid    = _get("_inf_S_I_acid",     "S_I_acid",    0.0)
            in_SI_ketone  = _get("_inf_S_I_ketone",   "S_I_ketone",  0.0)
            in_SI_humic   = _get("_inf_S_I_humic",    "S_I_humic",   0.0)
    else:
        D = 0.0

    def ode(t, y):
        S_su  = y[I("S_su")];  S_aa  = y[I("S_aa")];  S_fa   = y[I("S_fa")]
        S_va  = y[I("S_va")];  S_bu  = y[I("S_bu")];  S_pro  = y[I("S_pro")]
        S_ac  = y[I("S_ac")];  S_h2  = y[I("S_h2")];  S_ch4  = y[I("S_ch4")]
        S_IC  = y[I("S_IC")];  S_IN  = y[I("S_IN")];  S_I    = y[I("S_I")]
        X_xc  = y[I("X_xc")];  X_ch  = y[I("X_ch")];  X_pr   = y[I("X_pr")]
        X_li  = y[I("X_li")];  X_su  = y[I("X_su")];  X_aa   = y[I("X_aa")]
        X_fa  = y[I("X_fa")];  X_c4  = y[I("X_c4")];  X_pro  = y[I("X_pro")]
        X_ac  = y[I("X_ac")];  X_h2  = y[I("X_h2")];  X_I    = y[I("X_I")]
        S_cation   = y[I("S_cation")];  S_anion    = y[I("S_anion")]
        S_H_ion    = y[I("S_H_ion")];   S_hco3_ion = y[I("S_hco3_ion")]
        S_va_ion   = y[I("S_va_ion")];  S_bu_ion   = y[I("S_bu_ion")]
        S_pro_ion  = y[I("S_pro_ion")]; S_ac_ion   = y[I("S_ac_ion")]
        S_nh3      = y[I("S_nh3")];     S_gas_h2   = y[I("S_gas_h2")]
        S_gas_ch4  = y[I("S_gas_ch4")]; S_gas_co2  = y[I("S_gas_co2")]
        S_co2 = max(S_IC - S_hco3_ion, 0.0)

        # ── [PW-1] Read six SI sub-pool concentrations ────────────────────
        if pw_mode:
            S_I_furan   = max(y[I("S_I_furan")],   0.0)
            S_I_alcohol = max(y[I("S_I_alcohol")],  0.0)
            S_I_phenol  = max(y[I("S_I_phenol")],   0.0)
            S_I_acid    = max(y[I("S_I_acid")],     0.0)
            S_I_ketone  = max(y[I("S_I_ketone")],   0.0)
            S_I_humic   = max(y[I("S_I_humic")],    0.0)

            # First-order leak of each sub-pool into S_su
            leak_furan   = kl_furan   * S_I_furan
            leak_alcohol = kl_alcohol * S_I_alcohol
            leak_phenol  = kl_phenol  * S_I_phenol
            leak_acid    = kl_acid    * S_I_acid
            leak_ketone  = kl_ketone  * S_I_ketone
            leak_humic   = kl_humic   * S_I_humic
            Rho_leak_su  = (leak_furan + leak_alcohol + leak_phenol +
                            leak_acid  + leak_ketone  + leak_humic)

            # ── [PW-2] Non-competitive inhibition on methanogenesis ────────
            # All six sub-pools contribute multiplicatively.
            # Alcohol (K_I=10) and acid (K_I=10): minor but non-zero effect.
            # Applied only to I_10 (propionate), I_11 (acetate), I_12 (H2).
            I_tox = (KI_furan   / (KI_furan   + S_I_furan)  *
                     KI_alcohol / (KI_alcohol + S_I_alcohol) *
                     KI_phenol  / (KI_phenol  + S_I_phenol)  *
                     KI_acid    / (KI_acid    + S_I_acid)    *
                     KI_ketone  / (KI_ketone  + S_I_ketone)  *
                     KI_humic   / (KI_humic   + S_I_humic))
        else:
            S_I_furan = S_I_alcohol = S_I_phenol = 0.0
            S_I_acid  = S_I_ketone  = S_I_humic  = 0.0
            leak_furan = leak_alcohol = leak_phenol = 0.0
            leak_acid  = leak_ketone  = leak_humic  = 0.0
            Rho_leak_su = 0.0
            I_tox = 1.0

        # ── Standard BSM2 inhibition terms ────────────────────────────────
        I_pH_aa  = (K_pH_aa**nn_aa) / (S_H_ion**nn_aa + K_pH_aa**nn_aa)
        I_pH_ac  = (K_pH_ac**n_ac)  / (S_H_ion**n_ac  + K_pH_ac**n_ac)
        I_pH_h2  = (K_pH_h2**n_h2)  / (S_H_ion**n_h2  + K_pH_h2**n_h2)
        I_IN_lim = 1.0 / (1.0 + K_S_IN / max(S_IN, 1e-12))
        I_h2_fa  = 1.0 / (1.0 + S_h2 / K_I_h2_fa)
        I_h2_c4  = 1.0 / (1.0 + S_h2 / K_I_h2_c4)
        I_h2_pro = 1.0 / (1.0 + S_h2 / K_I_h2_pro)
        I_nh3    = 1.0 / (1.0 + S_nh3 / K_I_nh3)

        # ── [PW-2] I_tox applied selectively to methanogens only ──────────
        # I_5  to I_9  (acidogens, LCFA, C4): standard BSM2, no I_tox
        # I_10 to I_12 (propionate, acetate, H2 methanogens): multiplied by I_tox
        I_5  = I_pH_aa * I_IN_lim
        I_6  = I_5
        I_7  = I_pH_aa * I_IN_lim * I_h2_fa
        I_8  = I_pH_aa * I_IN_lim * I_h2_c4
        I_9  = I_8
        I_10 = I_pH_aa * I_IN_lim * I_h2_pro * I_tox
        I_11 = I_pH_ac * I_IN_lim * I_nh3    * I_tox
        I_12 = I_pH_h2 * I_IN_lim             * I_tox

        Rho_1  = k_dis    * X_xc
        Rho_2  = k_hyd_ch * X_ch
        Rho_3  = k_hyd_pr * X_pr
        Rho_4  = k_hyd_li * X_li
        Rho_5  = k_m_su  * S_su  / (K_S_su  + S_su)  * X_su  * I_5
        Rho_6  = k_m_aa  * S_aa  / (K_S_aa  + S_aa)  * X_aa  * I_6
        Rho_7  = k_m_fa  * S_fa  / (K_S_fa  + S_fa)  * X_fa  * I_7
        Rho_8  = k_m_c4  * S_va  / (K_S_c4  + S_va)  * X_c4  * (S_va / (S_bu + S_va + 1e-6)) * I_8
        Rho_9  = k_m_c4  * S_bu  / (K_S_c4  + S_bu)  * X_c4  * (S_bu / (S_bu + S_va + 1e-6)) * I_9
        Rho_10 = k_m_pro * S_pro / (K_S_pro + S_pro) * X_pro * I_10
        Rho_11 = k_m_ac  * S_ac  / (K_S_ac  + S_ac)  * X_ac  * I_11
        Rho_12 = k_m_h2  * S_h2  / (K_S_h2  + S_h2)  * X_h2  * I_12
        Rho_13 = k_dec_X_su  * X_su;  Rho_14 = k_dec_X_aa  * X_aa
        Rho_15 = k_dec_X_fa  * X_fa;  Rho_16 = k_dec_X_c4  * X_c4
        Rho_17 = k_dec_X_pro * X_pro; Rho_18 = k_dec_X_ac  * X_ac
        Rho_19 = k_dec_X_h2  * X_h2

        Rho_A_4  = k_A_B_va  * (S_va_ion  * (Ka_va  + S_H_ion) - Ka_va  * S_va)
        Rho_A_5  = k_A_B_bu  * (S_bu_ion  * (Ka_bu  + S_H_ion) - Ka_bu  * S_bu)
        Rho_A_6  = k_A_B_pro * (S_pro_ion * (Ka_pro + S_H_ion) - Ka_pro * S_pro)
        Rho_A_7  = k_A_B_ac  * (S_ac_ion  * (Ka_ac  + S_H_ion) - Ka_ac  * S_ac)
        Rho_A_10 = k_A_B_co2 * (S_hco3_ion* (Ka_co2 + S_H_ion) - Ka_co2 * S_IC)
        Rho_A_11 = k_A_B_IN  * (S_nh3     * (Ka_IN  + S_H_ion) - Ka_IN  * S_IN)

        p_gas_h2  = S_gas_h2  * R * T_op / MW_H2_gas
        p_gas_ch4 = S_gas_ch4 * R * T_op / COD_CH4
        p_gas_co2 = S_gas_co2 * R * T_op
        p_gas     = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o
        q_gas     = max(k_p * (p_gas - p_atm), 0.0) if mode == "continuous" else 0.0
        Rho_T_8  = k_L_a * (S_h2  - MW_H2_gas * K_H_h2  * p_gas_h2)
        Rho_T_9  = k_L_a * (S_ch4 - COD_CH4   * K_H_ch4 * p_gas_ch4)
        Rho_T_10 = k_L_a * (S_co2 - K_H_co2 * p_gas_co2)

        dec_sum = Rho_13+Rho_14+Rho_15+Rho_16+Rho_17+Rho_18+Rho_19
        s1  = -C_xc + f_sI_xc*C_sI + f_ch_xc*C_ch + f_pr_xc*C_pr + f_li_xc*C_li + f_xI_xc*C_xI
        s2  = -C_ch + C_su
        s3  = -C_pr + C_aa
        s4  = -C_li + (1-f_fa_li)*C_su + f_fa_li*C_fa
        s5  = -C_su + (1-Y_su)*(f_bu_su*C_bu + f_pro_su*C_pro + f_ac_su*C_ac) + Y_su*C_bac
        s6  = -C_aa + (1-Y_aa)*(f_va_aa*C_va + f_bu_aa*C_bu + f_pro_aa*C_pro + f_ac_aa*C_ac) + Y_aa*C_bac
        s7  = -C_fa + (1-Y_fa)*0.7*C_ac + Y_fa*C_bac
        s8  = -C_va + (1-Y_c4)*0.54*C_pro + (1-Y_c4)*0.31*C_ac + Y_c4*C_bac
        s9  = -C_bu + (1-Y_c4)*0.8*C_ac  + Y_c4*C_bac
        s10 = -C_pro + (1-Y_pro)*0.57*C_ac + Y_pro*C_bac
        s11 = -C_ac  + (1-Y_ac)*C_ch4 + Y_ac*C_bac
        s12 = (1-Y_h2)*C_ch4 + Y_h2*C_bac
        s13 = -C_bac + C_xc
        Sigma = (s1*Rho_1+s2*Rho_2+s3*Rho_3+s4*Rho_4
                 +s5*Rho_5+s6*Rho_6+s7*Rho_7+s8*Rho_8
                 +s9*Rho_9+s10*Rho_10+s11*Rho_11+s12*Rho_12
                 +s13*dec_sum)

        dy = np.zeros(len(STATE_NAMES))

        # ── Dilution terms (continuous mode) ──────────────────────────────
        if mode == "continuous":
            dy[I("S_su")]     += D * (in_su   - S_su)
            dy[I("S_aa")]     += D * (in_aa   - S_aa)
            dy[I("S_fa")]     += D * (in_fa   - S_fa)
            dy[I("S_va")]     += D * (in_va   - S_va)
            dy[I("S_bu")]     += D * (in_bu   - S_bu)
            dy[I("S_pro")]    += D * (in_pro  - S_pro)
            dy[I("S_ac")]     += D * (in_ac   - S_ac)
            dy[I("S_h2")]      = 0.0
            dy[I("S_ch4")]    += D * (in_ch4  - S_ch4)
            dy[I("S_IC")]     += D * (in_IC   - S_IC)
            dy[I("S_IN")]     += D * (in_IN   - S_IN)
            dy[I("S_I")]      += D * (in_SI   - S_I)
            dy[I("X_xc")]     += D * (in_Xxc  - X_xc)
            dy[I("X_ch")]     += D * (in_Xch  - X_ch)
            dy[I("X_pr")]     += D * (in_Xpr  - X_pr)
            dy[I("X_li")]     += D * (in_Xli  - X_li)
            dy[I("X_su")]     += D * (in_Xsu  - X_su)
            dy[I("X_aa")]     += D * (in_Xaa  - X_aa)
            dy[I("X_fa")]     += D * (in_Xfa  - X_fa)
            dy[I("X_c4")]     += D * (in_Xc4  - X_c4)
            dy[I("X_pro")]    += D * (in_Xpro - X_pro)
            dy[I("X_ac")]     += D * (in_Xac  - X_ac)
            dy[I("X_h2")]     += D * (in_Xh2  - X_h2)
            dy[I("X_I")]      += D * (in_XI   - X_I)
            dy[I("S_cation")] += D * (in_cat  - S_cation)
            dy[I("S_anion")]  += D * (in_an   - S_anion)
            # [PW-1] Six sub-pool dilution terms
            if pw_mode:
                dy[I("S_I_furan")]   += D * (in_SI_furan   - S_I_furan)
                dy[I("S_I_alcohol")] += D * (in_SI_alcohol - S_I_alcohol)
                dy[I("S_I_phenol")]  += D * (in_SI_phenol  - S_I_phenol)
                dy[I("S_I_acid")]    += D * (in_SI_acid    - S_I_acid)
                dy[I("S_I_ketone")]  += D * (in_SI_ketone  - S_I_ketone)
                dy[I("S_I_humic")]   += D * (in_SI_humic   - S_I_humic)

        # ── BSM2 biochemical reactions ─────────────────────────────────────
        # [PW-1] Total leak from all six sub-pools enters S_su
        dy[I("S_su")]     += Rho_2 + (1-f_fa_li)*Rho_4 - Rho_5 + Rho_leak_su
        dy[I("S_aa")]     += Rho_3 - Rho_6
        dy[I("S_fa")]     += f_fa_li*Rho_4 - Rho_7
        dy[I("S_va")]     += (1-Y_aa)*f_va_aa*Rho_6 - Rho_8
        dy[I("S_bu")]     += (1-Y_su)*f_bu_su*Rho_5 + (1-Y_aa)*f_bu_aa*Rho_6 - Rho_9
        dy[I("S_pro")]    += ((1-Y_su)*f_pro_su*Rho_5 + (1-Y_aa)*f_pro_aa*Rho_6
                              + (1-Y_c4)*0.54*Rho_8 - Rho_10)
        dy[I("S_ac")]     += ((1-Y_su)*f_ac_su*Rho_5 + (1-Y_aa)*f_ac_aa*Rho_6
                              + (1-Y_fa)*0.7*Rho_7 + (1-Y_c4)*0.31*Rho_8
                              + (1-Y_c4)*0.8*Rho_9 + (1-Y_pro)*0.57*Rho_10 - Rho_11)
        dy[I("S_ch4")]    += (1-Y_ac)*Rho_11 + (1-Y_h2)*Rho_12 - Rho_T_9
        dy[I("S_IC")]     += -Sigma - Rho_T_10
        dy[I("S_IN")]     += ((N_xc - f_xI_xc*N_I - f_sI_xc*N_I - f_pr_xc*N_aa)*Rho_1
                              - Y_su*N_bac*Rho_5 + (N_aa - Y_aa*N_bac)*Rho_6
                              - Y_fa*N_bac*Rho_7 - Y_c4*N_bac*(Rho_8 + Rho_9)
                              - Y_pro*N_bac*Rho_10 - Y_ac*N_bac*Rho_11
                              - Y_h2*N_bac*Rho_12 + (N_bac - N_xc)*dec_sum)
        dy[I("S_I")]      += f_sI_xc * Rho_1
        dy[I("X_xc")]     += -Rho_1 + dec_sum
        dy[I("X_ch")]     += f_ch_xc*Rho_1 - Rho_2
        dy[I("X_pr")]     += f_pr_xc*Rho_1 - Rho_3
        dy[I("X_li")]     += f_li_xc*Rho_1 - Rho_4
        dy[I("X_su")]     += Y_su*Rho_5  - Rho_13
        dy[I("X_aa")]     += Y_aa*Rho_6  - Rho_14
        dy[I("X_fa")]     += Y_fa*Rho_7  - Rho_15
        dy[I("X_c4")]     += Y_c4*(Rho_8 + Rho_9) - Rho_16
        dy[I("X_pro")]    += Y_pro*Rho_10 - Rho_17
        dy[I("X_ac")]     += Y_ac*Rho_11  - Rho_18
        dy[I("X_h2")]     += Y_h2*Rho_12  - Rho_19
        dy[I("X_I")]      += f_xI_xc*Rho_1

        dy[I("S_H_ion")]   = 0.0
        dy[I("S_va_ion")]  = -Rho_A_4
        dy[I("S_bu_ion")]  = -Rho_A_5
        dy[I("S_pro_ion")] = -Rho_A_6
        dy[I("S_ac_ion")]  = -Rho_A_7
        dy[I("S_hco3_ion")]= -Rho_A_10
        dy[I("S_co2")]     = 0.0
        dy[I("S_nh3")]     = -Rho_A_11
        dy[I("S_nh4_ion")] = 0.0

        dy[I("S_gas_h2")]  = Rho_T_8  * V_liq / V_gas - (q_gas / V_gas) * S_gas_h2
        dy[I("S_gas_ch4")] = Rho_T_9  * V_liq / V_gas - (q_gas / V_gas) * S_gas_ch4
        dy[I("S_gas_co2")] = Rho_T_10 * V_liq / V_gas - (q_gas / V_gas) * S_gas_co2

        # ── [PW-1] Six sub-pool ODEs: loss = leak + dilution ──────────────
        # Dilution already added in the continuous block above.
        # Biochemical loss = first-order leak to S_su only.
        dy[I("S_I_furan")]   -= leak_furan
        dy[I("S_I_alcohol")] -= leak_alcohol
        dy[I("S_I_phenol")]  -= leak_phenol
        dy[I("S_I_acid")]    -= leak_acid
        dy[I("S_I_ketone")]  -= leak_ketone
        dy[I("S_I_humic")]   -= leak_humic

        return dy

    return ode


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPER: build ADM1 state dict from derived fractions
# ─────────────────────────────────────────────────────────────────────────────

def _build_adm1_dict(fractions, COD_conc, S_IN, S_IC,
                     S_cation, S_anion, V_feed, V_dig_meta,
                     m_org, total_COD, mode, label):
    def f(key):
        return fractions.get(key, 0.0) * COD_conc

    d = {
        "S_su":  f("S_su"),  "S_aa":  f("S_aa"),  "S_fa":  f("S_fa"),
        "S_va":  f("S_va"),  "S_bu":  f("S_bu"),  "S_pro": f("S_pro"),
        "S_ac":  f("S_ac"),  "S_h2":  1e-8,        "S_ch4": 1e-8,
        "S_IC":  S_IC,       "S_IN":  S_IN,        "S_I":   f("S_I"),
        "X_xc":  f("X_xc"), "X_ch":  f("X_ch"),   "X_pr":  f("X_pr"),
        "X_li":  f("X_li"), "X_I":   f("X_I"),
        "S_cation": S_cation, "S_anion": S_anion,
        "_inf_S_su":   f("S_su"),  "_inf_S_aa":   f("S_aa"),
        "_inf_S_fa":   f("S_fa"),  "_inf_S_va":   f("S_va"),
        "_inf_S_bu":   f("S_bu"),  "_inf_S_pro":  f("S_pro"),
        "_inf_S_ac":   f("S_ac"),  "_inf_S_h2":   1e-8,
        "_inf_S_ch4":  1e-8,       "_inf_S_IC":   S_IC,
        "_inf_S_IN":   S_IN,       "_inf_S_I":    f("S_I"),
        "_inf_X_xc":   f("X_xc"), "_inf_X_ch":   f("X_ch"),
        "_inf_X_pr":   f("X_pr"), "_inf_X_li":   f("X_li"),
        "_inf_X_su":   0.0,        "_inf_X_aa":   0.0,
        "_inf_X_fa":   0.0,        "_inf_X_c4":   0.0,
        "_inf_X_pro":  0.0,        "_inf_X_ac":   0.0,
        "_inf_X_h2":   0.0,        "_inf_X_I":    f("X_I"),
        "_inf_S_cation": S_cation, "_inf_S_anion": S_anion,
        "_V_feed_m3":      V_feed,
        "_V_digester_m3":  V_dig_meta,
        "_COD_conc":       COD_conc,
        "_m_org_kg":       m_org,
        "_total_COD_kg":   total_COD,
        "_fractions":      fractions,
    }
    # Initialise PW sub-pool keys to zero (overwritten in translate_pw if pw_mode)
    for sp in SI_SUBPOOL_NAMES:
        d[f"S_I_{sp}"]      = 0.0
        d[f"_inf_S_I_{sp}"] = 0.0
    return d


# ─────────────────────────────────────────────────────────────────────────────
# PATHWAY A — translate HTC process water → ADM1 state dict
# ─────────────────────────────────────────────────────────────────────────────

def translate_pw_to_adm1(C_pw, H_pw, O_pw, N_pw, S_pw,
                          PW_yield_L,
                          mode="batch",
                          HRT_days=None,
                          pH_target=7.2,
                          T_op=308.15,
                          T_HTC=180.0,
                          RT_HTC=60.0,
                          ISR_batch=5.0,
                          COD_conc_target=10.0):
    """
    Convert HTC process water elemental composition + volume into ADM1 dict.

    [PW-3] SI sub-pool initialisation:
      Total SI = fractions["S_I"] * COD_conc
      Fast/slow split: linear interpolation from UASB calibration data
        f_slow = 0.40 at 180°C → 0.50 at 240°C (Tanguay-Rioux et al. 2026)
      Within each rate class, sub-pool fractions from SI_COMPOSITION table
      (temperature-interpolated).
    """
    m_org      = C_pw + H_pw + O_pw + N_pw + S_pw
    total_COD  = compute_thod(C_pw, H_pw, O_pw, N_pw, S_pw)
    V_feed     = PW_yield_L / 1000.0
    if V_feed <= 0:
        raise ValueError("PW_yield_L must be > 0")

    COD_conc_raw = total_COD / V_feed

    if COD_conc_raw > COD_conc_target:
        dilution_factor   = COD_conc_raw / COD_conc_target
        V_feed_diluted    = V_feed * dilution_factor
        dilution_water_m3 = V_feed_diluted - V_feed
        COD_conc          = COD_conc_target
        print(f"[translate_pw] PW dilution: COD {COD_conc_raw:.1f} → "
              f"{COD_conc:.1f} kgCOD/m³  (factor={dilution_factor:.2f}×  "
              f"dilution water={dilution_water_m3*1000:.1f} L)")
    else:
        V_feed_diluted    = V_feed
        dilution_water_m3 = 0.0
        COD_conc          = COD_conc_raw
        print(f"[translate_pw] No dilution needed — "
              f"COD_conc={COD_conc:.1f} kgCOD/m³")

    if mode == "continuous":
        if HRT_days is None:
            raise ValueError("HRT_days required for continuous mode")
        V_dig_meta = V_feed_diluted * HRT_days
    else:
        VS_sub     = total_COD / 1.4          # substrate VS (kg)
        rho_inoc   = 12.0                     # inoculum VS concentration (kgVS/m³)
        V_inoc     = (VS_sub * ISR_batch) / rho_inoc   # m³ of inoculum
        V_dig_meta = V_feed_diluted + V_inoc
                    
        print(f"[translate_pw batch] VS_sub={VS_sub:.2f} kg  "
              f"V_inoc={V_inoc:.4f} m³  ISR={ISR_batch}  "    # ← use ISR_batch
              f"V_dig={V_dig_meta:.4f} m³")

    S_IN, f_IN = compute_S_IN(N_pw, V_feed_diluted, "htc_pw", T_HTC, RT_HTC)
    S_IC = float(np.clip(COD_conc * 0.8 / 61.0, 0.08, 0.20)) #compute_S_IC(pH=pH_target, T_op_K=T_op, p_CO2_atm=0.0005)

    fractions = derive_adm1_fractions(C_pw, H_pw, O_pw, N_pw, S_pw,
                                       substrate_type="htc_pw", T_HTC=T_HTC)
    _print_fractions(fractions, "htc_pw", C_pw, H_pw, O_pw, N_pw, S_pw)

    S_cation, S_anion = compute_charge_pair(
        S_IC, S_IN,
        fractions.get("S_va",  0.0) * COD_conc,
        fractions.get("S_bu",  0.0) * COD_conc,
        fractions.get("S_pro", 0.0) * COD_conc,
        fractions.get("S_ac",  0.0) * COD_conc,
        pH_target=pH_target, T_op=T_op,
    )

    d = _build_adm1_dict(fractions, COD_conc, S_IN, S_IC,
                          S_cation, S_anion, V_feed_diluted, V_dig_meta,
                          m_org, total_COD, mode, label="pw")

    d["_dilution_water_m3"] = dilution_water_m3
    d["_COD_conc_raw"]      = COD_conc_raw
    d["_dilution_factor"]   = V_feed_diluted / V_feed

    # ── [PW-3] Initialise six SI sub-pools ───────────────────────────────
    S_I_total = fractions["S_I"] * COD_conc

    # Fast/slow partition varies linearly with HTC temperature
    # Calibrated from UASB steady-state effluent sCOD data:
    #   180°C: f_slow = 0.40  (effluent sCOD = 17.5% of feed)
    #   240°C: f_slow = 0.50  (effluent sCOD = 23.0% of feed)
    _f_slow_override = PW_ADM1_PARAMS.get("f_slow_override", None)
    if _f_slow_override is not None:
        f_slow = float(np.clip(_f_slow_override, 0, 1))
        print(f"[translate_pw] f_slow OVERRIDDEN to {f_slow:.3f}")
    else:
        f_slow = float(np.clip(
            0.40 + (T_HTC - 180.0) / (240.0 - 180.0) * 0.10,
            0, 1
        ))
    f_fast = 1.0 - f_slow

    # Temperature-interpolated composition fractions
    si_fracs = get_si_composition(T_HTC)

    # Assign each sub-pool from its rate class fraction × composition fraction
    # Fast pools: furan, alcohol, phenol, acid
    # Slow pools: ketone, humic
    # Within each class the composition fractions are re-normalised so they
    # sum to 1.0 within that class, then scaled by the class total COD.
    fast_total_frac = sum(si_fracs[sp] for sp in SI_FAST_POOLS)
    slow_total_frac = sum(si_fracs[sp] for sp in SI_SLOW_POOLS)

    for sp in SI_FAST_POOLS:
        val = S_I_total * f_fast * (si_fracs[sp] / fast_total_frac)
        d[f"S_I_{sp}"]      = val
        d[f"_inf_S_I_{sp}"] = val

    for sp in SI_SLOW_POOLS:
        val = S_I_total * f_slow * (si_fracs[sp] / slow_total_frac)
        d[f"S_I_{sp}"]      = val
        d[f"_inf_S_I_{sp}"] = val

    # Standard S_I zeroed — entirely replaced by six sub-pools
    d["S_I"]      = 0.0
    d["_inf_S_I"] = 0.0

    d["_f_IN"]   = f_IN
    d["_pw_mode"] = True

    print(f"[translate_pw] mode={mode}  COD_conc={COD_conc:.3f} kgCOD/m³  "
          f"T_HTC={T_HTC:.0f}°C  f_slow={f_slow:.2f}  f_fast={f_fast:.2f}")
    print(f"[translate_pw] SI sub-pools (kgCOD/m³):")
    for sp in SI_SUBPOOL_NAMES:
        print(f"  S_I_{sp:<8} = {d[f'S_I_{sp}']:.4f}")
    if mode == "continuous" and HRT_days:
        OLR = COD_conc / HRT_days
        print(f"[translate_pw] OLR={OLR:.2f} kgCOD/m³/d  HRT={HRT_days:.1f} d")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# PATHWAY B — translate raw feedstock → ADM1 state dict
# ─────────────────────────────────────────────────────────────────────────────

def translate_feedstock_to_adm1(C_feed, H_feed, O_feed, N_feed, S_feed,
                                  feedstock_wet_kg,
                                  ash_kg=0.0,
                                  mode="batch",
                                  HRT_days=None,
                                  pH_target=7.2,
                                  TS_target=0.08,
                                  slurry_density=1.02,
                                  T_op=308.15):
    m_org     = C_feed + H_feed + O_feed + N_feed + S_feed
    m_dry     = m_org + ash_kg
    total_COD = compute_thod(C_feed, H_feed, O_feed, N_feed, S_feed)
    TS_feed   = m_dry / feedstock_wet_kg

    if TS_feed > TS_target:
        m_slurry          = m_dry / TS_target
        dilution_water_kg = m_slurry - feedstock_wet_kg
    else:
        m_slurry          = feedstock_wet_kg
        dilution_water_kg = 0.0

    V_feed   = m_slurry / (slurry_density * 1000.0)
    COD_conc = total_COD / V_feed

    if mode == "continuous":
        if HRT_days is None:
            raise ValueError("HRT_days required for continuous mode")
        V_dig_meta   = V_feed * HRT_days
        COD_conc_inf = COD_conc
        S_IN, f_IN   = compute_S_IN(N_feed, V_feed, "feedstock")
    else:
        VS_sub       = total_COD / 1.4
        V_inoc       = 2.0 * VS_sub / 12.0
        V_dig_meta   = V_feed + V_inoc
        COD_conc_inf = total_COD / V_dig_meta
        S_IN         = min(N_feed / V_dig_meta / MW_N, 0.13)

    S_IC = compute_S_IC(pH=8, T_op_K=T_op, p_CO2_atm=0.0005)

    fractions = derive_adm1_fractions(C_feed, H_feed, O_feed, N_feed, S_feed,
                                       ash_kg=ash_kg, substrate_type="feedstock")
    _print_fractions(fractions, "feedstock", C_feed, H_feed, O_feed, N_feed, S_feed)

    S_cation, S_anion = compute_charge_pair(
        S_IC, S_IN,
        fractions.get("S_va",  0.0) * COD_conc_inf,
        fractions.get("S_bu",  0.0) * COD_conc_inf,
        fractions.get("S_pro", 0.0) * COD_conc_inf,
        fractions.get("S_ac",  0.0) * COD_conc_inf,
        pH_target=pH_target, T_op=T_op,
    )

    d = _build_adm1_dict(fractions, COD_conc_inf, S_IN, S_IC,
                          S_cation, S_anion, V_feed, V_dig_meta,
                          m_org, total_COD, mode, label="feedstock")

    d["_COD_conc_feed"]     = COD_conc
    d["_m_slurry_kg"]       = m_slurry
    d["_TS_feed"]           = TS_feed
    d["_TS_target"]         = TS_target
    d["_dilution_water_kg"] = dilution_water_kg
    d["_ash_kg"]            = ash_kg
    d["_m_dry_kg"]          = m_dry
    d["_m_org_kg"]          = m_org
    d["_pw_mode"]           = False   # Pathway B — no PW extensions

    print(f"[translate_feedstock] mode={mode}  COD_conc={COD_conc_inf:.3f} kgCOD/m³")
    if mode == "continuous" and HRT_days:
        print(f"[translate_feedstock] OLR={COD_conc_inf/HRT_days:.2f} kgCOD/m³/d  "
              f"HRT={HRT_days:.1f} d")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# STEADY-STATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_ss_state(sim_df, last_frac=0.05):
    n = len(sim_df)
    w = max(1, int(n * last_frac))
    tail = sim_df.iloc[-w:]
    return {col: float(tail[col].mean()) for col in tail.columns if col != "time"}


def _check_convergence(records, window_frac=0.20, tol=5e-3):
    KEY = ["S_ac","S_pro","S_IC","X_ac","X_h2","X_pro","pH"]
    n = len(records)
    w = max(2, int(n * window_frac))
    changes = {}
    for k in KEY:
        vals = [r.get(k, 0.0) for r in records]
        last = np.mean(vals[-w:]); prev = np.mean(vals[-2*w:-w])
        changes[k] = abs(last - prev) / max(abs(last), 1e-10)
    return all(v < tol for v in changes.values()), changes


# ─────────────────────────────────────────────────────────────────────────────
# PATHWAY B DIGESTATE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_digestate_composition(CH4_mass, CO2_mass, NH3_mass,
                                   C_feed, H_feed, O_feed, N_feed, S_feed,
                                   m_feedstock_wet, m_feedstock_dry):
    Ash   = max(m_feedstock_dry - (C_feed + H_feed + O_feed + N_feed + S_feed), 0.0)
    C_gas = CH4_mass * (MW_C / MW_CH4) + CO2_mass * (MW_C / MW_CO2)
    C_dig = max(C_feed - C_gas, 0.0)
    N_dig = N_feed
    S_dig = S_feed
    HC_feed_molar = (H_feed / MW_H) / max(C_feed / MW_C, 1e-9)
    H_dig = C_dig * (HC_feed_molar * MW_H / MW_C)
    HO_molar = (H_feed / MW_H) / max(O_feed / MW_O, 1e-9)
    O_dig = max(H_dig / (HO_molar * MW_H / MW_O), 0.01 * C_dig)
    total = C_dig + H_dig + O_dig + N_dig + S_dig + Ash
    if total <= 0:
        raise ValueError("Digestate dry mass is zero.")
    m_wet  = max(m_feedstock_wet - CH4_mass - CO2_mass, 0.0)
    m_dry  = max(m_feedstock_dry - C_gas, 0.0)
    WC_pct = float(np.clip((m_wet - m_dry) / m_wet * 100, 0, 99.9)) if m_wet > 0 else 0.0
    Ash_pct = Ash / total * 100
    FC_dig  = max(2.0, min(5.0, 100.0 - Ash_pct - 75.0))
    VM_dig  = max(0.0, 100.0 - Ash_pct - FC_dig)
    return {
        "C_feed":   round(C_dig / total * 100, 2),
        "H_feed":   round(H_dig / total * 100, 2),
        "O_feed":   round(O_dig / total * 100, 2),
        "N_feed":   round(N_dig / total * 100, 2),
        "S_feed":   round(S_dig / total * 100, 2),
        "Ash_feed": round(Ash   / total * 100, 2),
        "WC":       round(WC_pct, 2),
        "VM_feed":  round(VM_dig, 2),
        "FC_feed":  round(FC_dig, 2),
        "_C_dig_kg":   round(C_dig, 4), "_H_dig_kg": round(H_dig, 4),
        "_O_dig_kg":   round(O_dig, 4), "_N_dig_kg": round(N_dig, 4),
        "_S_dig_kg":   round(S_dig, 4), "_Ash_dig_kg": round(Ash, 4),
        "_m_wet_kg":   round(m_wet, 4), "_m_dry_kg":   round(m_dry, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_adm1(substrate,
             t_batch_days=30,
             headspace_fraction=0.40,
             mode="batch",
             q_ad=None,
             HRT_days=None,
             t_sim_days=None,
             ISR_batch=2.0,
             ss_convergence_tol=5e-3,
             max_hrt_multiplier=20,
             ss_initial_state=None,
             solvermethod="Radau",
             verbose=True,
             V_liq_override=None,
             T_op=308.15,
             K_I_nh3_override=None):

    if mode not in ("batch", "continuous"):
        raise ValueError(f"mode must be 'batch' or 'continuous', got '{mode}'")

    c = _build_constants(T_op)
    if K_I_nh3_override is not None:
        c["K_I_nh3"] = float(K_I_nh3_override)
        print(f"[run_adm1] K_I_nh3 overridden to {c['K_I_nh3']:.4f} kmolN/m³")

    R_gas = c["R"]; p_gas_h2o = c["p_gas_h2o"]
    k_p   = c["k_p"]; p_atm = c["p_atm"]

    if V_liq_override is not None:
        V_liq = float(V_liq_override)
    elif mode == "continuous":
        if q_ad is None or HRT_days is None:
            raise ValueError("continuous mode requires q_ad and HRT_days")
        V_liq = q_ad * HRT_days
    else:
        V_liq = substrate["_V_digester_m3"]

    V_gas = V_liq * (headspace_fraction / (1.0 - headspace_fraction)) \
            if mode == "batch" else V_liq * 0.25

    if mode == "continuous":
        HRT_actual   = V_liq / q_ad
        t_sim_max    = max_hrt_multiplier * HRT_actual
        t_sim_actual = t_sim_days if t_sim_days is not None else t_sim_max
        print(f"[run_adm1 continuous] q_ad={q_ad:.5f} m³/d  HRT={HRT_actual:.2f} d  "
              f"V_liq={V_liq:.5f} m³")
    else:
        t_sim_actual = t_batch_days
        print(f"[run_adm1 batch] t_batch={t_batch_days} d  V_liq={V_liq:.5f} m³")

    pw_mode = bool(substrate.get("_pw_mode", False))
    pw_mode_sub = substrate.get("_pw_mode", False)
    X_seed = None

    # ── Build initial state vector ─────────────────────────────────────────
    if ss_initial_state is not None:
        sv0 = dict(ss_initial_state)
        for k in STATE_NAMES:
            sv0.setdefault(k, 0.0)
        if mode == "batch":
            for k in SUBSTRATE_KEYS + PW_EXTRA_KEYS:
                if k in substrate:
                    sv0[k] = substrate[k]
        sv0["S_cation"] = substrate.get("S_cation", sv0.get("S_cation", 0.04))
        sv0["S_anion"]  = substrate.get("S_anion",  sv0.get("S_anion",  0.02))
        for k in PW_EXTRA_KEYS:
            sv0[k] = substrate.get(k, 0.0)
        bm = sum(sv0.get(k, 0.0) for k in BIOMASS_KEYS)
        print(f"[run_adm1] Warm start — biomass: {bm:.4f} kgCOD/m³")
    else:
        COD_conc = substrate.get("_COD_conc", 3.0)
        if "_force_seed_val" in substrate:
            X_seed = substrate["_force_seed_val"]
        elif mode == "batch":
            X_seed = float(np.clip(COD_conc * ISR_batch, 2.0, 20.0))

        else:
            X_seed = float(np.clip(max(BSM2_COLD_SEED_BIOMASS, COD_conc * 0.10), 3.0, 15.0))

        OLR = COD_conc / max(HRT_days, 1.0) if (mode == "continuous" and HRT_days) else 2.0
        print(f"[run_adm1] Cold start ({mode}) — COD_conc={COD_conc:.2f}  "
              f"OLR={OLR:.2f}  seed={X_seed:.2f} kgCOD/m³")

        sv0 = {k: 0.0 for k in STATE_NAMES}
        for k in SUBSTRATE_KEYS:
            sv0[k] = substrate.get(k, 0.0)
        for k in PW_EXTRA_KEYS:
            sv0[k] = substrate.get(k, 0.0)
        for bk, frac in BSM2_BIOMASS_FRACTIONS.items():
            sv0[bk] = frac * X_seed

        sv0["S_h2"] = 1e-8; sv0["S_ch4"] = 1e-8
        if mode == "batch":
            if pw_mode_sub:
                # Pathway A: S_IC already set correctly by translate_pw (0.08+)
                sv0["S_IC"]     = substrate.get("S_IC", 0.08)
                sv0["S_cation"] = substrate.get("S_cation", 0.04)
            else:
                # Pathway B: substrate dict has S_IC=0.01 (from compute_S_IC at pH=8)
                # Override with realistic inoculum buffer — VDI 4630 inocula
                # typically have 0.05-0.15 kmolC/m³ alkalinity
                sv0["S_IC"]     = max(substrate.get("S_IC", 0.07), 0.07)
                sv0["S_cation"] = max(substrate.get("S_cation", 0.04), 0.12)
        else:
            sv0["S_IC"] = substrate.get("S_IC", 0.01)
            sv0["S_cation"] = substrate.get("S_cation", 0.04)
        sv0["S_anion"] = substrate.get("S_anion", 0.02)

        S_IC = sv0["S_IC"]; S_IN = sv0["S_IN"]
        sv0["S_H_ion"]    = 10**-7.2
        sv0["S_hco3_ion"] = 0.83   * S_IC
        sv0["S_co2"]      = 0.17   * S_IC
        sv0["S_nh3"]      = 0.0105 * S_IN
        sv0["S_nh4_ion"]  = 0.9895 * S_IN
        sv0["S_va_ion"]   = c["K_a_va"]  / (c["K_a_va"]  + 10**-7.2) * sv0["S_va"]
        sv0["S_bu_ion"]   = c["K_a_bu"]  / (c["K_a_bu"]  + 10**-7.2) * sv0["S_bu"]
        sv0["S_pro_ion"]  = c["K_a_pro"] / (c["K_a_pro"] + 10**-7.2) * sv0["S_pro"]
        sv0["S_ac_ion"]   = c["K_a_ac"]  / (c["K_a_ac"]  + 10**-7.2) * sv0["S_ac"]
        sv0["S_gas_h2"]   = 1e-8; sv0["S_gas_ch4"] = 1e-8; sv0["S_gas_co2"] = 0.0

    if pw_mode:
        print(f"[run_adm1] PW-ADM1 ACTIVE — SI sub-pools (kgCOD/m³):")
        for sp in SI_SUBPOOL_NAMES:
            print(f"  S_I_{sp:<8} = {sv0.get(f'S_I_{sp}', 0):.4f}  "
                  f"K_I={PW_ADM1_PARAMS[f'K_I_{sp}']:.1f}  "
                  f"k_leak={PW_ADM1_PARAMS[f'k_leak_{sp}']:.4f} d⁻¹")

    ode_func = _build_ode(c, V_liq, V_gas, STATE_NAMES, IDX,
                           mode=mode,
                           influent=substrate if mode == "continuous" else {},
                           q_ad=q_ad if mode == "continuous" else 0.0,
                           pw_mode=pw_mode)

    current_state = [sv0[k] for k in STATE_NAMES]
    results  = [dict(sv0)]
    results[0]["pH"] = -np.log10(max(sv0.get("S_H_ion", 1e-7), 1e-14))
    gasflow  = [{"q_gas": 0.0, "q_ch4": 0.0, "q_co2": 0.0, "q_h2": 0.0, "time": 0.0}]
    ss_converged = False

    if mode == "continuous":
        dt    = HRT_actual
        n_pts = max(10, int(HRT_actual * 10))
        t_cur = 0.0; block = 0; MIN_BLOCKS = 5

        with tqdm(total=int(t_sim_actual), desc="ADM1 continuous",
                  unit="d", disable=not verbose) as pbar:
            while t_cur < t_sim_actual and not ss_converged:
                t_end   = min(t_cur + dt, t_sim_actual)
                t_block = np.linspace(t_cur, t_end, n_pts)
                for n in range(1, len(t_block)):
                    sol   = scipy.integrate.solve_ivp(
                        ode_func, [t_block[n-1], t_block[n]], current_state,
                        method=solvermethod, rtol=1e-6, atol=1e-8)
                    y_end = sol.y[:, -1]
                    sv_cur = dict(zip(STATE_NAMES, y_end))
                    sv_cur.update(_dae_solve(sv_cur, c))
                    p_h2  = sv_cur["S_gas_h2"]  * R_gas * T_op / MW_H2_gas
                    p_ch4 = sv_cur["S_gas_ch4"] * R_gas * T_op / COD_CH4
                    p_co2 = sv_cur["S_gas_co2"] * R_gas * T_op
                    p_tot = p_h2 + p_ch4 + p_co2 + p_gas_h2o
                    q_g   = max(k_p * (p_tot - p_atm), 0.0)
                    q_ch4_s = q_g * p_ch4 / p_tot if p_tot > 0 else 0.0
                    q_co2_s = q_g * p_co2 / p_tot if p_tot > 0 else 0.0
                    q_h2_s  = q_g * p_h2  / p_tot if p_tot > 0 else 0.0
                    gasflow.append({"q_gas": q_g, "q_ch4": q_ch4_s,
                                    "q_co2": q_co2_s, "q_h2": q_h2_s,
                                    "time": t_block[n]})
                    results.append(sv_cur)
                    current_state = [sv_cur[k] for k in STATE_NAMES]
                t_cur = t_end; block += 1
                pbar.update(int(dt))
                if block >= MIN_BLOCKS:
                    ss_converged, _ = _check_convergence(results, tol=ss_convergence_tol)
                    if ss_converged:
                        print(f"\n[run_adm1] Converged at t={t_cur:.1f} d ({block}×HRT)")
        if not ss_converged:
            print(f"[run_adm1] Did not converge within {t_sim_actual:.0f} d.")

    else:
        n_pts = int(t_batch_days * 10) + 1
        t_arr = np.linspace(0, t_batch_days, n_pts)
        for n in tqdm(range(1, len(t_arr)),
                      desc="ADM1 batch", unit="step", disable=not verbose):
            sol   = scipy.integrate.solve_ivp(
                ode_func, [t_arr[n-1], t_arr[n]], current_state,
                method=solvermethod, rtol=1e-6, atol=1e-8)
            y_end = sol.y[:, -1]
            sv_cur = dict(zip(STATE_NAMES, y_end))
            sv_cur.update(_dae_solve(sv_cur, c))
            gasflow.append({"q_gas": 0.0, "q_ch4": 0.0, "q_co2": 0.0,
                            "q_h2": 0.0, "time": t_arr[n]})
            results.append(sv_cur)
            current_state = [sv_cur[k] for k in STATE_NAMES]

    sim_df = pd.DataFrame(results)
    gas_df = pd.DataFrame(gasflow)
    times  = [r["time"] for r in gasflow]
    sim_df["time"] = times

    ss_out = extract_ss_state(sim_df, last_frac=0.05)
    final  = results[-1]

    if mode == "batch":
        CH4_mass   = final["S_gas_ch4"] * V_gas * (MW_CH4 / COD_CH4)
        CO2_mass   = final["S_gas_co2"] * V_gas * MW_CO2
        NH3_mass   = final["S_nh3"]     * V_liq * MW_NH3
        dig_kg     = max(0, V_liq * 1000 - CH4_mass - CO2_mass)
        tot_ch4_m3 = CH4_mass / RHO_CH4
        tot_co2_m3 = CO2_mass / RHO_CO2
        extra = {}
    else:
        ss0      = int(len(gas_df) * 0.80)
        q_ch4_ss = float(gas_df["q_ch4"].iloc[ss0:].mean())
        q_co2_ss = float(gas_df["q_co2"].iloc[ss0:].mean())
        q_gas_ss = float(gas_df["q_gas"].iloc[ss0:].mean())
        rho_CH4  = MW_CH4 / (R_gas * T_op)
        rho_CO2  = MW_CO2 / (R_gas * T_op)
        tot_ch4_m3 = q_ch4_ss * HRT_actual
        tot_co2_m3 = q_co2_ss * HRT_actual
        CH4_mass   = tot_ch4_m3 * rho_CH4
        CO2_mass   = tot_co2_m3 * rho_CO2
        NH3_mass   = float(np.mean([r["S_nh3"] for r in results[ss0:]])) * V_liq * MW_NH3
        dig_kg     = max(0, substrate.get("_V_feed_m3", 0.0) * 1000
                         - CH4_mass - CO2_mass)
        extra = {
            "q_ch4_ss": q_ch4_ss, "q_co2_ss": q_co2_ss, "q_gas_ss": q_gas_ss,
            "HRT_days": HRT_actual, "t_sim_days": times[-1],
            "ss_converged": ss_converged,
        }

    print(f"\n[run_adm1 {mode}] CH4 = {CH4_mass:.5f} kg  ({tot_ch4_m3:.5f} m³)")
    print(f"[run_adm1 {mode}] CO2 = {CO2_mass:.5f} kg  ({tot_co2_m3:.5f} m³)")
    print(f"[run_adm1 {mode}] NH3 = {NH3_mass:.5f} kg")

    print(f"\n=== SS DIAGNOSTIC ===")
    for k in ["X_ac","X_h2","X_pro","S_ac","S_pro","S_IN","S_nh3","pH","S_h2"]:
        print(f"  {k:<10}: {ss_out.get(k, 0):.5f}")
    if pw_mode:
        print(f"  --- PW-ADM1 sub-pools (SS) ---")
        I_tox_ss = 1.0
        for sp in SI_SUBPOOL_NAMES:
            conc = ss_out.get(f"S_I_{sp}", 0)
            ki   = PW_ADM1_PARAMS[f"K_I_{sp}"]
            inh  = ki / (ki + conc)
            I_tox_ss *= inh
            print(f"  S_I_{sp:<8}: {conc:.4f} kgCOD/m³  "
                  f"K_I={ki:.1f}  I={inh:.3f}")
        print(f"  I_tox (SS, combined): {I_tox_ss:.4f}  "
              f"({I_tox_ss*100:.1f}% of uninhibited rate)")
    print(f"=====================")

    return {
        "CH4_mass_kg":       CH4_mass,
        "CO2_mass_kg":       CO2_mass,
        "NH3_mass_kg":       NH3_mass,
        "_X_seed_used":   X_seed,
        "digestate_mass_kg": dig_kg,
        "total_ch4_m3":      tot_ch4_m3,
        "total_co2_m3":      tot_co2_m3,
        "simulate_results":  sim_df,
        "gasflow":           gas_df,
        "_V_liq":            V_liq,
        "ss_state":          ss_out,
        **extra,
    }