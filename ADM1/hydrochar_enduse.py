"""
hydrochar_enduse.py
===================
End use of the dried hydrochar: combustion in an industrial boiler for heat,
displacing the Quebec industrial combustion heat mix (electricity excluded).
Pure Python and configuration; the Brightway exchanges are written by
system_builder.build_hydrochar_enduse_exchanges().

  SF       = HHV_char / HHV_coal                      (both gross, MJ/kg)
  MJ_heat  = m_dry_char x SF / COAL_KG_PER_MJ
           = m_dry_char x HHV_char x 0.760            (useful heat delivered)
  Stack    = min(composition-based estimate, emission limit x dry flue gas)
"""

import numpy as np


# =============================================================================
# HEAT DELIVERED BY THE HYDROCHAR BOILER
# =============================================================================
# The boiler is represented by the ecoinvent activity "heat production, at
# hard coal industrial furnace 1-10MW" (CA-QC, ecoinvent 3.12 cut-off): 0.04325
# kg of hard coal per MJ of heat, with a net heating value of 28.9 MJ/kg. The
# HHV of the hydrochar predicted by the surrogate is a gross value, so the coal
# is put on the same basis with a net-to-gross ratio of 0.95 (IEA/OECD/
# Eurostat, 2005, Energy Statistics Manual): HHV_coal = 30.42 MJ/kg. The
# implied boiler efficiency is 1 / (0.04325 x 30.42) = 0.760 on a gross basis.
# The hydrochar is assumed to lose the same share of its HHV to water vapour
# as hard coal.
LHV_COAL_MJ_PER_KG = 28.9          # ecoinvent property "heating value, net"


COAL_NET_TO_GROSS = 0.95           # IEA/OECD/Eurostat (2005)


HHV_COAL_MJ_PER_KG = LHV_COAL_MJ_PER_KG / COAL_NET_TO_GROSS   # 30.42


COAL_KG_PER_MJ = 0.04325           # same ecoinvent activity, coal input per MJ heat


# Quebec industrial combustion heat mix (electricity excluded), normalized
# shares by energy carrier, from the industrial energy consumption reported in
# Whitmore, J. and Pineau, P.-O. (2026). Etat de l'energie au Quebec 2026, Graphique 25.
# Chaire de gestion du secteur de l'energie, HEC Montreal.
QC_HEAT_MIX = {
    "natural_gas": 0.506,
    "bioenergy":   0.295,
    "petroleum":   0.148,
    "coal_coke":   0.042,
    "lng":         0.008,
}


# ecoinvent 3.12 datasets for each group of the heat mix, with exact activity
# names and reference products, taken from the two Quebec heat markets of
# ecoinvent ("market for heat, district or industrial, natural gas" and
# "... other than natural gas"). Each entry is
#     (activity name, reference product, weight within the group).
# The group shares come from QC_HEAT_MIX; the weights only split a group
# across datasets and are normalized within the group. They are the relative
# amounts of each dataset in the ecoinvent CA-QC "other than natural gas"
# market (petroleum: heavy fuel oil 0.584, light fuel oil 0.0262, propane
# 0.00671; coal and coke: hard coal 0.155, coal coke 0.067). Bioenergy is
# represented by the only CA-QC wood heat dataset of that market (wood chips
# CHP); because the activity has two products, the reference product is also
# matched. Natural gas liquids, LGN (0.8 %, propane and butane; Paper 1 label "lng") use the natural gas heat market in the Paper 1 mix.
_H_NG = "heat, district or industrial, natural gas"


_H_OTHER = "heat, district or industrial, other than natural gas"


HEAT_MIX_DATASETS = {
    "natural_gas": [("market for heat, district or industrial, natural gas", _H_NG, 1.0)],
    "bioenergy":   [("heat and power co-generation, wood chips, 6667 kW, state-of-the-art 2014", _H_OTHER, 1.0)],
    "petroleum":   [("heat production, heavy fuel oil, at industrial furnace 1MW", _H_OTHER, 0.584),
                    ("heat production, light fuel oil, at industrial furnace 1MW", _H_OTHER, 0.0262),
                    ("heat production, propane, at industrial furnace >100kW", _H_OTHER, 0.00671)],
    "coal_coke":   [("heat production, at hard coal industrial furnace 1-10MW", _H_OTHER, 0.155),
                    ("heat production, at coal coke industrial furnace 1-10MW", _H_OTHER, 0.0670)],
    "lng":         [("market for heat, district or industrial, natural gas", _H_NG, 1.0)],
    # Paper 2 regional mixes: LPG and gas-plant NGL -> propane furnace
    "lpg_ngl":     [("heat production, propane, at industrial furnace >100kW", _H_OTHER, 1.0)],
}


# Paper 2: regional industrial combustion heat mixes, 2023. Source: Natural
# Resources Canada, Comprehensive Energy Use Database (OEE), Industrial
# Sector - Aggregated Industries, Table 1, energy use in PJ by energy source,
# stored verbatim in data/nrcan_ceud_2023_industrial_table1.csv (Canada = sum
# of the seven regions). Shares = combustion fuels only: electricity and
# "Other" (steam, cement waste fuels) excluded, then renormalized.
#   natural_gas  natural gas
#   bioenergy    wood waste and pulping liquor
#   petroleum    heavy fuel oil + diesel, light fuel oil and kerosene
#                (+ still gas and petroleum coke if INCLUDE_REFINERY_FUELS)
#   coal_coke    coal + coke and coke oven gas
#   lpg_ngl      LPG and gas-plant NGL (not liquefied natural gas)
# Still gas and petroleum coke are excluded by default: they are mostly
# refinery and upgrader by-product fuels burned where they are produced,
# which hydrochar does not displace. With INCLUDE_REFINERY_FUELS = True they
# are assigned to the heavy fuel oil dataset (closest ecoinvent proxy).
# Within petroleum and coal_coke, the dataset weights are the region's own
# PJ (regional_heat_mix_datasets), not the ecoinvent CA-QC market weights.
# NRCan publishes the four Atlantic provinces as one region (ATL) and BC with
# the Territories (BC_TER); those provinces take their region's mix.
# QC_HEAT_MIX above stays the Paper 1 default.
import csv as _csv
import os as _os

INCLUDE_REFINERY_FUELS = False

# Natural gas heat for industry (Paper 2 presets): the boiler and furnace
# technologies of the ecoinvent 3.12 CA-QC "market for heat, district or
# industrial, natural gas", with its weights, taken in every region from the
# region's location priority. The RoW market used outside Quebec is about 94 %
# combined heat and power (Russian and RoW conventional plants alone 60 %),
# whose heat carries only part of the plant burden through allocation
# (0.0377 vs 0.0690 kg CO2 eq/MJ for CA-QC, IMPACT World+ climate change short
# term); it does not represent industrial heat displaced in a Canadian
# province. Used for the natural gas share of the heat mix and for the heat
# displaced by biomethane.
NG_INDUSTRIAL_HEAT = [
    ("heat production, natural gas, at boiler modulating >100kW", _H_NG, 0.2532),
    ("heat production, natural gas, at industrial furnace >100kW", _H_NG, 0.2408),
    ("heat production, natural gas, at industrial furnace low-NOx >100kW", _H_NG, 0.2370),
    ("heat production, natural gas, at boiler condensing modulating >100kW", _H_NG, 0.2690),
]

NRCAN_TABLE1_CSV = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                 "data", "nrcan_ceud_2023_industrial_table1.csv")

REGIONAL_HEAT_MIX_SOURCE = ("NRCan Comprehensive Energy Use Database (OEE), Industrial Sector, "
                            "Aggregated Industries, Table 1, 2023")


def _load_nrcan_table1(path=NRCAN_TABLE1_CSV):
    with open(path, newline="") as f:
        rows = list(_csv.DictReader(l for l in f if not l.startswith("#")))
    out = {r["region"]: {k: float(v) for k, v in r.items() if k != "region"} for r in rows}
    cols = next(iter(out.values())).keys()
    out["CA"] = {k: sum(r[k] for r in out.values()) for k in cols}
    return out


NRCAN_TABLE1_PJ = _load_nrcan_table1()


def _group_datasets(pj, include_refinery_fuels):
    """{group: [(activity, product, PJ)]} for one region; empty entries dropped."""
    hfo = pj["heavy_fuel_oil"] + (pj["still_gas_petcoke"] if include_refinery_fuels else 0.0)
    g = {
        "natural_gas": [(n, prod, w * pj["natural_gas"]) for n, prod, w in NG_INDUSTRIAL_HEAT],
        "bioenergy":   [(HEAT_MIX_DATASETS["bioenergy"][0][0], _H_OTHER, pj["wood_waste_pulping_liquor"])],
        "petroleum":   [("heat production, heavy fuel oil, at industrial furnace 1MW", _H_OTHER, hfo),
                        ("heat production, light fuel oil, at industrial furnace 1MW", _H_OTHER,
                         pj["diesel_lfo_kerosene"])],
        "coal_coke":   [("heat production, at hard coal industrial furnace 1-10MW", _H_OTHER, pj["coal"]),
                        ("heat production, at coal coke industrial furnace 1-10MW", _H_OTHER,
                         pj["coke_coke_oven_gas"])],
        "lpg_ngl":     [("heat production, propane, at industrial furnace >100kW", _H_OTHER, pj["lpg_ngl"])],
    }
    return {k: [e for e in v if e[2] > 0] for k, v in g.items() if sum(e[2] for e in v) > 0}


def _region_key(province):
    key = str(province).upper().replace("CA-", "")
    region = PROVINCE_TO_HEAT_MIX_REGION.get(key, key)
    if region not in NRCAN_TABLE1_PJ:
        raise KeyError(f"no industrial heat mix for '{province}'; "
                       f"known: {sorted(set(PROVINCE_TO_HEAT_MIX_REGION) | set(NRCAN_TABLE1_PJ))}")
    return region


# provinces and territories -> NRCan region
PROVINCE_TO_HEAT_MIX_REGION = {
    "QC": "QC", "ON": "ON", "MB": "MB", "SK": "SK", "AB": "AB",
    "BC": "BC_TER", "YT": "BC_TER", "NT": "BC_TER", "NU": "BC_TER",
    "NB": "ATL", "NS": "ATL", "PE": "ATL", "NL": "ATL",
}


def regional_heat_mix(province, include_refinery_fuels=None):
    """Group shares (sum 1) of the 2023 industrial combustion heat mix for a
    province or territory code ("QC", "BC", "CA-ON") or an NRCan region key
    ("ATL", "BC_TER", "CA")."""
    inc = INCLUDE_REFINERY_FUELS if include_refinery_fuels is None else include_refinery_fuels
    g = _group_datasets(NRCAN_TABLE1_PJ[_region_key(province)], inc)
    tot = {k: sum(e[2] for e in v) for k, v in g.items()}
    s = sum(tot.values())
    return {k: v / s for k, v in tot.items()}


def regional_heat_mix_datasets(province, include_refinery_fuels=None):
    """HEAT_MIX_DATASETS-like {group: [(activity, product, weight)]} with the
    region's own PJ as weights (for bw_lookup.resolve_heat_mix)."""
    inc = INCLUDE_REFINERY_FUELS if include_refinery_fuels is None else include_refinery_fuels
    return _group_datasets(NRCAN_TABLE1_PJ[_region_key(province)], inc)


REGIONAL_HEAT_MIX = {r: regional_heat_mix(r) for r in NRCAN_TABLE1_PJ}


# =============================================================================
# FUEL NITROGEN TO NOx
# =============================================================================
# Share of the fuel N emitted as NOx (expressed as NO2). Houshfar et al. (2012,
# Energies 5, 270-290) report 1 to 25 % in biomass combustion, decreasing as
# the fuel N content rises (see also Glarborg et al., 2003, Prog. Energy
# Combust. Sci. 29, 89-113). The central value (10 %) is assumed within that
# range.
NOX_CONV_RANGE = (0.01, 0.25)


NOX_CONV_CENTRAL = 0.10


# =============================================================================
# EMISSION LIMIT VALUES FOR HYDROCHAR COMBUSTION
# =============================================================================
# A permitted boiler cannot emit above its limits, so stack emissions are
#     E = min(composition-based estimate, ELV x dry flue gas volume at 6 % O2).
# ELVs: Directive (EU) 2015/2193 (Medium Combustion Plant Directive), Annex II,
# Part 2, Table 1, new plants of 1 to 5 MW (the scale of the ecoinvent
# template), dry gas at 6 % O2, 273.15 K, 101.3 kPa:
#   NOx   500 mg/Nm3
#   SO2   400 mg/Nm3 ("other solid fuels"; 200 for solid biomass. Hydrochar
#         from mixed wastes is not clearly "solid biomass" under Art. 3, so the
#         higher limit is used)
#   dust   50 mg/Nm3
# Dust before the cap is the coal furnace PM scaled by the ash ratio
# Ash_char / Ash_coal; the PM size split of the coal template is kept.
# The burdens of flue gas treatment (reagents, filter electricity) and sulfur
# retention in the ash are not included.
# ELV_CAP_ENABLED = False gives uncapped emissions.
ELV_CAP_ENABLED = True


ELV_KG_PER_NM3 = {"NOx": 500e-6, "SO2": 400e-6, "dust": 50e-6}


ELV_REF_O2_PCT = 6.0


# Total PM of the ecoinvent template, kg per MJ of heat:
# PM2.5 2.5e-5 + PM2.5-10 2.5e-5 + PM>10 1.25e-5.
PM_TOTAL_COAL_KG_PER_MJ = 6.25e-5


ASH_COAL_PCT = 10.0   # reference coal ash content (%) for the PM scaling


def dry_flue_gas_Nm3_per_kg(C, H, O, N, S, o2_ref_pct=ELV_REF_O2_PCT):
    """
    Dry flue gas (Nm3 per kg dry fuel) at the reference O2, from the
    ultimate analysis in wt % (dry basis). Complete combustion in air
    (21 % O2, 79 % N2); dry products CO2 + SO2 + fuel N2 + air N2, then the
    excess-air correction of Directive 2015/2193 Annex IV:
        V_ref = V_stoich x 21 / (21 - O2_ref)
    """
    c, h, o, n, s = (x * 10.0 for x in (C, H, O, N, S))      # g per kg
    o2_stoich = c / 12.0 + h / 4.0 + s / 32.0 - o / 32.0     # mol O2 per kg
    v0 = (c / 12.0 + s / 32.0 + n / 28.0 + (79.0 / 21.0) * o2_stoich) * 0.022414
    return v0 * 21.0 / (21.0 - o2_ref_pct)


def compute_enduse_quantities(inv, HC_dry_mass_kg, nox_conv=NOX_CONV_CENTRAL):
    """
    Heat delivered and stack emissions of the hydrochar boiler, per FU.
    HC_dry_mass_kg: bone-dry hydrochar per FU (HHV_char is a dry-basis value,
    so it applies to the dry mass, not to the dried product at 10 % moisture).
    """
    C = float(inv["C_char (%)"]); H = float(inv["H_char (%)"])
    O = float(inv["O_char (%)"]); N = float(inv["N_char (%)"])
    S = float(inv["S_char (%)"]); Ash = float(inv["Ash_char (%)"])
    HHV = float(inv["HHV_char (MJ/kg)"])
    for name, v in [("C", C), ("H", H), ("O", O), ("N", N), ("S", S), ("Ash", Ash), ("HHV", HHV)]:
        if not np.isfinite(v):
            raise ValueError(f"[hydrochar_enduse] property {name} is not finite ({v})")
    for name, v in [("C", C), ("H", H), ("O", O), ("N", N), ("S", S), ("Ash", Ash)]:
        if not 0.0 <= v <= 100.0:
            raise ValueError(f"[hydrochar_enduse] {name}_char = {v} % outside [0, 100]")
    if not 0.0 < HHV < 40.0:
        raise ValueError(f"[hydrochar_enduse] HHV_char = {HHV} MJ/kg outside (0, 40)")
    if not NOX_CONV_RANGE[0] <= float(nox_conv) <= NOX_CONV_RANGE[1]:
        raise ValueError(f"[hydrochar_enduse] nox_conv = {nox_conv} outside {NOX_CONV_RANGE}")
    m_dry = float(HC_dry_mass_kg)
    if m_dry < 0:
        raise ValueError(f"[hydrochar_enduse] negative hydrochar mass {m_dry}")

    CO2_bio_kg = m_dry * C / 100.0 * 44.0 / 12.0
    SO2_kg = m_dry * S / 100.0 * 64.0 / 32.0
    NOx_kg = m_dry * N / 100.0 * float(nox_conv) * 46.0 / 14.0   # as NO2
    ash_kg = m_dry * Ash / 100.0

    SF = HHV / HHV_COAL_MJ_PER_KG
    MJ_heat = m_dry * SF / COAL_KG_PER_MJ
    per_MJ = (lambda x: x / MJ_heat) if MJ_heat > 0 else (lambda x: 0.0)

    V_fg_Nm3 = m_dry * dry_flue_gas_Nm3_per_kg(C, H, O, N, S)
    PM_uncapped_kg = PM_TOTAL_COAL_KG_PER_MJ * MJ_heat * Ash / ASH_COAL_PCT
    if ELV_CAP_ENABLED:
        NOx_comb_kg = min(NOx_kg, ELV_KG_PER_NM3["NOx"] * V_fg_Nm3)
        SO2_comb_kg = min(SO2_kg, ELV_KG_PER_NM3["SO2"] * V_fg_Nm3)
        PM_comb_kg = min(PM_uncapped_kg, ELV_KG_PER_NM3["dust"] * V_fg_Nm3)
    else:
        NOx_comb_kg, SO2_comb_kg, PM_comb_kg = NOx_kg, SO2_kg, PM_uncapped_kg
    PM_scale = (PM_comb_kg / (PM_TOTAL_COAL_KG_PER_MJ * MJ_heat)) if MJ_heat > 0 else 0.0
    _conc = (lambda x: x / V_fg_Nm3 * 1e6) if V_fg_Nm3 > 0 else (lambda x: float("nan"))

    return dict(
        m_hc_dry_kg=m_dry, nox_conv=float(nox_conv),
        CO2_bio_kg=CO2_bio_kg, SO2_kg=SO2_kg, NOx_kg=NOx_kg, ash_kg=ash_kg,
        SubstitutionFactor_HC_COAL=SF, MJ_hydrochar=MJ_heat,
        SO2_per_MJ=per_MJ(SO2_comb_kg), NOx_per_MJ=per_MJ(NOx_comb_kg),
        CO2_bio_per_MJ=per_MJ(CO2_bio_kg), Ash_char_pct=Ash, PM_scale=PM_scale,
        V_fluegas_Nm3=V_fg_Nm3, NOx_comb_kg=NOx_comb_kg, SO2_comb_kg=SO2_comb_kg,
        PM_comb_kg=PM_comb_kg, PM_uncapped_kg=PM_uncapped_kg,
        NOx_uncapped_mgNm3=_conc(NOx_kg), SO2_uncapped_mgNm3=_conc(SO2_kg),
        PM_uncapped_mgNm3=_conc(PM_uncapped_kg),
        NOx_capped=bool(NOx_comb_kg < NOx_kg), SO2_capped=bool(SO2_comb_kg < SO2_kg),
        PM_capped=bool(PM_comb_kg < PM_uncapped_kg),
    )
