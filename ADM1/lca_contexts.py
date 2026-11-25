"""
lca_contexts.py
===============
Regional LCA context presets (Paper 2). One preset bundles everything that
depends on where the plant is:

  location_priority  order of locations for every background dataset found
                     by location (grid electricity, digester and backup heat,
                     natural gas heat market, tap water, wastewater, the fuel
                     datasets of the heat mix)
  heat_mix_shares    industrial combustion heat mix displaced by hydrochar
                     (None = Paper 1 Quebec mix, hce.QC_HEAT_MIX; a key of
                     hce.REGIONAL_HEAT_MIX = NRCan CEUD 2023)

Biomethane displaces heat only. Paper 1 (QC_PAPER1): ecoinvent "market for
heat, district or industrial, natural gas". Province presets: natural gas
boilers and furnaces (hce.NG_INDUSTRIAL_HEAT, no CHP), also used for the
natural gas share of the heat mix.

Attributional framing: average mixes for supply and for the displaced
products. Pure Python, no Brightway import.

Usage
    ctxp = lca_contexts.get_preset("ON")
    LOC = ctxp["location_priority"]
    shares = sb.resolve_heat_mix_shares(ctxp["heat_mix_shares"])
    heat_mix = bwl.resolve_heat_mix(ei, lca_contexts.heat_mix_datasets(ctxp), shares.keys(), LOC)
    sb.SystemContext(..., **lca_contexts.ctx_kwargs(ctxp), heat_mix_components_qc=heat_mix)
"""

import copy
import csv
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("p2_hce_contexts", os.path.join(_HERE, "hydrochar_enduse.py"))
hce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hce)


# ecoinvent location codes of the Canadian provinces and territories
# (Yukon is CA-YK and Newfoundland and Labrador CA-NF in ecoinvent). Datasets missing for a province fall back
# to CA, then RoW, then GLO; bw_lookup prints every fallback.
_PROVINCES = {
    #      ecoinvent code   name                         NRCan heat-mix region
    "QC": ("CA-QC", "Quebec",                    "QC"),
    "ON": ("CA-ON", "Ontario",                   "ON"),
    "MB": ("CA-MB", "Manitoba",                  "MB"),
    "SK": ("CA-SK", "Saskatchewan",              "SK"),
    "AB": ("CA-AB", "Alberta",                   "AB"),
    "BC": ("CA-BC", "British Columbia",          "BC_TER"),
    "NB": ("CA-NB", "New Brunswick",             "ATL"),
    "NS": ("CA-NS", "Nova Scotia",               "ATL"),
    "PE": ("CA-PE", "Prince Edward Island",      "ATL"),
    "NL": ("CA-NF", "Newfoundland and Labrador", "ATL"),
    "YT": ("CA-YK", "Yukon",                     "BC_TER"),
    "NT": ("CA-NT", "Northwest Territories",     "BC_TER"),
    "NU": ("CA-NU", "Nunavut",                   "BC_TER"),
}

_NRCAN = (hce.REGIONAL_HEAT_MIX_SOURCE + ", electricity, other fuels"
          + ("" if hce.INCLUDE_REFINERY_FUELS else ", still gas and petroleum coke") + " excluded, renormalized")


# Ambient temperature per province: annual daily-average air temperature,
# ECCC Canadian Climate Normals 1991-2020, station of the largest city
# (data/eccc_normals_1991_2020_annual_mean_T.csv, downloaded from
# climate.weather.gc.ca/climate_normals, bulk download, 2026-11-25). Used as
# heat_integration.T_AMB_C (raw feed, tap water, outside of the digester).
# Below T_AMB_MIN_C a liquid feed or water main cannot be at the air
# temperature: those presets carry the value but ambient_overrides() refuses
# them.
AMBIENT_CSV = os.path.join(_HERE, "data", "eccc_normals_1991_2020_annual_mean_T.csv")
AMBIENT_SOURCE = ("ECCC Canadian Climate Normals 1991-2020, annual daily-average "
                  "temperature, station of the largest city")
T_AMB_MIN_C = 0.0


def _read_ambient(path=AMBIENT_CSV):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["province"]: r for r in csv.DictReader(f)}


_AMBIENT = _read_ambient()


def _preset(code, eic, name, region):
    note = ""
    if region == "ATL":
        note = "NRCan publishes the Atlantic provinces as one group: heat mix of the group."
    elif region == "BC_TER":
        note = "NRCan publishes BC with the Territories: heat mix of the group."
    return dict(
        key=code, name=name, location_country="CA", location_province=code,
        location_code=eic, location_priority=[eic, "CA", "RoW", "GLO"],
        heat_mix_shares=region, heat_mix_source=_NRCAN + f" (region {region})",
        biomethane_use="heat", ng_heat="boilers_furnaces", note=note,
        T_amb_C=float(_AMBIENT[code]["T_annual_mean_C"]),
        T_amb_source=f"{AMBIENT_SOURCE}: {_AMBIENT[code]['eccc_location_name']} "
                     f"(climate IDs {_AMBIENT[code]['climate_ids']}, code {_AMBIENT[code]['normals_code']})")


PRESETS = {code: _preset(code, *v) for code, v in _PROVINCES.items()}

# Paper 1 configuration, kept for reproducibility (frozen regression)
PRESETS["QC_PAPER1"] = dict(
    key="QC_PAPER1", name="Quebec (Paper 1)", location_country="CA", location_province="QC",
    location_code="CA-QC", location_priority=["CA-QC", "CA", "RoW", "GLO"],
    heat_mix_shares=None,
    heat_mix_source=("Whitmore and Pineau (2026), Etat de l'energie au Quebec 2026, "
                     "Graphique 25 (industry, 2023), electricity excluded, renormalized"),
    biomethane_use="heat", ng_heat="market", note="Paper 1 case study configuration.",
    T_amb_C=None, T_amb_source="Paper 1 model default (heat_integration.T_AMB_C)")

DEFAULT_PRESET = "QC_PAPER1"


def get_preset(key=DEFAULT_PRESET):
    """Copy of a preset: a province or territory code ("ON", "CA-ON") or "QC_PAPER1"."""
    k = str(key).upper()
    if k != "QC_PAPER1":
        k = k.replace("CA-", "")
        k = {"YK": "YT", "NF": "NL"}.get(k, k)
    if k not in PRESETS:
        raise KeyError(f"no LCA context preset '{key}'; known: {sorted(PRESETS)}")
    return copy.deepcopy(PRESETS[k])


def heat_mix_shares(preset):
    """Group shares of the preset's displaced heat mix."""
    spec = preset["heat_mix_shares"]
    return dict(hce.QC_HEAT_MIX) if spec is None else hce.regional_heat_mix(spec)


def heat_mix_datasets(preset):
    """{group: [(activity, product, weight)]} to resolve with bw_lookup.resolve_heat_mix:
    the Paper 1 weights (ecoinvent CA-QC market) or the region's NRCan PJ."""
    spec = preset["heat_mix_shares"]
    return hce.HEAT_MIX_DATASETS if spec is None else hce.regional_heat_mix_datasets(spec)


def ng_heat_datasets(preset):
    """Natural gas heat displaced by biomethane: None = ecoinvent "market for
    heat, district or industrial, natural gas" (Paper 1); otherwise the
    (activity, product, weight) list of hce.NG_INDUSTRIAL_HEAT."""
    return None if preset.get("ng_heat") == "market" else list(hce.NG_INDUSTRIAL_HEAT)


def ctx_kwargs(preset):
    """SystemContext keyword arguments set by the preset."""
    return dict(location_country=preset["location_country"],
                location_province=preset["location_province"],
                location_code=preset["location_code"],
                location_priority=list(preset["location_priority"]),
                heat_mix_shares=preset["heat_mix_shares"])


def ambient_overrides(preset):
    """{"T_AMB_C": value} for SystemContext.hi_overrides, from the preset's
    climate normal; {} for QC_PAPER1 (Paper 1 default kept). Raises for a
    preset below T_AMB_MIN_C (Northwest Territories, Nunavut)."""
    t = preset.get("T_amb_C")
    if t is None:
        return {}
    if t < T_AMB_MIN_C:
        raise ValueError(f"[ambient] {preset['key']}: annual mean air temperature {t} C is below "
                         f"{T_AMB_MIN_C} C and cannot stand for the temperature of the liquid feed "
                         f"and tap water; set T_AMB_C explicitly")
    return {"T_AMB_C": float(t)}
