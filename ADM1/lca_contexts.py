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

Biomethane displaces heat only, from the regional "market for heat, district
or industrial, natural gas", which follows location_priority.

Attributional framing: average mixes for supply and for the displaced
products. Pure Python, no Brightway import.

Usage
    ctxp = lca_contexts.get_preset("ON")
    LOC = ctxp["location_priority"]
    shares = sb.resolve_heat_mix_shares(ctxp["heat_mix_shares"])
    heat_mix = bwl.resolve_heat_mix(ei, hce.HEAT_MIX_DATASETS, shares.keys(), LOC)
    sb.SystemContext(..., **lca_contexts.ctx_kwargs(ctxp), heat_mix_components_qc=heat_mix)
"""

import copy
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

_NRCAN = hce.REGIONAL_HEAT_MIX_SOURCE + ", electricity and other fuels excluded, renormalized"


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
        biomethane_use="heat", note=note)


PRESETS = {code: _preset(code, *v) for code, v in _PROVINCES.items()}

# Paper 1 configuration, kept for reproducibility (frozen regression)
PRESETS["QC_PAPER1"] = dict(
    key="QC_PAPER1", name="Quebec (Paper 1)", location_country="CA", location_province="QC",
    location_code="CA-QC", location_priority=["CA-QC", "CA", "RoW", "GLO"],
    heat_mix_shares=None,
    heat_mix_source=("Whitmore and Pineau (2026), Etat de l'energie au Quebec 2026, "
                     "Graphique 25 (industry, 2023), electricity excluded, renormalized"),
    biomethane_use="heat", note="Paper 1 case study configuration.")

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


def ctx_kwargs(preset):
    """SystemContext keyword arguments set by the preset."""
    return dict(location_country=preset["location_country"],
                location_province=preset["location_province"],
                location_code=preset["location_code"],
                location_priority=list(preset["location_priority"]),
                heat_mix_shares=preset["heat_mix_shares"])
