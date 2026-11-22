"""
bw_lookup.py
============
Deterministic Brightway lookups for background activities and elementary
flows.

Activities are found by a full scan of the database on exact names, then
selected by location priority (for example CA-QC, CA, RoW, GLO), with ties
broken by sorted name, so the same dataset is chosen in every session. Each
selection is printed. Elementary flows are found by an exact (name,
categories) lookup; when location priorities are given, the location-named
flow of a regionalized biosphere is used if it exists.
"""

import bw2data as bd

_activity_name_cache = {}


def _real_search(db_name, query, limit=200):
    """All activities whose lower-case name contains every word of query."""
    if db_name not in _activity_name_cache:
        _activity_name_cache[db_name] = [(a["name"].lower(), a) for a in bd.Database(db_name)]
    words = query.lower().split()
    out = []
    for name_l, a in _activity_name_cache[db_name]:
        if all(w in name_l for w in words):
            out.append(a)
            if len(out) >= limit:
                break
    return out


def find_process(ei, query, priority, name_filter=None, fallback_any=False):
    """Full scan -> exact name matches, else names starting with query ->
    optional name_filter -> first location in priority -> sorted name.
    No location of the priority list: LookupError, unless fallback_any=True
    (then the alphabetically first location is used, e.g. "AE", with a warning)."""
    cand = _real_search(ei.name, query, limit=10 ** 9)
    ql = query.lower()
    pool = [r for r in cand if r["name"] == query]
    if not pool:
        pool = [r for r in cand if r["name"].lower().startswith(ql)] or list(cand)
    if name_filter:
        pool = [r for r in pool if name_filter.lower() in r["name"].lower()]
    pool = sorted(pool, key=lambda r: (r["name"], r.get("location", "")))
    for loc in priority:
        for r in pool:
            if r.get("location", "") == loc:
                print(f"  [background] '{query}' -> '{r['name']}' [{loc}]")
                return r
    if fallback_any and pool:
        r = pool[0]
        print(f"  [background] WARNING '{query}': no priority location, "
              f"using '{r['name']}' [{r.get('location', '?')}]")
        return r
    if pool:
        raise LookupError(f"'{query}': no dataset in locations {list(priority)}; "
                          f"available: {sorted({r.get('location', '?') for r in pool})}")
    raise ValueError(f"No process found for query: '{query}'")


def find_exact(ei, name, priority, product=None):
    """Exact activity name (and reference product if given), first location
    in priority. Raises LookupError if absent."""
    results = [r for r in _real_search(ei.name, name, limit=5000) if r["name"] == name]
    if product is not None:
        results = [r for r in results if r.get("reference product") == product]
    results = sorted(results, key=lambda r: r.get("location", ""))
    for loc in priority:
        for r in results:
            if r.get("location", "") == loc:
                print(f"  [dataset] '{name}' -> [{loc}]")
                return r
    if results:
        print(f"  [dataset] WARNING '{name}': no priority location, using "
              f"[{results[0].get('location', '?')}]")
        return results[0]
    raise LookupError(name)


def resolve_heat_mix(ei, datasets, groups, priority):
    """{group: [(activity, weight normalized within the group)]} for the
    hydrochar_enduse.HEAT_MIX_DATASETS entries of the given groups."""
    out = {}
    for grp in groups:
        entries = datasets[grp]
        wsum = float(sum(w for _, _, w in entries))
        out[grp] = [(find_exact(ei, n, priority, product=prod), w / wsum) for n, prod, w in entries]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ELEMENTARY FLOW LOOKUP
# ─────────────────────────────────────────────────────────────────────────────
# In a regionalized project, SO2, NOx, PM2.5, NH3, etc. exist as location-named
# flows in "biosphere3_spatialized_flows" (e.g. "Sulfur dioxide, CA-QC", same
# categories as the generic flow). Lookup order: "<name>, <location>" with the
# same categories in the spatial database (locations tried in order), then the
# generic flow in biosphere3 with those categories, then the generic flow of
# the top-level compartment.
SPATIAL_FLOW_DB = "biosphere3_spatialized_flows"


GENERIC_FLOW_DB = "biosphere3"


# Locations tried for foreground elementary flows. Empty: generic flows only.
# The case study uses ecoinvent 3.12 cut-off with generic (non-regionalized) flows,
# so the default (global) IMPACT World+ characterization factors apply.
CORE_FLOW_LOCATIONS = []


_FLOW_INDEX = {}


def _flow_index(db_name):
    """{(name, categories): flow} for one biosphere database, cached."""
    if db_name not in _FLOW_INDEX:
        idx = {}
        if db_name in bd.databases:
            for a in bd.Database(db_name):
                idx.setdefault((a["name"], tuple(a.get("categories") or ())), a)
        _FLOW_INDEX[db_name] = idx
    return _FLOW_INDEX[db_name]


def get_flow(name, categories=("air",), locations=None):
    """Elementary flow by exact name and categories (see lookup order above)."""
    cats = tuple(categories)
    if locations:
        sidx = _flow_index(SPATIAL_FLOW_DB)
        for loc in locations:
            f = sidx.get((f"{name}, {loc}", cats))
            if f is not None:
                return f
    gidx = _flow_index(GENERIC_FLOW_DB)
    f = gidx.get((name, cats)) or gidx.get((name, cats[:1]))
    if f is None:
        raise ValueError(f"Elementary flow not found: '{name}' {cats}")
    return f


def base_flow_name(flow):
    """'Particulate Matter, < 2.5 um, CA-QC' -> 'Particulate Matter, < 2.5 um'."""
    if flow.key[0] == SPATIAL_FLOW_DB and ", " in flow["name"]:
        return flow["name"].rsplit(", ", 1)[0]
    return flow["name"]


def relocate_flow(flow, locations):
    """Same substance and compartment as `flow`, emitted at `locations`."""
    return get_flow(base_flow_name(flow), flow.get("categories") or ("air",), locations)


def _template_cats(template_act, base_name, default=("air", "urban air close to ground")):
    """Categories of the template's emission of base_name (default if absent)."""
    for exc in template_act.biosphere():
        if base_flow_name(exc.input) == base_name:
            return tuple(exc.input.get("categories") or default)
    return default


_EI_NAME_INDEX = {}


def _regional_equivalent(ei, name, priorities):
    """Exact-name activity in `ei`, first location in `priorities`; None if absent."""
    if ei.name not in _EI_NAME_INDEX:
        idx = {}
        for a in ei:
            idx.setdefault(a["name"], []).append(a)
        _EI_NAME_INDEX[ei.name] = idx
    cands = _EI_NAME_INDEX[ei.name].get(name, [])
    for loc in priorities:
        for a in sorted(cands, key=lambda a: a["location"]):
            if a["location"] == loc:
                return a
    return None

