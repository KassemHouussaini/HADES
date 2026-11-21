"""
lca_contributions.py
====================
Contribution analysis of a HADES system, by process group, as in Figure 5 of
the article. The foreground tree under the waste treatment activity is walked:
each background input counts amount x its unit score, each foreground emission
counts amount x its characterization factor. The categories sum to the LCA
score of the waste treatment (checked by `contributions`).
"""
import bw2data as bd
from bw2calc import LCA

CATS = ["HTC and drying (electricity)", "Digester heat (natural gas)",
        "Dilution water and wastewater treatment",
        "Direct emissions (HTC off-gas, digestion, upgrading)",
        "Hydrochar combustion and ash disposal",
        "Other (AD and dewatering electricity, PSA)",
        "Credit: industrial heat mix", "Credit: natural gas heat",
        "HTC and drying (fuel heat)"]

_unit_score, _cf = {}, {}


def _db_stamp(db_name, _seen=None):
    """((database, modified), ...) for db_name and the databases it depends
    on. bw2data updates "modified" whenever an activity or exchange of the
    database is saved or deleted (bw2data 3.6 and 4.x: databases.set_dirty),
    so a rebuilt foreground or a re-imported background changes the stamp."""
    seen = set() if _seen is None else _seen
    if db_name in seen or db_name not in bd.databases:
        return ()
    seen.add(db_name)
    meta = bd.databases[db_name]
    out = ((db_name, meta.get("modified")),)
    for dep in sorted(meta.get("depends") or []):
        out += _db_stamp(dep, seen)
    return out


def unit_score(act, method):
    """Score of 1 unit of an activity. Cached per project, activity, method
    and database stamp (_db_stamp): a score is reused only while the
    activity's database and its dependencies are unchanged, so a change of
    background context or a rebuilt foreground never returns a stale score."""
    k = (bd.projects.current, act.key, tuple(method), _db_stamp(act.key[0]))
    if k not in _unit_score:
        lca = LCA({act: 1}, method); lca.lci(); lca.lcia()
        _unit_score[k] = lca.score
    return _unit_score[k]


def clear_cache():
    """Drop all cached unit scores and characterization factors. Not needed
    for correctness (the cache keys carry the project and database stamps);
    use it to free memory between large batches."""
    _unit_score.clear()
    _cf.clear()


def cf_of(flow_key, method):
    """Characterization factor of a biosphere flow (0 if the method has none)."""
    mk = (bd.projects.current, tuple(method))
    if mk not in _cf:
        _cf[mk] = {(tuple(k) if isinstance(k, (list, tuple)) else bd.get_activity(k).key): v
                   for k, v, *_ in bd.Method(method).load()}
    return _cf[mk].get(flow_key, 0.0)


def category(parent_code, exc):
    name, t = exc.input["name"].lower(), exc["type"]
    if parent_code == "qc_industrial_heat_mix":
        return CATS[6]
    if parent_code == "heat_naturalgas":
        return CATS[7]
    if parent_code == "heat_prod_hydrochar" or "wood ash" in name:
        return CATS[4]
    if t == "biosphere":
        return CATS[3]
    if "tap water" in name or "wastewater" in name:
        return CATS[2]
    if parent_code in ("htc_reactor", "hydrochar_drying") and "electricity" in name:
        return CATS[0]
    if parent_code in ("htc_reactor", "hydrochar_drying") and "heat" in name:
        return CATS[8]
    if parent_code == "anaerobic_digestion" and "heat" in name:
        return CATS[1]
    return CATS[5]


def contributions(wt, foreground_db, method, total=None, rel_tol=1e-4):
    """Impact of `wt` by category. If `total` (the LCA score) is given, the
    categories must sum to it within rel_tol, otherwise ValueError."""
    out = dict.fromkeys(CATS, 0.0)

    def walk(act, amount, parent_code):
        for e in act.exchanges():
            if e["type"] == "production" or e["amount"] == 0:
                continue
            a, inp = amount * e["amount"], e.input
            if e["type"] == "biosphere":
                out[category(parent_code, e)] += a * cf_of(inp.key, method)
            elif inp.key[0] == foreground_db:
                walk(inp, a, inp.key[1])
            else:
                out[category(parent_code, e)] += a * unit_score(inp, method)

    walk(wt, 1.0, "waste_treatment_1kg")
    if total is not None:
        s = sum(out.values())
        if abs(s - total) > rel_tol * max(1e-12, abs(total)):
            raise ValueError(f"[contributions] categories sum {s:.6g} != LCA score {total:.6g}")
    return out
