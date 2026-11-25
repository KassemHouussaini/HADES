"""
sweep_engine.py
===============
Engine for Paper 2 sweeps (many HADES runs). Used by notebooks/sweep.ipynb.

Design
------
1. Workers (parallel, no Brightway writes). Each worker reads ecoinvent
   (read-only, safe for many processes) and builds the foreground in memory
   (MemAct). The ADM1 integration (97 % of the run time) and the HTC surrogate
   load are cached per worker, so one process point is evaluated for every
   LCA context and heat configuration at the cost of one ADM1 run. Caps
   recorded inside ADM1 are replayed on a cache hit, so caps_active stays
   complete.
2. Scores. Impacts = walk of the in-memory foreground (same categories as
   lca_contributions) x a unit-score table of the background activities and
   the characterization factors of the foreground emissions, both computed
   once beforehand (precompute_scores, serial, Brightway).
3. Check. validate_point() rebuilds a point in a real Brightway foreground
   and compares the engine scores with a full LCA.

Each finished point is written to its own CSV file in the output folder, so
an interrupted sweep resumes where it stopped.

Designs: lhs_design() (Latin hypercube, McKay, Beckman & Conover 1979,
Technometrics 21:239) and morris_design() / morris_indices() (Morris 1991,
Technometrics 33:161; mu* of Campolongo, Cariboni & Saltelli 2007, Environ.
Model. Softw. 22:1509).
"""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ML_MODULE = os.path.join(ROOT, "ML training scripts and models", "htc_surrogate.py")

FG_DB = "HADES_sweep_memory"
# foreground roles: SystemContext field -> (code, name, unit), codes as in HADES.ipynb
FG_ROLES = {
    "htc_reactor": ("htc_reactor", "HTC reactor", "unit"),
    "drying": ("hydrochar_drying", "Hydrochar drying", "unit"),
    "heat_prod": ("heat_prod_hydrochar", "Heat production from hydrochar", "megajoule"),
    "sub_heat_coal": ("substitute_heat_coal", "Hard coal heat", "megajoule"),
    "ad_process": ("anaerobic_digestion", "Anaerobic digestion (ADM1)", "unit"),
    "dewatering": ("digestate dewatering", "Digestate dewatering", "unit"),
    "biogas_upgrading": ("biogas_upgrading", "Biogas upgrading (PSA)", "unit"),
    "heat_biomethane": ("heat_biomethane", "Heat production from biomethane", "megajoule"),
    "heat_naturalgas": ("heat_naturalgas", "Natural gas heat (substituted)", "megajoule"),
    "qc_heat_mix": ("qc_industrial_heat_mix", "Industrial combustion heat mix (substituted)", "megajoule"),
    "waste_treatment": ("waste_treatment_1kg", "Waste treatment of organic waste (HTC-AD system)", "kilogram"),
}
NG_BOILER = "heat production, natural gas, at boiler condensing modulating >100kW"

# heat configurations: consumer -> "electricity" | "ng" | "own_biogas"; None = Paper 1
HEAT_CONFIGS = {
    "paper1": None,
    "all_ng": dict(htc="ng", dryer="ng", ad="ng"),
    "own_biogas": dict(htc="own_biogas", dryer="own_biogas", ad="own_biogas"),
}


def _load(name, fname):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _lcx():
    return _load("lca_contexts", "lca_contexts.py")


def key_str(act):
    """'database|code' of a Brightway activity or flow."""
    k = act.key
    return f"{k[0]}|{k[1]}"


# =============================================================================
# IN-MEMORY FOREGROUND
# =============================================================================
class MemExc(dict):
    def __init__(self, act, input, amount, type, unit=None):
        super().__init__(amount=float(amount), type=type, unit=unit)
        self.act, self.input = act, input

    def save(self):
        if not any(e is self for e in self.act._exc):
            self.act._exc.append(self)

    def delete(self):
        self.act._exc = [e for e in self.act._exc if e is not self]


class MemAct(dict):
    """Foreground activity held in memory (the subset of the Brightway
    Activity interface that system_builder uses)."""
    def __init__(self, code, name, unit, location="CA-QC"):
        super().__init__(name=name, unit=unit, location=location, code=code, database=FG_DB)
        self.key = (FG_DB, code)
        self._exc = []

    def new_exchange(self, input, amount, type, unit=None, **kw):
        return MemExc(self, input, amount, type, unit)

    def exchanges(self):
        return list(self._exc)

    def technosphere(self):
        return [e for e in self._exc if e["type"] == "technosphere"]

    def biosphere(self):
        return [e for e in self._exc if e["type"] == "biosphere"]

    def production(self):
        return []

    def save(self):
        pass


def memory_foreground():
    return {role: MemAct(code, name, unit) for role, (code, name, unit) in FG_ROLES.items()}


# =============================================================================
# PER-WORKER CACHES: ADM1 RESULTS (WITH CAP REPLAY) AND THE HTC SURROGATE
# =============================================================================
class Adm1Memo:
    """Cache of system_builder.run_adm1 keyed by its exact inputs."""
    def __init__(self, sb):
        self.sb, self.core = sb, sb._adm1_core
        self.store, self.hits, self.misses = {}, 0, 0
        self._orig = sb.run_adm1

    def _call(self, *args, **kwargs):
        k = hashlib.sha256(pickle.dumps((args, sorted(kwargs.items())), protocol=4)).hexdigest()
        if k in self.store:
            self.hits += 1
            out, events = self.store[k]
            for ev in events:
                self.core.record_cap(*ev)
            return copy.deepcopy(out)
        self.misses += 1
        events, orig_rc = [], self.core.record_cap

        def logging_record_cap(name, raw, used, kind="correlation"):
            events.append((name, raw, used, kind))
            return orig_rc(name, raw, used, kind)

        self.core.record_cap = logging_record_cap
        try:
            out = self._orig(*args, **kwargs)
        finally:
            self.core.record_cap = orig_rc
        self.store[k] = (copy.deepcopy(out), events)
        return out

    def install(self):
        self.sb.run_adm1 = self._call
        return self

    def clear(self):
        self.store.clear()


def install_ml_cache(sb):
    """Load the HTC surrogate once per worker instead of once per build."""
    orig, cache = sb._load_ml_module, {}

    def cached(ml_module):
        if ml_module not in cache:
            cache[ml_module] = orig(ml_module)
        return cache[ml_module]

    sb._load_ml_module = cached


# =============================================================================
# LCA CONTEXT (same datasets as HADES.ipynb, section 2)
# =============================================================================
def resolve_context(ei, preset_key, sb, lcx):
    """Background datasets, heat mix and natural gas credit of one preset."""
    bwl = sb.bwl
    p = lcx.get_preset(preset_key)
    LOC = p["location_priority"]
    find = lambda q, prio=LOC, nf=None: bwl.find_process(ei, q, prio, name_filter=nf)
    bg = dict(
        elec=find("market for electricity, medium voltage", nf="market for electricity, medium voltage"),
        coal_heat=find("heat production, at hard coal industrial furnace 1-10MW", nf="hard coal industrial furnace"),
        ad_heat=find(NG_BOILER, nf=NG_BOILER),
        ng_heat_market=find("market for heat, district or industrial, natural gas",
                            nf="market for heat, district or industrial, natural gas"),
        biomethane_heat_ref=find("heat production, biomethane, at boiler condensing modulating",
                                 ["RoW", "GLO"] + LOC, nf="biomethane"),
        tap_water=find("market for tap water", nf="market for tap water"),
        wastewater_treat=find("treatment of wastewater, average", nf="treatment of wastewater, average"),
        psa_ref=find("biogas purification to biomethane by pressure swing adsorption", ["RoW", "GLO"] + LOC),
    )
    bg["ad_elec"] = bg["drying_heat"] = bg["elec"]
    shares = lcx.heat_mix_shares(p)
    heat_mix = bwl.resolve_heat_mix(ei, lcx.heat_mix_datasets(p), shares.keys(), LOC)
    ng = lcx.ng_heat_datasets(p)
    ng_comp = bwl.resolve_heat_mix(ei, {"ng": ng}, ["ng"], LOC)["ng"] if ng else None
    try:
        ash = bwl.find_exact(ei, "market for wood ash mixture, pure", LOC)
    except LookupError:
        ash = None
    return dict(preset=p, bg=bg, heat_mix=heat_mix, ng_comp=ng_comp, ash=ash,
                co2_air=bwl.get_flow("Carbon dioxide, non-fossil", ("air",), bwl.CORE_FLOW_LOCATIONS),
                co_air=bwl.get_flow("Carbon monoxide, non-fossil", ("air",), bwl.CORE_FLOW_LOCATIONS))


def heat_supply_for(config, bg):
    hs = HEAT_CONFIGS[config] if isinstance(config, str) else config
    if hs is None:
        return None, None
    src = {"electricity": "electricity", "ng": bg["ad_heat"], "own_biogas": "own_biogas"}
    return {c: src[v] for c, v in hs.items()}, bg["ad_heat"]


# =============================================================================
# ONE BUILD
# =============================================================================
def make_context(sb, ei, bios, rc, point, heat_config, fg):
    """SystemContext for one point (process inputs and sampled parameters),
    one resolved LCA context `rc` and one heat configuration."""
    feed = dict(point["feed"])
    feed.setdefault("O_feed", 100 - sum(feed[k] for k in ("C_feed", "H_feed", "N_feed", "S_feed", "Ash_feed")))
    prm = point.get("params", {})
    hs, backup = heat_supply_for(heat_config, rc["bg"])
    # ambient temperature: a sampled T_AMB_C wins; otherwise the province's
    # climate normal (point["ambient"] = "province", default) or the model
    # default 10 C ("default")
    hio = dict(prm.get("hi_overrides") or {})
    if "T_AMB_C" not in hio and point.get("ambient", "province") == "province":
        hio.update(_lcx().ambient_overrides(rc["preset"]))
    kw = dict(
        ei=ei, bios=bios, FOREGROUND_DB=FG_DB, **rc["bg"], co2_air=rc["co2_air"], co_air=rc["co_air"], **fg,
        PATHWAY=point["pathway"], m_feedstock=1000.0, default_params=feed,
        ml_params={**feed, "T": point["T"], "RT": point["RT"]},
        T_HTC=point["T"], RT_HTC=point["RT"], target_moisture_HTC=point.get("target_moisture_HTC", 0.85),
        MC_char=prm.get("MC_char", point.get("MC_char", 0.35)),
        target_MC_char_dry=point.get("target_MC_char_dry", 0.10),
        COD_conc_target=point.get("COD_target", 10.0), ISR_batch=point.get("ISR_batch", 2.0),
        feedstock_name=point["feedstock_type"],
        b_feed_overrides=prm.get("b_feed_overrides") or None,
        pw_ch4_multiplier=prm.get("pw_ch4_multiplier", 1.0),
        heat_supply=hs, heat_backup=backup,
        heat_mix_components_qc=rc["heat_mix"], ng_heat_components=rc["ng_comp"], ash_disposal=rc["ash"],
        hi_overrides=hio or None, u_fc=prm.get("u_fc"),
        foreground_activities=list(fg.values()),
        **sb_ctx_location(rc),
    )
    for k in ("eps_HR", "nox_conv", "own_biogas_heat_eff"):
        if k in prm:
            kw[k] = prm[k]
    return sb.SystemContext(**kw)


def sb_ctx_location(rc):
    p = rc["preset"]
    return dict(location_country=p["location_country"], location_province=p["location_province"],
                location_code=p["location_code"], location_priority=list(p["location_priority"]),
                heat_mix_shares=p["heat_mix_shares"])


def build_point(sb, ei, bios, rc, point, heat_config, ml_module=ML_MODULE, quiet=True):
    """Spin-up and build in memory; returns (waste_treatment MemAct, inventory dict)."""
    fg = memory_foreground()
    ctx = make_context(sb, ei, bios, rc, point, heat_config, fg)
    HRT, TS = point.get("HRT", 30), point.get("TS_target", 0.08)
    with contextlib.ExitStack() as stack:
        if quiet:                                   # build log and ADM1 progress bars
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        ss = sb.spinup_ss_state(ctx, pathway=point["pathway"], adm1_HRT_days=HRT, t_spinup=400,
                                TS_target=TS, verbose=False, ml_module=ml_module)
        wt, _, adm1_out, _, inv = sb.build_system(
            ctx, ml_module=ml_module, adm1_mode="continuous", adm1_HRT_days=HRT,
            adm1_ss_state=ss, ISR_batch=point.get("ISR_batch", 2.0), TS_target=TS)
    wt["_pH"] = (adm1_out.get("ss_state") or {}).get("pH")
    return wt, inv


# =============================================================================
# SCORING WITH PRECOMPUTED UNIT SCORES
# =============================================================================
def score(wt, scores, category, cats):
    """{method: {category: impact}} and the set of background keys missing
    from `scores` (dict with "unit_scores" {key: {method: s}} and "cfs"
    {method: {flow key: cf}})."""
    methods = list(scores["cfs"])
    out = {m: dict.fromkeys(cats, 0.0) for m in methods}
    missing = set()

    def walk(act, amount, parent_code):
        for e in act.exchanges():
            if e["type"] == "production" or e["amount"] == 0:
                continue
            a, inp = amount * e["amount"], e.input
            if e["type"] == "biosphere":
                c, k = category(parent_code, e), key_str(inp)
                for m in methods:
                    out[m][c] += a * scores["cfs"][m].get(k, 0.0)
            elif getattr(inp, "key", (None,))[0] == FG_DB:
                walk(inp, a, inp.key[1])
            else:
                k = key_str(inp)
                us = scores["unit_scores"].get(k)
                if us is None:
                    missing.add(k)
                    continue
                c = category(parent_code, e)
                for m in methods:
                    out[m][c] += a * us[m]

    walk(wt, 1.0, "waste_treatment_1kg")
    return out, missing


def background_keys(wt):
    """All background activity keys and biosphere flow keys under wt."""
    acts, flows = {}, {}

    def walk(act):
        for e in act.exchanges():
            if e["type"] == "production":
                continue
            inp = e.input
            if e["type"] == "biosphere":
                flows[key_str(inp)] = inp
            elif getattr(inp, "key", (None,))[0] == FG_DB:
                walk(inp)
            else:
                acts[key_str(inp)] = inp

    walk(wt)
    return acts, flows


# =============================================================================
# RESULT ROW
# =============================================================================
META_KEYS = ["CH4_mass_AD", "biomethane_m3", "MJ_hydrochar", "HC_dry_yield", "PW_yield",
             "E_HTC_elec_kWh", "E_dryer_heat_kWh", "Q_heat_AD_MJ", "CH4_own_use_MJ", "own_biogas_heat_eff",
             "heat_demand_htc_MJ", "heat_demand_dryer_MJ", "heat_demand_ad_MJ",
             "heat_shortfall_htc_MJ", "heat_shortfall_dryer_MJ", "heat_shortfall_ad_MJ",
             "pw_ch4_multiplier_effective", "adm1_fD", "adm1_f_sol", "adm1_k_hyd",
             "n_caps_correlation", "n_caps_design", "caps_active", "qa_flags", "_pH"]


def result_row(point, context, heat_config, wt, inv, imp, missing, method_labels):
    row = dict(point_id=point["point_id"], pathway=point["pathway"], context=context,
               heat_config=heat_config, feedstock=point.get("feedstock_label", point["feedstock_type"]),
               feedstock_type=point["feedstock_type"], T=point["T"], RT=point["RT"],
               HRT=point.get("HRT", 30), sample_id=point.get("sample_id"))
    for k, v in point.get("x", {}).items():
        row[f"x_{k}"] = v
    for k in META_KEYS:
        row[k] = wt.get(k)
    row["Y_char"] = inv.get("Y_char (%)")
    row["HHV_char"] = inv.get("HHV_char (MJ/kg)")
    for k in ("in_training_range", "domain_violations", "ad_knn_distance", "in_domain_knn"):
        row[k] = inv.get(k)
    row["missing_keys"] = ";".join(sorted(missing))
    for m, lab in method_labels.items():
        tot = float("nan") if missing else sum(imp[m].values())
        row[f"{lab}"] = tot
        for c, v in imp[m].items():
            row[f"{lab} | {c}"] = v
    return row


# =============================================================================
# WORKER
# =============================================================================
_W = {}


def worker_init(project, ei_name, scores_path, contexts, method_labels, out_dir):
    """Pool initializer: one Brightway read connection, caches, score table."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import bw2data as bd
    bd.projects.set_current(project)
    sb = _load("system_builder", "system_builder.py")
    lcx = _load("lca_contexts", "lca_contexts.py")
    lcc = _load("lca_contributions", "lca_contributions.py")
    deps = [d for d in bd.databases[ei_name].get("depends", [])
            if "biosphere" in d.lower() and "spatialized" not in d.lower()]
    sb.bwl.GENERIC_FLOW_DB = deps[0] if deps else "biosphere3"
    ei, bios = bd.Database(ei_name), bd.Database(sb.bwl.GENERIC_FLOW_DB)
    with contextlib.redirect_stdout(io.StringIO()):
        rcs = {c: resolve_context(ei, c, sb, lcx) for c in contexts}
    scores = json.load(open(scores_path))
    _W.update(sb=sb, lcc=lcc, ei=ei, bios=bios, rcs=rcs, scores=scores, method_labels=method_labels,
              out_dir=out_dir, memo=Adm1Memo(sb).install())
    install_ml_cache(sb)


def point_file(out_dir, point):
    return os.path.join(out_dir, f"{point['point_id']}.csv")


def run_point(point, heat_configs):
    """All contexts x heat configurations of one point; writes one CSV."""
    import pandas as pd
    w = _W
    path = point_file(w["out_dir"], point)
    if os.path.exists(path):
        return path, "done"
    w["memo"].clear()
    t0, rows = time.time(), []
    for c, rc in w["rcs"].items():
        for h in heat_configs:
            try:
                wt, inv = build_point(w["sb"], w["ei"], w["bios"], rc, point, h)
                imp, missing = score(wt, w["scores"], w["lcc"].category, w["lcc"].CATS)
                rows.append(result_row(point, c, h, wt, inv, imp, missing, w["method_labels"]))
            except Exception as e:                       # keep the sweep going, record the failure
                rows.append(dict(point_id=point["point_id"], pathway=point["pathway"], context=c,
                                 heat_config=h, error=f"{type(e).__name__}: {e}"))
    for r in rows:
        r["seconds_point"] = round(time.time() - t0, 1)
        r["adm1_runs"] = w["memo"].misses
    tmp = path + ".tmp"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path, "ok"


def _run_point_star(args):
    return run_point(*args)


def run_sweep(points, heat_configs, contexts, method_labels, project, ei_name, scores_path,
              out_dir, workers=12, chunksize=1, progress=True):
    """Run all points in parallel (spawn start method, also on Windows).
    Points already written to out_dir are skipped."""
    import multiprocessing as mp
    os.makedirs(out_dir, exist_ok=True)
    todo = [p for p in points if not os.path.exists(point_file(out_dir, p))]
    if progress:
        print(f"{len(points) - len(todo)} of {len(points)} points already done; running {len(todo)} "
              f"on {workers} workers")
    if not todo:
        return
    os.environ["OMP_NUM_THREADS"] = "1"           # one thread per worker (NumPy, XGBoost)
    ctx = mp.get_context("spawn")
    t0, n = time.time(), 0
    with ctx.Pool(workers, initializer=worker_init,
                  initargs=(project, ei_name, scores_path, contexts, method_labels, out_dir)) as pool:
        for _ in pool.imap_unordered(_run_point_star, [(p, heat_configs) for p in todo], chunksize):
            n += 1
            if progress and (n % max(1, workers) == 0 or n == len(todo)):
                el = time.time() - t0
                print(f"  {n}/{len(todo)} points, {el / 60:.1f} min, "
                      f"about {el / n * (len(todo) - n) / 60:.0f} min left", flush=True)


def collect(out_dir):
    """All point files of a sweep as one DataFrame."""
    import glob
    import pandas as pd
    files = sorted(glob.glob(os.path.join(out_dir, "*.csv")))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


# =============================================================================
# PRECOMPUTE: UNIT SCORES AND CHARACTERIZATION FACTORS (serial, Brightway)
# =============================================================================
def precompute_scores(probe_points, heat_configs, contexts, methods, project, ei_name, path):
    """Build the probe points in memory for every context and heat
    configuration, collect every background activity and foreground flow they
    use, and store their unit scores and characterization factors in `path`
    (JSON). Run once per sweep design; re-run if a sweep reports missing keys."""
    import bw2data as bd
    bd.projects.set_current(project)
    sb = _load("system_builder", "system_builder.py")
    lcx = _load("lca_contexts", "lca_contexts.py")
    lcc = _load("lca_contributions", "lca_contributions.py")
    deps = [d for d in bd.databases[ei_name].get("depends", [])
            if "biosphere" in d.lower() and "spatialized" not in d.lower()]
    sb.bwl.GENERIC_FLOW_DB = deps[0] if deps else "biosphere3"
    ei, bios = bd.Database(ei_name), bd.Database(sb.bwl.GENERIC_FLOW_DB)
    memo = Adm1Memo(sb).install()
    install_ml_cache(sb)
    acts, flows = {}, {}
    with contextlib.redirect_stdout(io.StringIO()):
        rcs = {c: resolve_context(ei, c, sb, lcx) for c in contexts}
    for pt in probe_points:
        memo.clear()
        for c, rc in rcs.items():
            for h in heat_configs:
                wt, _ = build_point(sb, ei, bios, rc, pt, h)
                a, f = background_keys(wt)
                acts.update(a)
                flows.update(f)
        print(f"  probe {pt['point_id']}: {len(acts)} background activities, {len(flows)} flows so far")
    old = json.load(open(path)) if os.path.exists(path) else {"unit_scores": {}, "cfs": {}}
    us = old["unit_scores"]
    labels = {json.dumps(list(m)): m for m in methods}
    todo = [k for k in acts if k not in us or any(json.dumps(list(m)) not in us[k] for m in methods)]
    print(f"computing unit scores of {len(todo)} activities x {len(methods)} methods")
    t0 = time.time()
    for m in methods:
        vals = _unit_scores_one_method({k: acts[k] for k in todo}, m, lcc)
        for k, v in vals.items():
            us.setdefault(k, {})[json.dumps(list(m))] = v
        print(f"  {m[-1]}: done ({(time.time() - t0) / 60:.1f} min)")
    cfs = {}
    for mk, m in labels.items():
        cfs[mk] = {k: lcc.cf_of(f.key, m) for k, f in flows.items()}
        cfs[mk].update({k: v for k, v in old["cfs"].get(mk, {}).items() if k not in cfs[mk]})
    out = {"unit_scores": us, "cfs": cfs, "methods": [list(m) for m in methods],
           "contexts": list(contexts), "project": project, "ei_name": ei_name}
    json.dump(out, open(path, "w"))
    print(f"written {path}: {len(us)} activities, {sum(len(v) for v in cfs.values())} CFs")
    return out


def _unit_scores_one_method(acts, method, lcc):
    """Unit scores of many activities for one method. Fast path: one LCA,
    factorized once, re-solved per activity (redo_lcia, bw2calc 1.x and 2.x,
    demand by id or key); checked against lca_contributions.unit_score on two
    activities. Any failure or mismatch falls back to one LCA per activity."""
    from bw2calc import LCA
    items = list(acts.items())
    out = {}
    if not items:
        return out
    try:
        lca = LCA({items[0][1]: 1}, method)
        lca.lci()
        lca.lcia()
        out[items[0][0]] = float(lca.score)

        def redo(a):
            for dem in ({a.id: 1} if getattr(a, "id", None) is not None else None, {a: 1}, {a.key: 1}):
                if dem is None:
                    continue
                try:
                    lca.redo_lcia(dem)
                    return float(lca.score)
                except Exception:
                    continue
            raise RuntimeError("redo_lcia failed")

        for i, (k, a) in enumerate(items[1:]):
            v = redo(a)
            if i < 2 and not np.isclose(v, lcc.unit_score(a, method), rtol=1e-6, atol=1e-15):
                raise RuntimeError("fast unit score differs from a full LCA")
            out[k] = v
        return out
    except Exception as e:
        print(f"    fast path not used ({e}); one LCA per activity")
        return {k: float(lcc.unit_score(a, method)) for k, a in items}


def method_labels_for(methods_by_label):
    """{json method key: label} used for the result columns."""
    return {json.dumps(list(m)): lab for lab, m in methods_by_label.items()}


# =============================================================================
# VALIDATION AGAINST A FULL BRIGHTWAY LCA
# =============================================================================
def validate_point(point, context, heat_config, methods_by_label, project, ei_name, scores_path,
                   rel_tol=1e-4):
    """Engine score vs real foreground + bw2calc LCA for one point. Returns a
    DataFrame; raises AssertionError if any method differs by more than rel_tol."""
    import bw2data as bd
    import pandas as pd
    from bw2calc import LCA
    bd.projects.set_current(project)
    sb = _load("system_builder", "system_builder.py")
    lcx = _load("lca_contexts", "lca_contexts.py")
    lcc = _load("lca_contributions", "lca_contributions.py")
    deps = [d for d in bd.databases[ei_name].get("depends", [])
            if "biosphere" in d.lower() and "spatialized" not in d.lower()]
    sb.bwl.GENERIC_FLOW_DB = deps[0] if deps else "biosphere3"
    ei, bios = bd.Database(ei_name), bd.Database(sb.bwl.GENERIC_FLOW_DB)
    memo = Adm1Memo(sb).install()
    install_ml_cache(sb)
    scores = json.load(open(scores_path))
    with contextlib.redirect_stdout(io.StringIO()):
        rc = resolve_context(ei, context, sb, lcx)
    # engine
    wt_m, _ = build_point(sb, ei, bios, rc, point, heat_config)
    imp, missing = score(wt_m, scores, lcc.category, lcc.CATS)
    # real Brightway foreground
    name = f"HADES_sweep_check_{point['pathway']}"
    if name not in bd.databases:
        bd.Database(name).register()
    db = bd.Database(name)
    fg = {role: sb.get_or_create(db, code, nm, unit) for role, (code, nm, unit) in FG_ROLES.items()}
    ctx = make_context(sb, ei, bios, rc, point, heat_config, fg)
    ctx.FOREGROUND_DB, ctx.foreground_activities = name, None
    with contextlib.redirect_stdout(io.StringIO()):
        ss = sb.spinup_ss_state(ctx, pathway=point["pathway"], adm1_HRT_days=point.get("HRT", 30),
                                t_spinup=400, TS_target=point.get("TS_target", 0.08), verbose=False,
                                ml_module=ML_MODULE)
        wt, *_ = sb.build_system(ctx, ml_module=ML_MODULE, adm1_mode="continuous",
                                 adm1_HRT_days=point.get("HRT", 30), adm1_ss_state=ss,
                                 ISR_batch=point.get("ISR_batch", 2.0),
                                 TS_target=point.get("TS_target", 0.08))
    rows = []
    for lab, m in methods_by_label.items():
        lca = LCA({wt: 1}, m)
        lca.lci()
        lca.lcia()
        eng = sum(imp[json.dumps(list(m))].values())
        rows.append(dict(method=lab, brightway=lca.score, engine=eng,
                         rel_diff=abs(eng - lca.score) / max(1e-12, abs(lca.score))))
    df = pd.DataFrame(rows)
    if missing:
        raise AssertionError(f"missing unit scores: {sorted(missing)}")
    bad = df[df["rel_diff"] > rel_tol]
    if len(bad):
        raise AssertionError(f"engine differs from Brightway:\n{bad}")
    return df


# =============================================================================
# DESIGNS
# =============================================================================
def make_points(samples, feeds, T_levels, RT_levels, pathways, space, adm1_core, prefix, extra=None,
                HRT_levels=None):
    """Points for every unit sample x feedstock x T x RT x pathway (x HRT);
    both pathways share each sample (paired comparison). feeds: {label: dict(
    feedstock_type=..., feed={WC, C_feed, ...})}. HRT_levels: None (HRT from
    the sample or the default 30 d) or {pathway: [d, ...]}, e.g.
    {"A": [30], "B": [20, 30, 40]}: Pathway A digests soluble process water,
    so HRT is gridded for B only (F13)."""
    pts = []
    for sid, u in enumerate(samples):
        for fl, f in feeds.items():
            prm, x = map_sample(u, space, f["feedstock_type"], adm1_core)
            design = prm.pop("point", {})
            feed = dict(f["feed"])
            if "feed_WC" in x:
                if "WC_range" not in f:
                    raise ValueError(f"feedstock {fl}: feed_WC is sampled but no WC_range is given")
                lo, up = f["WC_range"]
                feed["WC"] = lo + x["feed_WC"] * (up - lo)
                x = {**x, "feed_WC": feed["WC"]}
            for T in T_levels:
                for RT in RT_levels:
                    for P in pathways:
                        hrts = [None] if HRT_levels is None else list(HRT_levels[P])
                        for H in hrts:
                            if H is not None and "HRT" in design:
                                raise ValueError("HRT is both sampled and gridded")
                            hd = {} if H is None else {"HRT": float(H)}
                            tag = "" if H is None else f"_HRT{H:g}"
                            pts.append(dict(point_id=f"{prefix}{sid:05d}_{fl}_T{T}_RT{RT}_{P}{tag}",
                                            sample_id=sid, pathway=P, feed=feed,
                                            feedstock_type=f["feedstock_type"], feedstock_label=fl,
                                            T=T, RT=RT, params=prm, x=x,
                                            **{**(extra or {}), **design, **hd}))
    return pts


def lhs_design(n, k, seed):
    """n x k Latin hypercube on [0, 1) (scipy.stats.qmc)."""
    from scipy.stats import qmc
    return qmc.LatinHypercube(d=k, seed=seed).random(n)


def morris_design(k, r, levels=4, seed=0):
    """r one-at-a-time trajectories of k+1 points on a `levels`-level grid in
    [0, 1] (Morris 1991). Step delta = levels / (2 (levels - 1)). Returns
    (X with r*(k+1) rows, list of (trajectory, step, factor moved, delta))."""
    rng = np.random.default_rng(seed)
    delta = levels / (2.0 * (levels - 1))
    grid = np.arange(levels) / (levels - 1)
    X, steps = [], []
    for t in range(r):
        x = rng.choice(grid, size=k)
        X.append(x.copy())
        steps.append((t, 0, None, 0.0))
        for s, i in enumerate(rng.permutation(k), start=1):
            d = delta if x[i] + delta <= 1.0 + 1e-12 else -delta
            x[i] += d
            X.append(x.copy())
            steps.append((t, s, int(i), d))
    return np.array(X), steps


def morris_indices(y, steps, names):
    """mu* (mean |EE|), mu and sigma of the elementary effects per factor.
    y: outputs aligned with the design rows (NaN rows skipped)."""
    import pandas as pd
    y = np.asarray(y, float)
    ee = {n: [] for n in names}
    for row, (t, s, i, d) in enumerate(steps):
        if i is None:
            continue
        a, b = y[row - 1], y[row]
        if np.isfinite(a) and np.isfinite(b):
            ee[names[i]].append((b - a) / d)
    out = []
    for n in names:
        e = np.array(ee[n])
        out.append(dict(factor=n, mu_star=np.abs(e).mean() if len(e) else np.nan,
                        mu=e.mean() if len(e) else np.nan,
                        sigma=e.std(ddof=1) if len(e) > 1 else np.nan, n=len(e)))
    return pd.DataFrame(out).sort_values("mu_star", ascending=False, ignore_index=True)


# =============================================================================
# PARAMETER SPACE: unit sample -> physical parameters of a point
# =============================================================================
# feedstock moisture: natural variability within the class range given per
# feedstock (feeds[...]["WC_range"], % wet basis); kind "feedwc"
FEED_MOISTURE_FACTOR = dict(name="feed_WC", kind="feedwc", range=(0.0, 1.0))

# fixed design of the main sweep (user decision, decision log F13); pass as
# make_points(..., extra=MAIN_SWEEP_FIXED_DESIGN)
MAIN_SWEEP_FIXED_DESIGN = dict(target_moisture_HTC=0.85,   # HTC slurry moisture
                               COD_target=25.0,           # kg COD/m3, Pathway A digester feed
                               TS_target=0.10)            # Pathway B digester feed TS
MAIN_SWEEP_HRT_LEVELS = {"A": [30], "B": [20, 30, 40]}   # d (F13)

# design variables screened with Morris (plant designer's choices; the main
# sweep puts the ones that matter on a grid). kind "point": sets the point key.
DESIGN_FACTORS = [
    dict(name="HRT", kind="point", range=(20.0, 40.0)),                  # d
    dict(name="target_moisture_HTC", kind="point", range=(0.75, 0.90)),  # surrogate training 71-95 %
    dict(name="COD_target", kind="point", range=(5.0, 40.0)),            # kg COD/m3, Pathway A
    dict(name="TS_target", kind="point", range=(0.05, 0.12)),            # Pathway B
]


def parameter_space(sb, hce, adm1_core, include_design=False, include_feed_moisture=False,
                    exclude=()):
    """Sampled parameters with their range. kind: ctx (SystemContext field),
    hi (heat_integration constant), bfeed (Pathway B degradability, range of
    the feedstock class), u (position 0-1 used as is), point (design variable
    of the point, only with include_design)."""
    hi = sb.hi
    sp = [dict(name="pw_ch4_multiplier", kind="ctx", range=sb.PW_CH4_MULTIPLIER_RANGE),
          dict(name="fD", kind="bfeed"), dict(name="k_hyd", kind="bfeed"), dict(name="f_sol", kind="bfeed"),
          dict(name="u_fc", kind="u", range=(0.0, 1.0)),
          dict(name="eps_HR", kind="ctx", range=hi.EPS_HR_RANGE),
          dict(name="MC_char", kind="ctx", range=hi.MC_CHAR_RANGE),
          dict(name="nox_conv", kind="ctx", range=hce.NOX_CONV_RANGE)]
    for k, r in sb.SWEEP_HI_PARAMS.items():
        sp.append(dict(name=k, kind="hi", range=getattr(hi, r)))
    if include_design:
        sp += [dict(d) for d in DESIGN_FACTORS]
    if include_feed_moisture:
        sp.append(dict(FEED_MOISTURE_FACTOR))
    unknown = set(exclude) - {s["name"] for s in sp}
    if unknown:
        raise ValueError(f"[parameter_space] cannot exclude unknown factor(s) {sorted(unknown)}")
    return [s for s in sp if s["name"] not in set(exclude)]


def map_sample(u, space, feedstock_type, adm1_core):
    """Physical parameters of one point from a unit sample u (len(space))."""
    prm, bfeed, hio, x, pnt = {}, {}, {}, {}, {}
    cls = adm1_core.PATHWAY_B_FEED_PARAMS.get(feedstock_type, {})
    for ui, s in zip(u, space):
        if s["kind"] == "bfeed":
            rng = cls.get(f"{s['name']}_range")
            v = float(cls[s["name"]]) if rng is None else rng[0] + float(ui) * (rng[1] - rng[0])
            bfeed[s["name"]] = v
        elif s["kind"] == "feedwc":
            v = float(ui)                     # position in the feedstock's WC_range (make_points)
        else:
            lo, up = s["range"]
            v = lo + float(ui) * (up - lo)
            if s["kind"] == "hi":
                hio[s["name"]] = v
            elif s["kind"] == "point":
                pnt[s["name"]] = v
            elif s["kind"] == "u":
                prm[s["name"]] = v
            else:
                prm[s["name"]] = v
        x[s["name"]] = v
    prm["b_feed_overrides"] = bfeed
    prm["hi_overrides"] = hio
    prm["point"] = pnt
    return prm, x
