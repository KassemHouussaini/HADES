"""
Sweep engine run without Brightway (mock background, as dryrun_mock_bw.py):
one point per pathway, two LCA contexts x two heat configurations, in memory.
Writes a JSON summary used by test_sweep_engine.py.

usage: python engine_mock_run.py <ADM1 folder> <out.json>
"""
import importlib.util, json, os, sys, types

folder, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(folder, "tests"))

# ---------------------------------------------------------------- mock bw2data / bw2calc
_DBS = {}


class Exc(dict):
    def __init__(self, act, input, amount, type, unit=None):
        super().__init__(amount=float(amount), type=type, unit=unit)
        self.act, self.input = act, input

    def save(self):
        if not any(e is self for e in self.act._exc):
            self.act._exc.append(self)

    def delete(self):
        self.act._exc = [e for e in self.act._exc if e is not self]


class Act(dict):
    def __init__(self, name, unit="unit", location="CA-QC", db="mock", categories=None):
        super().__init__(name=name, unit=unit, location=location, code=name)
        if categories is not None:
            self["categories"] = tuple(categories)
        self.key = (db, f"{name}|{location}")
        self._exc = []
        _DBS.setdefault(db, []).append(self)

    def exchanges(self): return list(self._exc)
    def biosphere(self): return [e for e in self._exc if e["type"] == "biosphere"]
    def technosphere(self): return [e for e in self._exc if e["type"] == "technosphere"]
    def new_exchange(self, input, amount, type, unit=None): return Exc(self, input, amount, type, unit)
    def save(self): pass


class FakeDB(list):
    def __init__(self, name):
        super().__init__(_DBS.get(name, [])); self.name = name
    def search(self, q, limit=None): return []


bd = types.ModuleType("bw2data"); bd.Database = FakeDB; bd.databases = _DBS
sys.modules["bw2data"] = bd
bc = types.ModuleType("bw2calc"); bc.LCA = object
sys.modules["bw2calc"] = bc

# ---------------------------------------------------------------- load HADES
def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(folder, fname))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m

sb = load("system_builder", "system_builder.py")
lcx = load("lca_contexts", "lca_contexts.py")
lcc = load("lca_contributions", "lca_contributions.py")
se = load("sweep_engine", "sweep_engine.py")

_flows = {}
def flow(name, cats=("air",)):
    k = (name, tuple(cats))
    if k not in _flows:
        _flows[k] = Act(name, unit="kilogram", db="biosphere3", categories=cats)
    return _flows[k]

sb.get_flow = lambda name, categories=("air",), locations=None: flow(name, categories)
sb.relocate_flow = lambda f, locs: flow(f["name"], f.get("categories", ("air",)))
sb.bwl.get_flow = sb.get_flow

def tech(name, unit="megajoule", loc="CA-QC"): return Act(name, unit=unit, location=loc, db="ecoinvent")

def background(loc):
    coal = tech("heat production, at hard coal industrial furnace 1-10MW", loc=loc)
    for n, a in [("Carbon dioxide, fossil", 0.105), ("Sulfur dioxide", 4e-4), ("Nitrogen oxides", 2e-4),
                 ("Particulate Matter, < 2.5 um", 2.5e-5)]:
        coal.new_exchange(flow(n), a, "biosphere", "kilogram").save()
    coal.new_exchange(tech("market for hard coal", "kilogram", loc), 0.04325, "technosphere", "kilogram").save()
    psa = tech("biogas purification to biomethane by pressure swing adsorption", "cubic meter", "RoW")
    psa.new_exchange(tech("market for electricity, low voltage", "kilowatt hour", "RoW"), 0.3,
                     "technosphere", "kilowatt hour").save()
    psa.new_exchange(flow("Methane, non-fossil"), 0.01, "biosphere", "kilogram").save()
    bm = tech("heat production, biomethane, at boiler condensing modulating", loc="RoW")
    bm.new_exchange(flow("Nitrogen oxides"), 2e-5, "biosphere", "kilogram").save()
    elec = tech("market for electricity, medium voltage", "kilowatt hour", loc)
    bg = dict(elec=elec, ad_elec=elec, drying_heat=elec, coal_heat=coal, psa_ref=psa, biomethane_heat_ref=bm,
              ad_heat=tech(se.NG_BOILER, loc=loc),
              ng_heat_market=tech("market for heat, district or industrial, natural gas", loc=loc),
              tap_water=tech("market for tap water", "kilogram", loc),
              wastewater_treat=tech("treatment of wastewater, average", "cubic meter", loc))
    return bg

def rc_for(key, loc):
    p = lcx.get_preset(key)
    ds = lcx.heat_mix_datasets(p)
    mix = {g: [(tech(n, loc=loc), w / sum(x[2] for x in ds[g])) for n, _, w in ds[g]]
           for g in lcx.heat_mix_shares(p)}
    ng = lcx.ng_heat_datasets(p)
    return dict(preset=p, bg=background(loc), heat_mix=mix,
                ng_comp=[(tech(n, loc=loc), w) for n, _, w in ng] if ng else None, ash=None,
                co2_air=flow("Carbon dioxide, non-fossil"), co_air=flow("Carbon monoxide, non-fossil"))

ei, bios = FakeDB("ecoinvent"), [flow("Carbon dioxide, non-fossil"), flow("Carbon monoxide, non-fossil")]
memo = se.Adm1Memo(sb).install()
se.install_ml_cache(sb)
rcs = {"QC_PAPER1": rc_for("QC_PAPER1", "CA-QC"), "ON": rc_for("ON", "RoW")}

feed = dict(WC=85.0, C_feed=38.25, H_feed=5.47, N_feed=3.66, S_feed=0.5, Ash_feed=9.1, VM_feed=82.6, FC_feed=8.3)
res = {}
for P in ("A", "B"):
    memo.clear(); memo.misses = memo.hits = 0
    pt = dict(point_id=f"test_{P}", pathway=P, feed=feed, feedstock_type="food_waste", T=180, RT=60,
              HRT=30, MC_char=0.35, params={})
    runs = []
    for c, rc in rcs.items():
        for h in ("paper1", "own_biogas"):
            wt, inv = se.build_point(sb, ei, bios, rc, pt, h, ml_module=se.ML_MODULE)
            acts, flows = se.background_keys(wt)
            scores = {"unit_scores": {k: {"m": 1.0} for k in acts}, "cfs": {"m": {k: 1.0 for k in flows}}}
            imp, missing = se.score(wt, scores, lcc.category, lcc.CATS)
            row = se.result_row(pt, c, h, wt, inv, imp, missing, {"m": "GWP"})
            runs.append(dict(context=c, heat=h, MJ_hydrochar=wt["MJ_hydrochar"], CH4=wt["CH4_mass_AD"],
                             biomethane=wt["biomethane_m3"], caps=wt["caps_active"],
                             total=row["GWP"], cat_sum=sum(v for k, v in row.items() if k.startswith("GWP | ")),
                             missing=sorted(missing), n_acts=len(acts)))
    res[P] = dict(runs=runs, adm1_misses=memo.misses, adm1_hits=memo.hits)
# worker path: run_point writes one CSV per point and skips it when rerun
import tempfile
pt = dict(point_id="wp_B", pathway="B", feed=feed, feedstock_type="food_waste", T=180, RT=60, HRT=30,
          MC_char=0.35, params={"hi_overrides": {"F_LOSS_AD": 0.3}, "b_feed_overrides": {"fD": 0.8}},
          x={"F_LOSS_AD": 0.3, "fD": 0.8})
keys = {}
for rc in rcs.values():
    for h in ("paper1", "own_biogas"):
        wt, _ = se.build_point(sb, ei, bios, rc, pt, h, ml_module=se.ML_MODULE)
        keys.update(se.background_keys(wt)[0])
d = tempfile.mkdtemp()
se._W.update(sb=sb, lcc=lcc, ei=ei, bios=bios, rcs=rcs, out_dir=d, memo=memo,
             scores={"unit_scores": {k: {"m": 1.0} for k in keys}, "cfs": {"m": {}}}, method_labels={"m": "GWP"})
p1 = se.run_point(pt, ["paper1", "own_biogas"])
p2 = se.run_point(pt, ["paper1", "own_biogas"])
import pandas as pd
w = pd.read_csv(p1[0])
res["worker"] = dict(status=[p1[1], p2[1]], rows=len(w), cols=list(w.columns),
                     fD=w["adm1_fD"].tolist(), x=w["x_F_LOSS_AD"].tolist(), err=("error" in w))
json.dump(res, open(out, "w"), indent=1, default=str)
