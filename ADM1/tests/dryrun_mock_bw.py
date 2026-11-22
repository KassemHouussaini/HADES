"""
Dry run of build_system() without Brightway: bw2data is replaced by an
in-memory mock, so the process model (HTC surrogate, ADM1, heat integration,
mass balances) and the exchange amounts written per FU can be checked
without an ecoinvent project. Used by test_regression_s1_dryrun.py. LCIA
scores need the real project.

usage: python dryrun_mock_bw.py <ADM1 folder> <PATHWAY A|B> <out.json> [MC_char] [ML wrapper file]
"""
import sys, os, json, types, importlib.util

# ---------------------------------------------------------------- mock bw2data
_DBS = {}
class Exc(dict):
    def __init__(self, act, input, amount, type, unit=None):
        super().__init__(amount=float(amount), type=type, unit=unit)
        self.act, self.input = act, input
    def save(self):
        if not any(e is self for e in self.act._exc):   # identity, not dict equality
            self.act._exc.append(self)
    def delete(self):
        self.act._exc = [e for e in self.act._exc if e is not self]

class Act(dict):
    def __init__(self, name, unit="unit", location="CA-QC", db="mock", code=None, categories=None):
        super().__init__(name=name, unit=unit, location=location, code=code or name)
        if categories is not None:
            self["categories"] = tuple(categories)
        self.key = (db, code or name)
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

bd = types.ModuleType("bw2data")
bd.Database = FakeDB
bd.databases = _DBS
sys.modules["bw2data"] = bd

def flow(name, cats=("air",)):
    return Act(name, unit="kilogram", db="biosphere3", categories=cats)

# ---------------------------------------------------------------- load builder
folder, PATHWAY, out = sys.argv[1], sys.argv[2], sys.argv[3]
MC_char = float(sys.argv[4]) if len(sys.argv) > 4 else 0.85
spec = importlib.util.spec_from_file_location("system_builder", os.path.join(folder, "system_builder.py"))
sb = importlib.util.module_from_spec(spec); sys.modules["system_builder"] = sb
spec.loader.exec_module(sb)
PORTED = hasattr(sb, "hce")
if PORTED:   # exact flow lookups return on-demand mock flows
    _cache = {}
    def _gf(name, categories=("air",), locations=None):
        key = (name, tuple(categories), tuple(locations or ()))
        if key not in _cache:
            _cache[key] = flow(name + (f", {locations[0]}" if locations else ""), categories)
        return _cache[key]
    sb.get_flow = _gf
    sb.relocate_flow = lambda f, locs: _gf(f["name"], f.get("categories", ("air",)), locs)
    sb.bwl.get_flow = sb.get_flow

# biosphere list for name-based flow lookups
bios = [flow(n) for n in ["Carbon dioxide, non-fossil", "Carbon monoxide, non-fossil",
                          "Methane, non-fossil", "Ammonia", "Sulfur dioxide", "Nitrogen oxides"]]

def tech(name, unit, loc="CA-QC"): return Act(name, unit=unit, location=loc, db="ecoinvent")
elec = tech("market for electricity, medium voltage", "kilowatt hour")
coal_heat = tech("heat production, at hard coal industrial furnace 1-10MW", "megajoule")
# coal furnace template, kg per MJ (illustrative magnitudes, the dry run only
# checks that every flow is transferred and scaled)
for n, a in [("Carbon dioxide, fossil", 0.105), ("Carbon monoxide, fossil", 1e-4),
             ("Methane, fossil", 1e-6), ("Sulfur dioxide", 4e-4), ("Nitrogen oxides", 2e-4),
             ("Particulate Matter, < 2.5 um", 2.5e-5), ("Dinitrogen monoxide", 5e-7),
             ("Mercury", 1e-8)]:
    coal_heat.new_exchange(flow(n), a, "biosphere", "kilogram").save()
coal_heat.new_exchange(tech("market for hard coal", "kilogram"), 0.04325, "technosphere", "kilogram").save()
psa = tech("biogas purification to biomethane by pressure swing adsorption", "cubic meter", "RoW")
psa.new_exchange(tech("market for electricity, low voltage", "kilowatt hour"), 0.3, "technosphere", "kilowatt hour").save()
psa.new_exchange(flow("Methane, non-fossil"), 0.01, "biosphere", "kilogram").save()
psa.new_exchange(flow("Carbon dioxide, non-fossil"), 1.0, "biosphere", "kilogram").save()
bm_heat = tech("heat production, biomethane, at boiler condensing modulating", "megajoule", "RoW")
bm_heat.new_exchange(flow("Nitrogen oxides"), 2e-5, "biosphere", "kilogram").save()

FG = f"Pathway{PATHWAY}"
def fg(name, unit="unit"):
    a = Act(name, unit=unit, db=FG); a.new_exchange(a, 1, "production", unit).save(); return a

ei = FakeDB("ecoinvent")
WC = 85.0; C, H, N, S, Ash = 38.25, 5.47, 3.66, 0.5, 9.1
dp = {"WC": WC, "C_feed": C, "H_feed": H, "N_feed": N, "O_feed": 100 - C - H - N - S - Ash,
      "S_feed": S, "Ash_feed": Ash, "VM_feed": 82.6, "FC_feed": 8.3}
kw = dict(ei=ei, bios=bios, FOREGROUND_DB=FG, elec=elec, ad_elec=elec,
          ad_heat=tech("heat production, natural gas", "megajoule"), drying_heat=elec,
          coal_heat=coal_heat, ng_heat_market=tech("market for heat, NG", "megajoule"),
          biomethane_heat_ref=bm_heat, tap_water=tech("market for tap water", "kilogram"),
          wastewater_treat=tech("treatment of wastewater, average", "cubic meter"), psa_ref=psa,
          co2_air=bios[0], co_air=bios[1],
          htc_reactor=fg("htc_reactor"), drying=fg("drying"), heat_prod=fg("heat_prod", "megajoule"),
          sub_heat_coal=fg("sub_heat_coal", "megajoule"), ad_process=fg("ad"), dewatering=fg("dewatering"),
          biogas_upgrading=fg("psa"), heat_biomethane=fg("heat_biomethane", "megajoule"),
          heat_naturalgas=fg("heat_ng", "megajoule"), waste_treatment=fg("waste_treatment", "kilogram"),
          PATHWAY=PATHWAY, m_feedstock=1000.0, default_params=dp, ml_params={**dp, "T": 180, "RT": 60},
          T_HTC=180, RT_HTC=60, target_moisture_HTC=0.85, MC_char=MC_char, target_MC_char_dry=0.10,
          COD_conc_target=10.0, ISR_batch=2.0)
if PORTED:
    mix = fg("qc_heat_mix", "megajoule")
    comps = {g: [(tech(f"{g} heat dataset", "megajoule"), 1.0)] for g in sb.hce.QC_HEAT_MIX}
    kw.update(feedstock_name="food_waste", qc_heat_mix=mix, heat_mix_components_qc=comps)
# optional heat supply, e.g. HADES_HEAT_SUPPLY="htc=own_biogas,dryer=ng,ad=own_biogas"
# (values: electricity, own_biogas, ng = the natural gas heat dataset above)
_hs = os.environ.get("HADES_HEAT_SUPPLY")
if _hs:
    _map = {"electricity": "electricity", "own_biogas": "own_biogas", "ng": kw["ad_heat"]}
    kw["heat_supply"] = {k: _map[v] for k, v in (x.split("=") for x in _hs.split(","))}
# optional Paper 2 options: HADES_CONTEXT="ON" (lca_contexts preset: displaced heat mix),
# HADES_B_FEED='{"fD": 0.8}' (Pathway B degradability overrides), HADES_PW_M="1.2"
_cx = os.environ.get("HADES_CONTEXT")
if _cx and PORTED:
    _spec = importlib.util.spec_from_file_location("lca_contexts", os.path.join(folder, "lca_contexts.py"))
    lcx = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lcx)
    _p = lcx.get_preset(_cx)
    _ds = lcx.heat_mix_datasets(_p)
    kw["heat_mix_components_qc"] = {g: [(tech(f"{n} heat dataset", "megajoule"), w / sum(x[2] for x in _ds[g]))
                                        for n, _, w in _ds[g]] for g in lcx.heat_mix_shares(_p)}
    kw.update({k: v for k, v in lcx.ctx_kwargs(_p).items() if k == "heat_mix_shares"})
    _ng = lcx.ng_heat_datasets(_p)
    if _ng:
        kw["ng_heat_components"] = [(tech(f"{n} heat dataset", "megajoule"), w) for n, _, w in _ng]
if os.environ.get("HADES_B_FEED"):
    kw["b_feed_overrides"] = json.loads(os.environ["HADES_B_FEED"])
if os.environ.get("HADES_PW_M"):
    kw["pw_ch4_multiplier"] = float(os.environ["HADES_PW_M"])
ctx = sb.SystemContext(**kw)

ml_file = sys.argv[5] if len(sys.argv) > 5 else "htc_surrogate.py"
ml = os.path.join(folder, "..", "ML training scripts and models", ml_file)
ss = sb.spinup_ss_state(ctx, pathway=PATHWAY, adm1_HRT_days=30, t_spinup=400, verbose=False, ml_module=ml)
wt, ss_state, adm1_out, feed_inputs, inv = sb.build_system(
    ctx, ml_module=ml, adm1_mode="continuous", adm1_HRT_days=30, adm1_t_sim_days=None,
    adm1_ss_state=ss, ISR_batch=2.0)

def dump(a):
    return [{"input": e.input["name"], "type": e["type"], "unit": e["unit"], "amount": e["amount"]}
            for e in a.exchanges() if e["type"] != "production"]
res = {"meta": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                for k, v in dict(wt).items()},
       "inv": {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in inv.items()},
       "exchanges": {a["name"]: dump(a) for a in _DBS[FG]}}
json.dump(res, open(out, "w"), indent=1, default=str)
print("DONE", out)
