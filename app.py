"""
HADES Streamlit interface.

Run with:  streamlit run app.py

The user decisions follow the article (Supporting Information, Table S1):
process configuration, feedstock properties, HTC operating conditions, AD
operating conditions and LCA settings (location, background database, LCIA
method). Model defaults that are not user decisions in the article are grouped
under "Advanced". All process logic is imported from ADM1/system_builder.py,
the module used by the case study notebook; background datasets are selected
with the same deterministic location-priority lookups (ADM1/bw_lookup.py).
"""

import os
import sys
import importlib.util

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model modules, loaded by path
# ---------------------------------------------------------------------------
def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sb = _load_module("system_builder", os.path.join(_APP_DIR, "ADM1", "system_builder.py"))
bwl, hce, hi = sb.bwl, sb.hce, sb.hi
ML_MODULE_PATH = os.path.join(_APP_DIR, "ML training scripts and models", "htc_surrogate.py")

# Defaults of the article's case study (background database and LCIA indicators)
DEFAULT_EI = "ecoinvent-3.12-cutoff"
IW_MID = "IMPACT World+ Midpoint 2.2.1_regionalized for ecoinvent v3.12"
IW_DMG = "IMPACT World+ Damage 2.2.1_regionalized for ecoinvent v3.12"
DEFAULT_METHODS = [(IW_MID, "Midpoint", "Climate change, short term"),
                   (IW_DMG, "Human health", "Total human health"),
                   (IW_DMG, "Ecosystem quality", "Total ecosystem quality")]
FEED_PRESETS = {   # case study feedstocks, % dry basis (O by difference), moisture % wet basis
    "Food waste (Base scenario)": dict(C=38.25, H=5.47, N=3.66, S=0.50, Ash=9.10, VM=82.6, FC=8.3, WC=85.0,
                                       dig="food_waste"),
    "OFMSW": dict(C=45.00, H=6.00, N=2.50, S=0.20, Ash=5.30, VM=78.0, FC=16.7, WC=70.0, dig="food_waste"),
}
DIGESTATE_TYPES = {"Food waste / OFMSW": "food_waste", "Dairy manure": "dairy_manure",
                   "Sewage sludge": "sewage_sludge"}
HTC_T_RANGE = (160, 280)    # validity of the S_I sub-pool composition (PW-ADM1)


def method_label(m):
    return " | ".join(m)


def ecoinvent_projects(bd):
    """{project: [ecoinvent databases]} for the Brightway projects of this
    computer that hold at least one ecoinvent database."""
    out, current = {}, bd.projects.current
    for p in sorted(pr.name for pr in bd.projects):
        try:
            bd.projects.set_current(p)
            eis = sorted(d for d in bd.databases if "ecoinvent" in d.lower() and "biosphere" not in d.lower())
        except Exception:
            eis = []
        if eis:
            out[p] = eis
    try:
        bd.projects.set_current(current)
    except Exception:
        pass
    return out


def biosphere_of(bd, ei_name):
    """Biosphere database the background database depends on (biosphere3 if none is declared)."""
    deps = [d for d in bd.databases[ei_name].get("depends", []) if "biosphere" in d.lower()
            and "spatialized" not in d.lower()]
    return deps[0] if deps else "biosphere3"


SETUP_HELP = ("No Brightway project with an ecoinvent database was found on this computer. "
              "Run `python setup_hades.py` once to create a project, import ecoinvent (your own licence) "
              "and IMPACT World+; see the README, section *Setup*.")


def release_bw_locks():
    """Force-close open Brightway/peewee database connections."""
    try:
        import bw2data as bd
        for db_name in bd.databases:
            try:
                db_obj = bd.Database(db_name)
                if hasattr(db_obj, "_db") and hasattr(db_obj._db, "close"):
                    db_obj._db.close()
            except Exception:
                pass
        try:
            from bw2data.backends.peewee import sqlite3_lci_db
            sqlite3_lci_db.close()
        except Exception:
            pass
    except Exception:
        pass


# ============================================================
# Location registry: country -> region -> ecoinvent location code and fallbacks
# ============================================================

LOCATION_HIERARCHY = {
    "Canada": {
        "Alberta":                    {"code": "CA-AB", "fallback": ["CA", "RoW", "GLO"]},
        "British Columbia":           {"code": "CA-BC", "fallback": ["CA", "RoW", "GLO"]},
        "Manitoba":                   {"code": "CA-MB", "fallback": ["CA", "RoW", "GLO"]},
        "New Brunswick":              {"code": "CA-NB", "fallback": ["CA", "RoW", "GLO"]},
        "Newfoundland and Labrador":  {"code": "CA-NL", "fallback": ["CA", "RoW", "GLO"]},
        "Northwest Territories":      {"code": "CA-NT", "fallback": ["CA", "RoW", "GLO"]},
        "Nova Scotia":                {"code": "CA-NS", "fallback": ["CA", "RoW", "GLO"]},
        "Nunavut":                    {"code": "CA-NU", "fallback": ["CA", "RoW", "GLO"]},
        "Ontario":                    {"code": "CA-ON", "fallback": ["CA", "RoW", "GLO"]},
        "Prince Edward Island":       {"code": "CA-PE", "fallback": ["CA", "RoW", "GLO"]},
        "Quebec":                     {"code": "CA-QC", "fallback": ["CA", "RoW", "GLO"]},
        "Saskatchewan":               {"code": "CA-SK", "fallback": ["CA", "RoW", "GLO"]},
        "Yukon":                      {"code": "CA-YT", "fallback": ["CA", "RoW", "GLO"]},
    },
    "USA": {
        "California":     {"code": "US-WECC", "fallback": ["US", "RNA", "RoW", "GLO"]},
        "Texas":          {"code": "US-TRE",  "fallback": ["US", "RNA", "RoW", "GLO"]},
        "New York":       {"code": "US-NPCC", "fallback": ["US", "RNA", "RoW", "GLO"]},
    },
    "Germany":        {"—": {"code": "DE", "fallback": ["RER", "RoW", "GLO"]}},
    "France":         {"—": {"code": "FR", "fallback": ["RER", "RoW", "GLO"]}},
    "United Kingdom": {"—": {"code": "GB", "fallback": ["RER", "RoW", "GLO"]}},
    "Rest of World":  {"—": {"code": "RoW", "fallback": ["GLO"]}},
    "Global":         {"—": {"code": "GLO", "fallback": ["RoW"]}},
}


def location_priority(country, province):
    info = LOCATION_HIERARCHY.get(country, {}).get(province) or {"code": "GLO", "fallback": ["RoW"]}
    return info["code"], [info["code"]] + list(info["fallback"])


def resolve_background(ei, prio):
    """Background datasets of the case study notebook, chosen by location priority."""
    find = lambda q, p=prio, nf=None: bwl.find_process(ei, q, p, name_filter=nf)
    bg = dict(
        elec=find("market for electricity, medium voltage", nf="market for electricity, medium voltage"),
        coal_heat=find("heat production, at hard coal industrial furnace 1-10MW", nf="hard coal industrial furnace"),
        ad_heat=find("heat production, natural gas", nf="heat production, natural gas"),
        ng_heat_market=find("market for heat, district or industrial, natural gas",
                            nf="market for heat, district or industrial, natural gas"),
        biomethane_heat_ref=find("heat production, biomethane, at boiler condensing modulating",
                                 ["RoW", "GLO"] + prio, nf="biomethane"),
        tap_water=find("market for tap water", nf="market for tap water"),
        wastewater_treat=find("treatment of wastewater, average", nf="treatment of wastewater, average"),
        psa_ref=find("biogas purification to biomethane by pressure swing adsorption", ["RoW", "GLO"] + prio),
    )
    bg["ad_elec"] = bg["drying_heat"] = bg["elec"]        # AD and dryer run on grid electricity
    heat_mix = bwl.resolve_heat_mix(ei, hce.HEAT_MIX_DATASETS, hce.QC_HEAT_MIX.keys(), prio)
    try:
        ash = bwl.find_exact(ei, "market for wood ash mixture, pure", prio)
    except LookupError:
        ash = None
    return bg, heat_mix, ash


def foreground(bd, pathway):
    name = f"Pathway{pathway}"
    if name not in bd.databases:
        bd.Database(name).register()
    db = bd.Database(name)
    g = lambda code, label, unit: sb.get_or_create(db, code, label, unit)
    fg = dict(
        htc_reactor=g("htc_reactor", "HTC reactor", "unit"),
        drying=g("hydrochar_drying", "Hydrochar drying", "unit"),
        heat_prod=g("heat_prod_hydrochar", "Heat production from hydrochar", "megajoule"),
        sub_heat_coal=g("substitute_heat_coal", "Hard coal heat", "megajoule"),
        ad_process=g("anaerobic_digestion", "Anaerobic digestion (ADM1)", "unit"),
        dewatering=g("digestate dewatering", "Digestate dewatering", "unit"),
        biogas_upgrading=g("biogas_upgrading", "Biogas upgrading (PSA)", "unit"),
        heat_biomethane=g("heat_biomethane", "Heat production from biomethane", "megajoule"),
        heat_naturalgas=g("heat_naturalgas", "Natural gas heat (substituted)", "megajoule"),
        qc_heat_mix=g("qc_industrial_heat_mix", "Industrial combustion heat mix (substituted)", "megajoule"),
        waste_treatment=g("waste_treatment_1kg", "Waste treatment of organic waste (HTC-AD system)", "kilogram"),
    )
    wt0 = fg["waste_treatment"]
    keep = {"name", "code", "unit", "database", "type", "location", "reference product", "id"}
    for k in [k for k in list(wt0._data) if k not in keep]:
        del wt0._data[k]
    wt0.save()
    return name, fg


def apply_preset():
    p = FEED_PRESETS.get(st.session_state["preset"])
    if p:
        for k in ("C", "H", "N", "S", "Ash", "VM", "FC", "WC"):
            st.session_state[f"feed_{k}"] = p[k]
        st.session_state["dig_type"] = [k for k, v in DIGESTATE_TYPES.items() if v == p["dig"]][0]


# ============================================================
# PAGE
# ============================================================
st.set_page_config(page_title="HADES", page_icon="⚗️", layout="wide")
st.title("HADES: HTC and anaerobic digestion life cycle assessment")
st.caption("Functional unit: treatment of 1 t of wet feedstock. Negative scores are net environmental benefits.")

if "preset" not in st.session_state:
    st.session_state["preset"] = "Food waste (Base scenario)"
    apply_preset()

# ============================================================
# SIDEBAR: user decisions (article, Table S1)
# ============================================================
with st.sidebar:
    st.header("1. Process configuration")
    PATHWAY = st.radio("Pathway", ["A", "B"],
                       captions=["HTC, then AD of the process water", "AD, then HTC of the digestate"])

    st.header("2. Feedstock properties")
    st.selectbox("Load feedstock", list(FEED_PRESETS) + ["Custom"], key="preset", on_change=apply_preset)
    WC_feed = st.number_input("Moisture content (% wet basis)", 1.0, 99.0, step=0.5, key="feed_WC")
    c1, c2 = st.columns(2)
    VM_feed = c1.number_input("Volatile matter (% dry)", 0.0, 100.0, step=0.1, key="feed_VM")
    FC_feed = c2.number_input("Fixed carbon (% dry)", 0.0, 100.0, step=0.1, key="feed_FC")
    Ash_feed = c1.number_input("Ash (% dry)", 0.0, 100.0, step=0.1, key="feed_Ash")
    C_feed = c2.number_input("C (% dry)", 0.0, 100.0, step=0.1, key="feed_C")
    H_feed = c1.number_input("H (% dry)", 0.0, 100.0, step=0.01, key="feed_H")
    N_feed = c2.number_input("N (% dry)", 0.0, 100.0, step=0.01, key="feed_N")
    S_feed = c1.number_input("S (% dry)", 0.0, 100.0, step=0.01, key="feed_S")
    O_feed = 100.0 - C_feed - H_feed - N_feed - S_feed - Ash_feed
    c2.metric("O (% dry, by difference)", f"{O_feed:.2f}")

    st.header("3. HTC operating conditions")
    T_HTC = st.slider("Temperature (°C)", HTC_T_RANGE[0], HTC_T_RANGE[1], 180, step=5)
    RT_HTC = st.number_input("Residence time (min)", 5, 900, 60, step=5)
    moist_pct = st.slider("Moisture target in the reactor (%)", 50, 95, 85,
                          help="Sets the solid loading; drier feedstocks are diluted with tap water before HTC.")

    st.header("4. AD operating conditions")
    adm1_mode = st.radio("Mode", ["continuous", "batch"], horizontal=True)
    if adm1_mode == "continuous":
        adm1_HRT_days = st.number_input("Hydraulic retention time (d)", 5, 90, 30)
        ISR_batch, batch_days = 2.0, None
    else:
        adm1_HRT_days = None
        ISR_batch = st.number_input("Inoculum to substrate ratio (VS basis)", 0.5, 10.0, 2.0, step=0.5)
        batch_days = st.number_input("Duration (d)", 5, 120, 30)
    if PATHWAY == "A":
        COD_conc_target = st.number_input("COD target of the diluted process water (kg COD/m³)", 1.0, 50.0, 10.0, step=1.0)
        TS_pct = 8.0
        dig_type = st.session_state.get("dig_type", "Food waste / OFMSW")
    else:
        TS_pct = st.number_input("Total solids target of the digester feed (%)", 2.0, 15.0, 8.0, step=0.5)
        COD_conc_target = 10.0
        dig_type = st.selectbox("Digestate coefficients", list(DIGESTATE_TYPES), key="dig_type",
                                help="Measured digestate H/C, O/C and fixed carbon (SI, digestate characterization).")

    st.header("5. LCA settings")
    sel_country = st.selectbox("Country", list(LOCATION_HIERARCHY), index=list(LOCATION_HIERARCHY).index("Canada"))
    sel_province = st.selectbox("Province / state / region", list(LOCATION_HIERARCHY[sel_country]),
                                index=(list(LOCATION_HIERARCHY[sel_country]).index("Quebec")
                                       if sel_country == "Canada" else 0))
    loc_code, loc_prio = location_priority(sel_country, sel_province)
    st.caption(f"Dataset location priority: {' → '.join(loc_prio)}")
    if loc_code != "CA-QC":
        st.info("The industrial heat mix displaced by hydrochar uses Quebec fuel shares; "
                "its datasets follow the location priority above.")

    try:
        import bw2data as bd
        if "ei_projects" not in st.session_state:
            st.session_state["ei_projects"] = ecoinvent_projects(bd)
        ei_projects = st.session_state["ei_projects"]
    except Exception as e:
        bd, ei_projects = None, {}
        st.error(f"Brightway not available: {e}")
    projects = list(ei_projects)
    BW_PROJECT = st.selectbox("Brightway project", projects,
                              help="Projects of this computer that contain an ecoinvent database.") if projects else None
    if st.button("Refresh project list", help="After creating a project with setup_hades.py"):
        st.session_state.pop("ei_projects", None)
        st.rerun()
    dbs, methods_all = ei_projects.get(BW_PROJECT, []), []
    if BW_PROJECT:
        bd.projects.set_current(BW_PROJECT)
        methods_all = sorted(bd.methods)
    EI_NAME = st.selectbox("Background database (ecoinvent version and system model)", dbs,
                           index=dbs.index(DEFAULT_EI) if DEFAULT_EI in dbs else 0) if dbs else None
    default_m = [m for m in DEFAULT_METHODS if m in methods_all]
    if BW_PROJECT and len(default_m) < len(DEFAULT_METHODS):
        st.warning("The IMPACT World+ 2.2.1 indicators of the article are not in this project. "
                   "Import them with `python setup_hades.py --iw <file>` or choose other indicators.")
    METHODS = st.multiselect("LCIA method (indicators)", methods_all, default=default_m, format_func=method_label)
    hc_credit = st.radio("Heat displaced by hydrochar", ["Industrial heat mix", "Hard coal heat"],
                         help="Article: industrial heat mix; hard coal heat is the alternative of SI Section S6.2.")

    with st.expander("Advanced: model defaults"):
        MC_char = st.number_input("Hydrochar moisture after the filter press (kg/kg)", 0.30, 0.50,
                                  float(hi.MC_CHAR_CENTRAL), step=0.01)
        target_MC_char_dry = st.number_input("Hydrochar moisture after drying (kg/kg)", 0.0, 0.30, 0.10, step=0.01)
        eps_HR = st.number_input("HTC feed preheat effectiveness", 0.0, 0.9, float(hi.EPS_HR_CENTRAL), step=0.05)
        nox_conv = st.number_input("Fuel N converted to NOx in the boiler", 0.01, 0.25, float(hce.NOX_CONV_CENTRAL), step=0.01)
        t_spinup = st.number_input("AD spin-up duration (d)", 100, 1000, 400, step=50)

    run_button = st.button("▶ Run", type="primary", use_container_width=True)

# ============================================================
# INPUT CHECKS
# ============================================================
errors, warns = [], []
if O_feed < 0:
    errors.append(f"C + H + N + S + ash = {100 - O_feed:.2f} % > 100 %")
if abs(VM_feed + FC_feed + Ash_feed - 100.0) > 1.0:
    warns.append(f"VM + FC + ash = {VM_feed + FC_feed + Ash_feed:.1f} % (expected 100 %)")
if moist_pct / 100.0 < WC_feed / 100.0 - 1e-9:
    warns.append("The feedstock is wetter than the reactor moisture target: no water is added before HTC.")
if not BW_PROJECT or not EI_NAME:
    errors.append(SETUP_HELP)
if not METHODS:
    errors.append("Select at least one LCIA indicator.")
for w in warns:
    st.warning(w)
if errors:
    for e in errors:
        st.error(e)
    st.stop()

# ============================================================
# RUN
# ============================================================
if run_button:
    release_bw_locks()
    feed = dict(WC=WC_feed, C_feed=C_feed, H_feed=H_feed, N_feed=N_feed, O_feed=O_feed,
                S_feed=S_feed, Ash_feed=Ash_feed, VM_feed=VM_feed, FC_feed=FC_feed)
    progress = st.progress(0, text="Connecting to Brightway…")
    try:
        from bw2calc import LCA
        lcc = _load_module("lca_contributions", os.path.join(_APP_DIR, "ADM1", "lca_contributions.py"))
        lcc.clear_cache()
        bd.projects.set_current(BW_PROJECT)
        bwl.GENERIC_FLOW_DB = biosphere_of(bd, EI_NAME)
        ei, bios = bd.Database(EI_NAME), bd.Database(bwl.GENERIC_FLOW_DB)
        progress.progress(10, text="Resolving background datasets…")
        bg, heat_mix, ash = resolve_background(ei, loc_prio)
        log = pd.DataFrame([{"Role": k, "Dataset": a["name"], "Location": a.get("location", "?"),
                             "Exact": "yes" if a.get("location") == loc_code else "fallback"}
                            for k, a in bg.items() if k not in ("ad_elec", "drying_heat")])
        co2_air = bwl.get_flow("Carbon dioxide, non-fossil", ("air",), bwl.CORE_FLOW_LOCATIONS)
        co_air = bwl.get_flow("Carbon monoxide, non-fossil", ("air",), bwl.CORE_FLOW_LOCATIONS)
        FG, fg = foreground(bd, PATHWAY)
        ctx = sb.SystemContext(
            ei=ei, bios=bios, FOREGROUND_DB=FG, **bg, co2_air=co2_air, co_air=co_air, **fg,
            PATHWAY=PATHWAY, m_feedstock=1000.0, default_params=feed,
            ml_params={**feed, "T": T_HTC, "RT": RT_HTC},
            T_HTC=T_HTC, RT_HTC=RT_HTC, target_moisture_HTC=moist_pct / 100.0,
            MC_char=MC_char, target_MC_char_dry=target_MC_char_dry,
            COD_conc_target=COD_conc_target, ISR_batch=ISR_batch,
            feedstock_name=DIGESTATE_TYPES[dig_type], eps_HR=eps_HR, nox_conv=nox_conv,
            heat_mix_components_qc=heat_mix, ash_disposal=ash,
            location_country=sel_country, location_province=sel_province, location_code=loc_code,
            location_priority=loc_prio)
        ss = None
        if adm1_mode == "continuous":
            progress.progress(30, text="AD spin-up to steady state…")
            ss = sb.spinup_ss_state(ctx, pathway=PATHWAY, adm1_HRT_days=adm1_HRT_days, t_spinup=t_spinup,
                                    TS_target=TS_pct / 100.0, verbose=False, ml_module=ML_MODULE_PATH)
        progress.progress(55, text="Building the system…")
        wt, _, adm1_out, _, inv = sb.build_system(
            ctx, ml_module=ML_MODULE_PATH, adm1_mode=adm1_mode, adm1_HRT_days=adm1_HRT_days,
            adm1_t_sim_days=batch_days, adm1_ss_state=ss, ISR_batch=ISR_batch, TS_target=TS_pct / 100.0)
        progress.progress(80, text="Life cycle impact assessment…")
        MJ_hc = float(wt.get("MJ_hydrochar") or 0.0)
        scores, contrib = {}, {}
        for m in METHODS:
            lca = LCA({wt: 1}, m); lca.lci(); lca.lcia()
            s_mix = lca.score
            c = lcc.contributions(wt, FG, m, total=s_mix)
            if hc_credit == "Hard coal heat":
                # same credit with coal heat instead of the heat mix (SI Section S6.2)
                d = MJ_hc * (lcc.unit_score(fg["qc_heat_mix"], m) - lcc.unit_score(bg["coal_heat"], m))
                c["Credit: industrial heat mix"] += d
                c = {("Credit: hard coal heat" if k == "Credit: industrial heat mix" else k): v for k, v in c.items()}
                s_mix += d
            scores[m], contrib[m] = s_mix, c
        units = {m: bd.methods[m].get("unit", "") for m in METHODS}
        progress.progress(100, text="Done")
        flags = [f for f in str(wt.get("qa_flags") or "").split("; ") if f]
        st.session_state["res"] = dict(wt={k: v for k, v in wt._data.items()}, scores=scores, contrib=contrib, units=units,
                                       flags=flags, log=log, pathway=PATHWAY, credit=hc_credit, loc=loc_code)
    except Exception as e:
        progress.empty()
        st.error(f"Run failed: {e}")
        st.exception(e)
        st.stop()

# ============================================================
# RESULTS
# ============================================================
if "res" in st.session_state:
    R = st.session_state["res"]
    wt = R["wt"]
    st.subheader(f"Results: Pathway {R['pathway']} ({R['loc']}; hydrochar displaces {R['credit'].lower()})")
    if R["flags"]:
        with st.container(border=True):
            st.markdown("**Applicability and plausibility flags**")
            for f in R["flags"]:
                st.warning(f)
    else:
        st.success("All inputs within the training domain of the HTC surrogate; all plausibility checks passed.")

    cols = st.columns(len(R["scores"]))
    for col, (m, s) in zip(cols, R["scores"].items()):
        col.metric(f"{m[-1]} ({R['units'].get(m) or 'unit n/a'} per t)", f"{s:.4g}", help=method_label(m))

    tab_lca, tab_proc, tab_bg = st.tabs(["Impacts by contribution", "Process outputs", "Background datasets"])
    with tab_lca:
        for m, c in R["contrib"].items():
            df = pd.Series(c).loc[lambda x: x.abs() > 0]
            fig, ax = plt.subplots(figsize=(8, 0.45 * len(df) + 1.2))
            ax.barh(df.index, df.values, color=["#c0392b" if v < 0 else "#2f6fd6" for v in df.values])
            ax.axvline(0, color="0.2", lw=0.8)
            ax.set_title(f"{method_label(m)}   (net {R['scores'][m]:.4g} {R['units'].get(m, '')})", fontsize=10)
            ax.invert_yaxis(); ax.grid(axis="x", color="0.92"); ax.set_axisbelow(True)
            st.pyplot(fig)
            plt.close(fig)
    with tab_proc:
        rows = [("Dried hydrochar (kg)", "HC_dry_yield"), ("Hydrochar heat delivered (MJ)", "MJ_hydrochar"),
                ("Process water (kg)", "PW_yield"), ("Biogas (m³)", "biogas_m3"),
                ("Biomethane (m³)", "biomethane_m3"), ("CH4 from AD (kg)", "CH4_mass_AD"),
                ("HTC electricity (kWh)", "E_HTC_elec_kWh"), ("Dryer electricity (kWh)", "E_dryer_heat_kWh"),
                ("Digester heat, natural gas (MJ)", "Q_AD_util_MJ"), ("Tap water added before HTC (kg)", "added_water_HTC"),
                ("Dilution water before AD, Pathway A (m³)", "dilution_water_PW_m3"),
                ("Dilution water before AD, Pathway B (kg)", "dilution_water_AD_kg")]
        st.dataframe(pd.DataFrame([{"Output (per t wet feed)": l, "Value": round(float(wt[k]), 3)}
                                   for l, k in rows if wt.get(k) is not None]),
                     use_container_width=True, hide_index=True)
    with tab_bg:
        st.dataframe(R["log"], use_container_width=True, hide_index=True)
else:
    st.info("Set the inputs in the sidebar and click **Run**. Continuous mode takes about 1 min (AD spin-up).")
