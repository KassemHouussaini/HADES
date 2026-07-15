"""
app.py — HTC-AD Process Simulator (Streamlit front-end)
Run with:  streamlit run app.py

This app now imports ALL process logic (build_system, spinup_ss_state,
mass balance, exchange builders, ADM1 coupling) from system_builder.py —
the exact same module the notebook uses. Nothing about the model itself
is duplicated here anymore. If you change the model, edit
ADM1/system_builder.py once and both the notebook and this app pick it
up automatically.
"""

import os
import sys
import importlib.util

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ─────────────────────────────────────────────────────────────────────────────
# LOAD system_builder.py BY PATH (robust regardless of cwd / folder naming)
# ─────────────────────────────────────────────────────────────────────────────

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Adjust this if your ADM1 folder is somewhere else relative to app.py
_SYSTEM_BUILDER_PATH = os.path.join(_APP_DIR, "ADM1", "system_builder.py")

sb = _load_module("system_builder", _SYSTEM_BUILDER_PATH)

SystemContext = sb.SystemContext
build_system = sb.build_system
spinup_ss_state = sb.spinup_ss_state
get_or_create = sb.get_or_create

# Path to your ML model file. Spaces in the filename are fine — it's
# loaded by path, not by module name.
ML_MODULE_PATH = os.path.join(
    _APP_DIR, "ML training scripts and models", "ML2 model.py"
)


def release_bw_locks():
    """Force-close all open Brightway/peewee database connections."""
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
        try:
            from bw2data.project import ProjectDataset
            ProjectDataset._meta.database.close()
        except Exception:
            pass
    except Exception:
        pass


# ============================================================
# LOCATION REGISTRY — 3-level hierarchy: Country → Province → City
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
        # ... (trim as needed / keep your full list from before)
    },
    "Germany":        {"—": {"code": "DE", "fallback": ["RER", "RoW", "GLO"]}},
    "France":         {"—": {"code": "FR", "fallback": ["RER", "RoW", "GLO"]}},
    "United Kingdom": {"—": {"code": "GB", "fallback": ["RER", "RoW", "GLO"]}},
    "Rest of World":  {"—": {"code": "RoW", "fallback": ["GLO"]}},
    "Global":         {"—": {"code": "GLO", "fallback": ["RoW"]}},
}

PROCESS_TEMPLATES = {
    "electricity_htc": [
        "market for electricity, medium voltage, {loc}",
        "market for electricity, medium voltage",
    ],
    "electricity_ad": [
        "market for electricity, medium voltage, {loc}",
        "market for electricity, medium voltage",
    ],
    "heat_coal": [
        "heat production, at hard coal industrial furnace 1-10MW, {loc}",
        "heat production, at hard coal industrial furnace 1-10MW",
    ],
    "heat_ng_ad": [
        "heat production, natural gas, {loc}",
        "heat and power co-generation, natural gas, {loc}",
        "heat production, natural gas",
    ],
    "heat_ng_drying": [
        "heat production, natural gas, {loc}",
        "heat production, natural gas",
    ],
    "heat_ng_market": [
        "market for heat, district or industrial, natural gas, {loc}",
        "market for heat, district or industrial, natural gas",
    ],
    "tap_water": [
        "market for tap water, tap water, {loc}",
        "market for tap water, {loc}",
        "market for tap water",
    ],
    "wastewater": [
        "treatment of wastewater, average, {loc}",
        "treatment of wastewater, {loc}",
        "treatment of wastewater, average",
    ],
}


def resolve_provider(ei_db, template_key, loc_code, fallback_codes):
    templates = PROCESS_TEMPLATES[template_key]
    all_codes = [loc_code] + list(fallback_codes)
    for code in all_codes:
        for tmpl in templates:
            query = tmpl.replace("{loc}", code) if "{loc}" in tmpl else tmpl
            try:
                results = ei_db.search(query)
                exact = [a for a in results if a.get("location", "") == code]
                hit = exact[0] if exact else (results[0] if results else None)
                if hit:
                    is_proxy = hit.get("location", "") != loc_code
                    return hit, hit.get("location", "?"), is_proxy
            except Exception:
                continue
    raise ValueError(
        f"Could not resolve '{template_key}' for '{loc_code}' or fallbacks {fallback_codes}."
    )


def resolve_all_providers(ei_db, country, province):
    loc_info = LOCATION_HIERARCHY.get(country, {}).get(province)
    if loc_info is None:
        loc_info = {"code": "GLO", "fallback": ["RoW"]}

    loc_code = loc_info["code"]
    fallbacks = loc_info["fallback"]

    providers = {}
    log_rows = []

    for key in PROCESS_TEMPLATES:
        act, resolved_loc, is_proxy = resolve_provider(ei_db, key, loc_code, fallbacks)
        providers[key] = act
        log_rows.append({
            "Process": key,
            "Activity": act["name"],
            "Resolved location": resolved_loc,
            "Status": "⚠️ proxy" if is_proxy else "✅ exact",
        })

    return providers, loc_code, log_rows


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="HTC–AD Simulator", page_icon="⚗️", layout="wide")
st.title("⚗️ HTC–AD Process Simulator")
st.caption("Hydrothermal Carbonisation + Anaerobic Digestion — Mass Balance & LCA")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("🔧 Control Panel")

    st.subheader("🌍 Plant Location")
    _countries = [c for c in LOCATION_HIERARCHY if c not in ("Rest of World", "Global")]
    _countries_full = _countries + ["Rest of World", "Global"]

    sel_country = st.selectbox("Country", options=_countries_full,
                                index=_countries_full.index("Canada"), key="sel_country")
    _provinces = list(LOCATION_HIERARCHY.get(sel_country, {"—": {}}).keys())
    sel_province = st.selectbox("Province / State / Region", options=_provinces, key="sel_province")

    _preview = LOCATION_HIERARCHY.get(sel_country, {}).get(sel_province, {"code": "GLO"})
    st.caption(f"📍 ecoinvent location code: **{_preview['code']}**")
    st.divider()

    st.subheader("Process Pathway")
    PATHWAY = st.radio("Order of operations", options=["A", "B"],
                        captions=["HTC first → AD on process water", "AD first → HTC on digestate"])
    st.divider()

    st.subheader("Feedstock")
    m_feedstock = st.number_input("Feedstock mass (kg wet)", value=1000.0, step=100.0, min_value=1.0)
    WC_feed = st.slider("Moisture content — wet basis (%)", 0.0, 95.0, 75.0, step=0.5)

    with st.expander("📊 Ultimate Analysis (dry basis %) — must sum to 100"):
        C_feed = st.number_input("C %", value=47.0, step=0.1)
        H_feed = st.number_input("H %", value=6.3, step=0.1)
        N_feed = st.number_input("N %", value=3.5, step=0.1)
        O_feed = st.number_input("O %", value=38.5, step=0.1)
        S_feed = st.number_input("S %", value=0.2, step=0.05)
        Ash_feed = st.number_input("Ash %", value=4.5, step=0.1)
        elem_sum = C_feed + H_feed + N_feed + O_feed + S_feed + Ash_feed
        if abs(elem_sum - 100.0) > 0.5:
            st.error(f"⚠️ Sums to {elem_sum:.1f}% — must be 100%")
        else:
            st.success(f"✅ Sums to {elem_sum:.1f}%")

    with st.expander("📋 Proximate Analysis (dry basis %)"):
        V_feed = st.number_input("Volatile matter %", value=80.5, step=0.1)
        Fc_feed = st.number_input("Fixed carbon %", value=15.0, step=0.1)

    st.divider()
    st.subheader("HTC Conditions")
    T_HTC = st.slider("Temperature (°C)", 150, 280, 180, step=5)
    RT_HTC = st.slider("Residence time (min)", 10, 240, 60, step=5)
    target_moisture_HTC = st.slider("Slurry moisture target (fraction)", 0.50, 0.95, 0.80, step=0.01)
    MC_char = st.slider("Wet hydrochar moisture (kg/kg)", 0.30, 0.90, 0.75, step=0.01)
    target_MC_char_dry = st.slider("Dried hydrochar moisture (kg/kg)", 0.0, 0.30, 0.10, step=0.01)

    st.divider()
    st.subheader("AD Conditions")
    adm1_mode = st.selectbox("AD Mode", ["continuous", "batch"])
    TS_target = st.slider("Feed total solids target (kg/kg)", 0.02, 0.15, 0.08, step=0.01)
    COD_conc_target = st.number_input("PW COD dilution target (kgCOD/m³)", value=10.0, step=1.0)
    ISR_batch = st.number_input("Inoculum-to-substrate ratio (batch)", value=2.0, step=0.5)

    if adm1_mode == "continuous":
        adm1_HRT_days = st.number_input("HRT (days)", value=30, min_value=5, max_value=90)
        t_spinup = st.number_input("Spin-up duration (days)", value=400, min_value=100)
        adm1_t_sim = None
    else:
        adm1_HRT_days = None
        t_spinup = 400
        adm1_t_sim = None

    st.divider()
    run_button = st.button("▶ Run Simulation", type="primary", use_container_width=True)
    st.caption("⏱ Continuous mode with spin-up takes ~30–60 s")

# ============================================================
# PARAMETER SUMMARY
# ============================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Pathway", f"{'A: HTC→AD' if PATHWAY == 'A' else 'B: AD→HTC'}")
col2.metric("Feedstock", f"{m_feedstock:.0f} kg")
col3.metric("HTC Temp", f"{T_HTC} °C")
col4.metric("AD Mode", adm1_mode.capitalize())
st.divider()

# ============================================================
# VALIDATION
# ============================================================
errors = []
elem_sum = C_feed + H_feed + N_feed + O_feed + S_feed + Ash_feed
if abs(elem_sum - 100.0) > 0.5:
    errors.append(f"Ultimate analysis sums to {elem_sum:.1f}% — must be 100%")
if not (0.0 < target_moisture_HTC < 1.0):
    errors.append("HTC moisture target must be between 0 and 1")
if not (0.0 < TS_target < 1.0):
    errors.append("TS target must be between 0 and 1")
if adm1_mode == "continuous" and (adm1_HRT_days is None or adm1_HRT_days <= 0):
    errors.append("HRT must be > 0 in continuous mode")
if errors:
    for e in errors:
        st.error(f"❌ {e}")
    st.stop()

# ============================================================
# SIMULATION — only when button clicked
# ============================================================
if run_button:
    release_bw_locks()

    default_params = {
        "WC": WC_feed, "C_feed": C_feed, "H_feed": H_feed,
        "N_feed": N_feed, "O_feed": O_feed, "S_feed": S_feed,
        "Ash_feed": Ash_feed, "VM_feed": V_feed, "FC_feed": Fc_feed,
    }
    ml_params = {**default_params, "T": T_HTC, "RT": RT_HTC}

    progress = st.progress(0, text="Initialising Brightway…")

    try:
        import bw2data as bd
        import bw2calc as bc

        bd.projects.set_current("HTC parametrization")
        ei = bd.Database("ecoinvent-3.10.1-cutoff")
        bios = bd.Database("ecoinvent-3.10.1-biosphere")

        progress.progress(10, text="Brightway connected — resolving providers…")

        with st.spinner(f"Resolving providers for {sel_province}, {sel_country}…"):
            providers, loc_code, resolution_log = resolve_all_providers(ei, sel_country, sel_province)

        elec = providers["electricity_htc"]
        ad_elec = providers["electricity_ad"]
        ad_heat = providers["heat_ng_ad"]
        coal_heat = providers["heat_coal"]
        drying_heat = providers["heat_ng_drying"]
        ng_heat_market = providers["heat_ng_market"]
        tap_water = providers["tap_water"]
        wastewater_treat = providers["wastewater"]

        psa_ref = ei.search("biogas purification to biomethane by pressure swing adsorption")[0]
        biomethane_heat_ref = ei.search("heat production, biomethane, at boiler condensing modulating <100kW")[0]

        co2_air = [a for a in bios if "Carbon dioxide, non-fossil" in a["name"] and "air" in a["categories"]][0]
        co_air = [a for a in bios if "Carbon monoxide, non-fossil" in a["name"] and "air" in a["categories"]][0]

        proxies = [r for r in resolution_log if "proxy" in r["Status"]]
        if proxies:
            st.warning(f"⚠️ {len(proxies)} process(es) used a proxy location for "
                       f"**{sel_province}** (code: `{loc_code}`). See details below.")
        else:
            st.success(f"✅ All providers resolved exactly for `{loc_code}`.")

        with st.expander("🔌 Resolved background providers", expanded=bool(proxies)):
            st.dataframe(pd.DataFrame(resolution_log), use_container_width=True, hide_index=True)

        progress.progress(15, text=f"Providers resolved ({loc_code}) — setting up foreground DB…")

        FOREGROUND_DB = f"Pathway{PATHWAY}"
        if FOREGROUND_DB not in bd.databases:
            bd.Database(FOREGROUND_DB).register()
        db = bd.Database(FOREGROUND_DB)

        htc_reactor      = get_or_create(db, "htc_reactor",         "HTC reactor",                                           "unit")
        drying           = get_or_create(db, "hydrochar_drying",     "Hydrochar drying",                                      "unit")
        heat_prod        = get_or_create(db, "heat_prod_hydrochar",  "Heat production from hydrochar",                        "megajoule")
        sub_heat_coal    = get_or_create(db, "substitute_heat_coal", "Substitution of heat from hard coal",                   "megajoule", location="QC-CA")
        ad_process       = get_or_create(db, "anaerobic_digestion",
                                         "Anaerobic digestion of HTC process water (ADM1)" if PATHWAY == "A"
                                         else "Anaerobic digestion of raw feedstock (ADM1)", "unit")
        dewatering       = get_or_create(db, "digestate dewatering", "dewatering_B",                                          "unit")
        biogas_upgrading = get_or_create(db, "biogas_upgrading",     "Biogas upgrading (PSA)",                                "unit")
        heat_biomethane  = get_or_create(db, "heat_biomethane",      "Heat production from biomethane",                       "megajoule")
        heat_naturalgas  = get_or_create(db, "heat_naturalgas",      "Substitution of heat from Natural Gas",                 "megajoule")
        waste_treatment  = get_or_create(db, "waste_treatment_1kg",  "Waste treatment of organic waste (HTC system)",         "kilogram")

        progress.progress(30, text="Foreground activities ready — building context…")

        # ── Build the SAME SystemContext the notebook uses ─────────────────
        ctx = sb.SystemContext(
            ei=ei, bios=bios, FOREGROUND_DB=FOREGROUND_DB,
            elec=elec, ad_elec=ad_elec, ad_heat=ad_heat, drying_heat=drying_heat,
            coal_heat=coal_heat, ng_heat_market=ng_heat_market,
            biomethane_heat_ref=biomethane_heat_ref, tap_water=tap_water,
            wastewater_treat=wastewater_treat, psa_ref=psa_ref,
            co2_air=co2_air, co_air=co_air,
            htc_reactor=htc_reactor, drying=drying, heat_prod=heat_prod,
            sub_heat_coal=sub_heat_coal, ad_process=ad_process, dewatering=dewatering,
            biogas_upgrading=biogas_upgrading, heat_biomethane=heat_biomethane,
            heat_naturalgas=heat_naturalgas, waste_treatment=waste_treatment,
            PATHWAY=PATHWAY, m_feedstock=m_feedstock,
            default_params=default_params, ml_params=ml_params,
            T_HTC=T_HTC, RT_HTC=RT_HTC, target_moisture_HTC=target_moisture_HTC,
            MC_char=MC_char, target_MC_char_dry=target_MC_char_dry,
            COD_conc_target=COD_conc_target, ISR_batch=ISR_batch,
            location_country=sel_country, location_province=sel_province, location_code=loc_code,
        )

        progress.progress(45, text="Context ready — running simulation…")

        if adm1_mode == "continuous":
            ss_adapted = spinup_ss_state(
                ctx, pathway=PATHWAY, adm1_HRT_days=adm1_HRT_days,
                t_spinup=t_spinup, verbose=False, ml_module=ML_MODULE_PATH,
            )
        else:
            ss_adapted = None

        progress.progress(60, text="Building LCA system…")

        wt, ss_state, adm1_out, feed_inputs, inv = build_system(
            ctx,
            ml_module=ML_MODULE_PATH,
            adm1_mode=adm1_mode,
            adm1_HRT_days=adm1_HRT_days,
            adm1_t_sim_days=adm1_t_sim,
            adm1_ss_state=ss_adapted,
            ISR_batch=ISR_batch,
        )

        progress.progress(90, text="Running LCA…")

        method = ("IMPACT World+ v2.0.1, footprint version", "climate change", "carbon footprint")
        lca = bc.LCA({wt: 1}, method)
        lca.lci()
        lca.lcia()
        gwp_score = lca.score

        progress.progress(100, text="Done!")
        st.session_state["wt"] = wt
        st.session_state["gwp"] = gwp_score
        st.session_state["pathway"] = PATHWAY
        st.session_state["T_HTC"] = T_HTC
        st.session_state["RT_HTC"] = RT_HTC

    except Exception as e:
        progress.empty()
        st.error(f"❌ Simulation failed: {e}")
        st.exception(e)
        st.stop()

# ============================================================
# RESULTS  (unchanged from your original app — reads from wt / gwp)
# ============================================================
if "wt" in st.session_state:
    wt = st.session_state["wt"]
    gwp = st.session_state["gwp"]

    st.subheader(f"📊 Results — Pathway {st.session_state['pathway']}")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("GWP (kg CO₂-eq / FU)", f"{gwp:.2f}")
    k2.metric("Hydrochar (kg wet)", f"{wt.get('HC_wet_yield', 0):.1f}")
    k3.metric("Biogas (m³)", f"{wt.get('biogas_m3', 0):.3f}")
    k4.metric("Biomethane (m³)", f"{wt.get('biomethane_m3', 0):.3f}")
    k5.metric("Process water (kg)", f"{wt.get('PW_yield', 0):.1f}")

    tab_mb, tab_lca = st.tabs(["📦 Mass Balance", "🌍 LCA"])

    with tab_mb:
        st.subheader("Mass Balance Summary")
        keys_to_show = [
            ("m_feedstock", "Feedstock in (kg)"),
            ("added_water_HTC", "Dilution water added (kg)"),
            ("HC_wet_yield", "Hydrochar — wet (kg)"),
            ("HC_dry_yield", "Hydrochar — dried (kg)"),
            ("PW_yield", "Process water (kg)"),
            ("CO2_yield_HTC", "HTC CO₂ off-gas (kg)"),
            ("CO_yield_HTC", "HTC CO off-gas (kg)"),
            ("CH4_mass_AD", "AD CH₄ produced (kg)"),
            ("CO2_mass_AD", "AD CO₂ produced (kg)"),
            ("NH3_mass_AD", "AD NH₃ produced (kg)"),
            ("biogas_m3", "Biogas (m³)"),
            ("biomethane_m3", "Biomethane (m³)"),
            ("wastewater_drying_m3", "Drying wastewater (m³)"),
            ("dilution_water_PW_m3", "AD dilution water (m³)"),
        ]
        if st.session_state["pathway"] == "B":
            keys_to_show += [
                ("m_cake_dewater_kg", "Dewatered cake to HTC (kg)"),
                ("m_centrate_dewater_kg", "Centrate to WWT (kg)"),
                ("E_centrifuge_kWh", "Centrifuge electricity (kWh)"),
                ("dilution_water_AD_kg", "AD dilution water (kg)"),
            ]
        mb_rows = []
        for key, label in keys_to_show:
            val = wt.get(key)
            if val is not None:
                mb_rows.append({"Stream / Variable": label, "Value": f"{float(val):.4f}"})
        st.dataframe(pd.DataFrame(mb_rows), use_container_width=True, hide_index=True)

        st.subheader("Mass Balance Check")
        mb_checks = {}
        if st.session_state["pathway"] == "A":
            mb_checks["HTC (A)"] = wt.get("MB_HTC_A", "—")
            mb_checks["AD (A)"] = wt.get("MB_AD_A", "—")
        else:
            mb_checks["AD (B)"] = wt.get("MB_AD_B", "—")
            mb_checks["DW (B)"] = wt.get("MB_DW_B", "—")
            mb_checks["HTC (B)"] = wt.get("MB_HTC_B", "—")
        for name, val in mb_checks.items():
            if val == "—":
                st.write(f"**{name}**: —")
            elif abs(float(val)) < 0.01:
                st.success(f"✅ {name} residual: {float(val):.6f} kg")
            else:
                st.warning(f"⚠️ {name} residual: {float(val):.6f} kg")

    with tab_lca:
        st.subheader("Life Cycle Assessment")
        st.metric("GWP100 (kg CO₂-eq per functional unit)", f"{gwp:.4f}")

        st.divider()
        st.subheader("Contribution Analysis")

        method = ("IMPACT World+ v2.0.1, footprint version", "climate change", "carbon footprint")

        import bw2calc as bc

        results = []
        for exc in wt.exchanges():
            if exc["type"] != "technosphere":
                continue
            act = exc.input
            try:
                lca_temp = bc.LCA({act: exc["amount"]}, method=method)
                lca_temp.lci()
                lca_temp.lcia()
                results.append((act["name"], lca_temp.score))
            except Exception:
                pass

        if results:
            df_contrib = pd.DataFrame(results, columns=["Activity", "Impact"])
            df_contrib = df_contrib.sort_values("Impact", ascending=True, key=abs)

            pos = df_contrib[df_contrib["Impact"] > 0]
            neg = df_contrib[df_contrib["Impact"] < 0]

            fig_lca, ax_lca = plt.subplots(figsize=(10, 6))
            colors = plt.cm.tab20(np.linspace(0, 1, len(df_contrib)))
            color_map = dict(zip(df_contrib["Activity"], colors))

            bottom_pos = bottom_neg = 0
            for _, row in pos.iterrows():
                ax_lca.bar(0, row["Impact"], bottom=bottom_pos, color=color_map[row["Activity"]],
                           edgecolor="black", linewidth=0.8, width=0.5, label=row["Activity"])
                bottom_pos += row["Impact"]
            for _, row in neg.iterrows():
                ax_lca.bar(0, row["Impact"], bottom=bottom_neg, color=color_map[row["Activity"]],
                           edgecolor="black", linewidth=0.8, width=0.5, label=row["Activity"])
                bottom_neg += row["Impact"]

            ax_lca.scatter(0, gwp, color="black", s=90, zorder=5, marker="o")
            ax_lca.annotate(f"Net: {gwp:.3e}", xy=(0, gwp), xytext=(0.3, gwp),
                            fontsize=9, va="center",
                            arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
            ax_lca.axhline(0, color="black", linewidth=1.2)
            ax_lca.set_ylabel("Impact (kg CO₂-eq)")
            ax_lca.set_title(f"First-Tier Contributions — Pathway {st.session_state['pathway']}")
            ax_lca.set_xticks([])
            ax_lca.grid(axis="y", linestyle="--", alpha=0.3)
            ax_lca.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_lca)

else:
    st.info("👈 Set your parameters in the sidebar and click **▶ Run Simulation** to begin.")