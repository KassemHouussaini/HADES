"""
system_builder.py
============================================================
SINGLE SOURCE OF TRUTH for the HTC-AD system model.

Both the notebook and the Streamlit app import THIS file and call
build_system() / spinup_ss_state() from here. Nothing about the
process logic should ever be duplicated elsewhere again — if you
need to change mass balances, exchange wiring, or ADM1 coupling,
change it here and both callers pick it up automatically.

Place this file inside the "ADM1" folder, next to ADM1.py and
plot_adm1_batch.py. It loads both of those by explicit file path
(not by `import ADM1`), so it is immune to the folder/module name
collision that happens when a folder and a .py file share a name.

USAGE
-----
    import importlib.util, sys

    def load_module(name, filepath):
        spec = importlib.util.spec_from_file_location(name, filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    sb = load_module("system_builder", "ADM1/system_builder.py")

    ctx = sb.SystemContext(...)                    # see class below
    ss  = sb.spinup_ss_state(ctx, pathway="A", ...)
    wt, ss_state, adm1_out, feed_inputs, inv = sb.build_system(ctx, ...)
"""

import os
import sys
import inspect
import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import bw2data as bd


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ADM1.py AND plot_adm1_batch.py BY EXPLICIT PATH (no `import ADM1`)
# This is the fix for the ADM1-folder-vs-ADM1.py collision.
# ─────────────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_by_path(name, relpath):
    path = os.path.join(_HERE, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_adm1_core = _load_by_path("ADM1_core", "ADM1.py")
_plot_mod = _load_by_path("plot_adm1_batch_core", "plot_adm1_batch.py")

run_adm1 = _adm1_core.run_adm1
translate_pw_to_adm1 = _adm1_core.translate_pw_to_adm1
translate_feedstock_to_adm1 = _adm1_core.translate_feedstock_to_adm1
extract_digestate_composition = _adm1_core.extract_digestate_composition
plot_adm1_results = _plot_mod.plot_adm1_results


def _load_by_exact_path(name, filepath):
    """
    Like _load_by_path, but does NOT join with the ADM1 folder's own
    location — filepath is used exactly as given (absolute, or relative
    to the current working directory / notebook). Used for the ML module,
    which typically lives in a sibling folder, not inside ADM1/.
    """
    spec = importlib.util.spec_from_file_location(name, filepath)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(
            f"Could not load module '{name}' from '{filepath}' — "
            f"check the path is correct and the file exists."
        )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_ml_module(ml_module):
    """
    Accepts either:
      - a plain module name already on sys.path, e.g. "ML2_model"
      - a filesystem path ending in .py, e.g. "ML training scripts and
        models/ML2 model.py" (spaces are fine — loaded by path, not by
        name). The path is resolved relative to the current working
        directory / notebook, NOT relative to the ADM1 folder.
    """
    if ml_module.endswith(".py"):
        resolved = ml_module if os.path.isabs(ml_module) else os.path.abspath(ml_module)
        if not os.path.exists(resolved):
            raise FileNotFoundError(
                f"ML module not found at '{resolved}'. "
                f"ml_module='{ml_module}' is resolved relative to the "
                f"current working directory (where the notebook/app is "
                f"running from), not relative to the ADM1 folder. "
                f"Pass an absolute path if unsure."
            )
        return _load_by_exact_path("HTC_ML_module", resolved)
    mod = importlib.import_module(ml_module)
    return importlib.reload(mod)


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL / ENGINEERING CONSTANTS  (formerly notebook Cell 3)
# ─────────────────────────────────────────────────────────────────────────────

T_0 = 25
cp_water = 4.18
eta_heater = 0.9
latent_heat_evap = 2260 / 3600

MW_C = 12
MW_H = 1
MW_O = 16
MW_N = 14
MW_S = 32
MW_CO2 = 44
MW_CO = 28

CH4_density = 0.716
CO2_density = 1.977

BSM2_INIT = {
    "S_su": 0.012394, "S_aa": 0.0055432, "S_fa": 0.10741,
    "S_va": 0.012333, "S_bu": 0.014003, "S_pro": 0.017584,
    "S_ac": 0.089315, "S_h2": 2.51e-7, "S_ch4": 0.05549,
    "S_IC": 0.095149, "S_IN": 0.094468, "S_I": 0.13087,
    "X_xc": 0.10792, "X_ch": 0.020517, "X_pr": 0.08422,
    "X_li": 0.043629, "X_su": 0.31222, "X_aa": 0.93167,
    "X_fa": 0.33839, "X_c4": 0.33577, "X_pro": 0.10112,
    "X_ac": 0.67724, "X_h2": 0.28484, "X_I": 17.2162,
    "S_cation": 0.0521, "S_anion": 0.0052101,
    "S_H_ion": 5.46e-8, "S_va_ion": 0.012284, "S_bu_ion": 0.013953,
    "S_pro_ion": 0.017511, "S_ac_ion": 0.089035, "S_hco3_ion": 0.08568,
    "S_co2": 0.0094689, "S_nh3": 0.001884, "S_nh4_ion": 0.092584,
    "S_gas_h2": 1.1e-5, "S_gas_ch4": 1.6535, "S_gas_co2": 0.01354,
    "pH": 7.4655,
}


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM CONTEXT — replaces every "global X" in the old notebook/app code.
# Build one of these once (per pathway/location), pass it to build_system().
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SystemContext:
    # Brightway
    ei: Any
    bios: Any
    FOREGROUND_DB: str

    # Background technosphere providers
    elec: Any
    ad_elec: Any
    ad_heat: Any
    drying_heat: Any
    coal_heat: Any
    ng_heat_market: Any
    biomethane_heat_ref: Any
    tap_water: Any
    wastewater_treat: Any
    psa_ref: Any

    # Biosphere flows
    co2_air: Any
    co_air: Any

    # Foreground activities
    htc_reactor: Any
    drying: Any
    heat_prod: Any
    sub_heat_coal: Any
    ad_process: Any
    dewatering: Any
    biogas_upgrading: Any
    heat_biomethane: Any
    heat_naturalgas: Any
    waste_treatment: Any

    # Process parameters
    PATHWAY: str
    m_feedstock: float
    default_params: dict
    ml_params: dict
    T_HTC: float
    RT_HTC: float
    target_moisture_HTC: float
    MC_char: float
    target_MC_char_dry: float
    COD_conc_target: float = 10.0
    ISR_batch: float = 2.0

    # Optional metadata (used by Streamlit export / diagrams)
    location_country: str = ""
    location_province: str = ""
    location_code: str = ""


def get_or_create(db, code, name, unit, location="CA-QC", type_="process"):
    """
    Return the activity matching `code` in `db`, or create it if absent.
    Clears its exchanges if it already exists (idempotent rebuild) and
    always (re)adds a production exchange of amount 1.
    """
    existing = [a for a in db if a.get("code") == code]
    if existing:
        act = existing[0]
        for exc in list(act.exchanges()):
            exc.delete()
        act["name"] = name
        act.save()
    else:
        act = db.new_activity(code=code, name=name, unit=unit,
                               location=location, type=type_)
        act.save()
    act.new_exchange(input=act, amount=1, type="production", unit=unit).save()
    return act


def run_ml_model(HTC_parametric_ML, params=None):
    p = params if params is not None else {}
    valid_args = inspect.signature(HTC_parametric_ML).parameters.keys()
    usable_args = {k: v for k, v in p.items() if k in valid_args}
    result = HTC_parametric_ML(**usable_args)
    return result if isinstance(result, tuple) and len(result) == 2 else (None, result)


# ─────────────────────────────────────────────────────────────────────────────
# MASS BALANCE FUNCTIONS  (formerly notebook Cell 7)
# ─────────────────────────────────────────────────────────────────────────────

def compute_htc_mass_balance(m_feed_eff, WC_feed, Y_char_pct, T_HTC,
                              MC_char, target_MC_char_dry):
    dry_fraction = 1 - WC_feed
    Y_char_dry = Y_char_pct / 100

    CO2_frac_mol = 0.95
    CO_frac_mol = 0.05

    dry_mass_in = m_feed_eff * dry_fraction
    water_mass_in = m_feed_eff * WC_feed

    HC_dry_mass = dry_mass_in * Y_char_dry
    HC_water_retained = min(HC_dry_mass * (MC_char / (1 - MC_char)),
                             water_mass_in * 0.90)
    HC_wet_yield = HC_dry_mass + HC_water_retained

    print(f"\n[MB diagnostic]")
    print(f"  m_feed_eff        = {m_feed_eff:.2f} kg")
    print(f"  WC_feed           = {WC_feed:.4f}")
    print(f"  dry_mass_in       = {dry_mass_in:.2f} kg")
    print(f"  water_mass_in     = {water_mass_in:.2f} kg")
    print(f"  Y_char_pct        = {Y_char_pct:.2f} %")
    print(f"  HC_dry_mass       = {HC_dry_mass:.2f} kg")
    print(f"  MC_char           = {MC_char:.3f}")
    print(f"  HC_water_retained = {HC_water_retained:.2f} kg")
    print(f"  HC_wet_yield      = {HC_wet_yield:.2f} kg")

    Gas_yield_dry = max(0.0, 0.121 * T_HTC - 18.34) / 100
    Gas_yield_total = m_feed_eff * dry_fraction * Gas_yield_dry
    CO2_yield = Gas_yield_total * (CO2_frac_mol * MW_CO2) / (CO2_frac_mol * MW_CO2 + CO_frac_mol * MW_CO)
    CO_yield = Gas_yield_total * (CO_frac_mol * MW_CO) / (CO2_frac_mol * MW_CO2 + CO_frac_mol * MW_CO)
    PW_yield = m_feed_eff - (HC_wet_yield + CO2_yield + CO_yield)

    HC_dry_yield = HC_wet_yield * (1 - MC_char) / (1 - target_MC_char_dry)
    wastewater_m3 = (HC_wet_yield - HC_dry_yield) / 1000
    elec_kWh = (HC_wet_yield - HC_dry_yield) * latent_heat_evap
    elec_MJ = elec_kWh * 3.6

    Q_heat_kWh = (m_feed_eff * cp_water * (T_HTC - T_0)) / 3600 / eta_heater

    return dict(
        HC_dry_mass=HC_dry_mass, HC_wet_yield=HC_wet_yield, HC_dry_yield=HC_dry_yield,
        Gas_yield_total=Gas_yield_total, CO2_yield=CO2_yield, CO_yield=CO_yield,
        PW_yield=PW_yield, wastewater_m3=wastewater_m3,
        Q_heat_kWh=Q_heat_kWh, elec_MJ=elec_MJ, elec_kWh=elec_kWh,
        Y_gas_drybasis=Gas_yield_total / dry_mass_in * 100,
    )


def print_mass_balance_diagnostic(m_feed_eff, mb, WC_feed):
    print(f"\n========== MASS BALANCE DIAGNOSTIC ==========")
    print(f"m_feedstock_effective : {m_feed_eff:.4f} kg")
    print(f"HC_wet_yield          : {mb['HC_wet_yield']:.4f} kg")
    print(f"CO2_yield + CO_yield  : {mb['CO2_yield'] + mb['CO_yield']:.4f} kg")
    print(f"Sum outputs           : {mb['HC_wet_yield'] + mb['CO2_yield'] + mb['CO_yield']:.4f} kg")
    print(f"PW_yield              : {mb['PW_yield']:.4f} kg  "
          f"{'⚠️ NEGATIVE' if mb['PW_yield'] < 0 else '✅ OK'}")


# ─────────────────────────────────────────────────────────────────────────────
# ADM1 RUNNER  (formerly notebook Cell 8)
# ─────────────────────────────────────────────────────────────────────────────

def _blank_inputs(pw_inputs):
    blank = {k: v for k, v in pw_inputs.items()}
    for k in ["S_su", "S_aa", "S_fa", "S_ac", "S_pro", "S_bu", "S_va",
              "X_xc", "X_ch", "X_pr", "X_li", "X_I",
              "_total_COD_kgCOD_m3", "_COD_conc"]:
        if k in blank:
            blank[k] = 0.0
    for sp in ["furan", "alcohol", "phenol", "acid", "ketone", "humic"]:
        blank[f"S_I_{sp}"] = 0.0
        blank[f"_inf_S_I_{sp}"] = 0.0
    blank["_pw_mode"] = False
    return blank


def run_adm1_and_correct(pw_inputs, adm1_mode, adm1_q_ad, adm1_HRT_days,
                          adm1_t_sim_days, T_HTC, RT_HTC,
                          ISR_batch=5.0,
                          disable_tox=False, label="", ss_initial_state=None):
    dilution_water_m3 = pw_inputs.get("_dilution_water_m3", 0.0)
    if dilution_water_m3 > 0:
        print(f"[Pathway A | continuous] Dilution water for AD: "
              f"{dilution_water_m3*1000:.1f} L ({dilution_water_m3*1000:.2f} kg)")

    adm1_out = run_adm1(
        substrate=pw_inputs,
        t_batch_days=adm1_t_sim_days if adm1_t_sim_days is not None else 30,
        solvermethod="Radau",
        verbose=True,
        mode=adm1_mode,
        q_ad=adm1_q_ad,
        HRT_days=adm1_HRT_days,
        ISR_batch=ISR_batch,
        t_sim_days=adm1_t_sim_days,
        ss_initial_state=ss_initial_state,
    )

    if adm1_mode == "batch":
        forced_seed = adm1_out.get("_X_seed_used")
        if forced_seed is None:
            raise ValueError("run_adm1 did not return _X_seed_used")

        print(f"⚙️ Synchronizing Blank: Forcing seed={forced_seed:.2f} to match Test")

        blank_inputs = _blank_inputs(pw_inputs)
        blank_inputs["_force_seed_val"] = forced_seed

        adm1_out_blank = run_adm1(
            substrate=blank_inputs,
            t_batch_days=adm1_t_sim_days if adm1_t_sim_days is not None else 30,
            mode="batch",
            V_liq_override=adm1_out["_V_liq"],
            verbose=False,
            ss_initial_state=ss_initial_state,
            ISR_batch=ISR_batch,
        )

        CH4_mass = max(0, adm1_out["CH4_mass_kg"] - adm1_out_blank["CH4_mass_kg"])
        CO2_mass = max(0, adm1_out["CO2_mass_kg"] - adm1_out_blank["CO2_mass_kg"])
        NH3_mass = max(0, adm1_out["NH3_mass_kg"] - adm1_out_blank["NH3_mass_kg"])

        print(f"\n[ADM1 results {label} | batch] "
              f"CH4={CH4_mass:.4f} kg, CO2={CO2_mass:.4f} kg, "
              f"NH3={NH3_mass:.4f} kg  (blank-corrected)")

        V_liq_val = adm1_out["_V_liq"]
        V_gas_val = V_liq_val * (0.4 / 0.6)

        plot_adm1_results(
            adm1_out=adm1_out,
            adm1_out_blank=adm1_out_blank,
            pw_inputs=pw_inputs,
            V_liq=V_liq_val,
            V_gas=V_gas_val,
        )
    else:
        adm1_out_blank = None
        CH4_mass = adm1_out["CH4_mass_kg"]
        CO2_mass = adm1_out["CO2_mass_kg"]
        NH3_mass = adm1_out["NH3_mass_kg"]
        print(f"\n[ADM1 results {label} | continuous] "
              f"CH4={CH4_mass:.4f} kg, CO2={CO2_mass:.4f} kg, "
              f"NH3={NH3_mass:.4f} kg  (time-integrated, no blank)")

        ss_results = adm1_out["simulate_results"]
        ss_end = ss_results.iloc[-20:]
        print(f"\n=== SS DIAGNOSTIC ===")
        print(f"X_ac  (acetoclastic methanogens) : {ss_end['X_ac'].mean():.4f} kgCOD/m³")
        print(f"X_h2  (hydrogenotrophic methano) : {ss_end['X_h2'].mean():.4f} kgCOD/m³")
        print(f"X_pro (propionate oxidisers)     : {ss_end['X_pro'].mean():.4f} kgCOD/m³")
        print(f"S_ac  (acetate)                  : {ss_end['S_ac'].mean():.4f} kgCOD/m³")
        print(f"S_pro (propionate)               : {ss_end['S_pro'].mean():.4f} kgCOD/m³")
        print(f"S_IN  (inorg. nitrogen)          : {ss_end['S_IN'].mean():.6f} kmolN/m³")
        print(f"S_nh3 (free ammonia)             : {ss_end['S_nh3'].mean():.6f} kmolN/m³")
        print(f"pH                               : {ss_end['pH'].mean():.3f}")
        print(f"S_h2  (dissolved H2)             : {ss_end['S_h2'].mean():.2e} kgCOD/m³")
        print(f"=====================\n")

    return CH4_mass, CO2_mass, NH3_mass, adm1_out, adm1_out_blank


# ─────────────────────────────────────────────────────────────────────────────
# BRIGHTWAY EXCHANGE BUILDERS  (formerly notebook Cell 9)
# Fixed: build_ad_exchanges and build_drying_exchanges now take
# ad_elec / ad_heat / drying_heat explicitly instead of relying on
# notebook globals (that was a latent bug in the old code).
# ─────────────────────────────────────────────────────────────────────────────

def build_htc_reactor_exchanges(htc_reactor, elec, tap_water, co2_air, co_air,
                                 wastewater_treat, Q_heat_kWh, CO2_yield, CO_yield,
                                 added_water, PW_yield, PATHWAY,
                                 ad_process=None, digestate_mass=None,
                                 dewatering_activity=None, m_digestate_for_htc=None):
    htc_reactor.new_exchange(input=elec, amount=Q_heat_kWh,
                              type="technosphere", unit="kilowatt hour").save()
    htc_reactor.new_exchange(input=co2_air, amount=CO2_yield,
                              type="biosphere", unit="kilogram").save()
    htc_reactor.new_exchange(input=co_air, amount=CO_yield,
                              type="biosphere", unit="kilogram").save()

    if added_water > 0:
        htc_reactor.new_exchange(input=tap_water, amount=added_water,
                                  type="technosphere", unit="kilogram").save()
        print(f"Added tap water exchange: {added_water:.3f} kg")

    if PATHWAY == "B":
        htc_reactor.new_exchange(input=dewatering_activity, amount=m_digestate_for_htc / 1000,
                                  type="technosphere", unit="cubic meter").save()
        pw_wastewater_m3 = PW_yield / 1000.0
        htc_reactor.new_exchange(input=wastewater_treat, amount=pw_wastewater_m3,
                                  type="technosphere", unit="cubic meter").save()
        print(f"[Pathway B] Dewatered cake -> HTC reactor: {m_digestate_for_htc:.4f} kg")
        print(f"[Pathway B] HTC PW -> wastewater: {pw_wastewater_m3:.5f} m³")


def build_drying_exchanges(drying, elec, drying_heat, wastewater_treat, elec_MJ, wastewater_m3):
    drying.new_exchange(input=drying_heat, amount=elec_MJ,
                         type="technosphere", unit="megajoule").save()
    drying.new_exchange(input=wastewater_treat, amount=wastewater_m3,
                         type="technosphere", unit="cubic meter").save()


def build_coal_substitution_exchange(sub_heat_coal, coal_heat):
    sub_heat_coal.new_exchange(input=coal_heat, amount=1,
                                type="technosphere", unit="megajoule").save()


def build_hydrochar_heat_exchanges(hc_heat, coal_heat, bios, S_char_pct, N_char_pct,
                                    Ash_char_pct, HHV_char):
    SO2_per_MJ = (S_char_pct / 100) * (64 / 32) / HHV_char
    NOx_per_MJ = (N_char_pct / 100) * (46 / 14) / HHV_char

    Ash_coal_pct = 10.0
    PM_scale = (Ash_char_pct / Ash_coal_pct) if Ash_coal_pct > 0 else 1.0

    print(f"[HC heat] SO₂ = {SO2_per_MJ:.6f} kg/MJ  "
          f"(S_char={S_char_pct:.2f}%  HHV={HHV_char:.2f} MJ/kg)")
    print(f"[HC heat] NOx = {NOx_per_MJ:.6f} kg/MJ  "
          f"(N_char={N_char_pct:.2f}%  HHV={HHV_char:.2f} MJ/kg)")
    print(f"[HC heat] PM  scale = {PM_scale:.3f}  "
          f"(Ash_char={Ash_char_pct:.2f}%  Ash_coal={Ash_coal_pct:.1f}%)")

    so2_flow = [f for f in bios if "Sulfur dioxide" in f["name"]
                and "air" in f["categories"]][0]
    nox_flow = [f for f in bios if "Nitrogen oxides" in f["name"]
                and "air" in f["categories"]][0]

    so2_written = False
    nox_written = False

    for exc in coal_heat.exchanges():
        if exc["type"] == "technosphere":
            hc_heat.new_exchange(input=exc.input, amount=0,
                                  type="technosphere", unit=exc["unit"]).save()

        elif exc["type"] == "biosphere":
            name = exc.input["name"].lower()

            if any(w in name for w in [
                "radioactive", "uranium", "radon", "thorium", "lead-210", "polonium",
                "arsenic", "cadmium", "mercury", "antimony", "chromium", "nickel",
                "molybdenum", "vanadium", "thallium", "tin", "beryllium",
                "phosphorus", "dioxin", "benzene", "toluene", "xylene", "pa", "nmvoc"]):
                continue

            if "carbon dioxide, fossil" in name:
                nonf = [f for f in bios if f["name"] == "Carbon dioxide, non-fossil"
                        and "air" in f["categories"]][0]
                hc_heat.new_exchange(input=nonf, amount=exc["amount"],
                                      type="biosphere", unit=exc["unit"]).save()

            elif "carbon monoxide, fossil" in name:
                nonf = [f for f in bios if f["name"] == "Carbon monoxide, non-fossil"
                        and "air" in f["categories"]][0]
                hc_heat.new_exchange(input=nonf, amount=exc["amount"],
                                      type="biosphere", unit=exc["unit"]).save()

            elif "methane, fossil" in name:
                nonf = [f for f in bios if f["name"] == "Methane, non-fossil"
                        and "air" in f["categories"]][0]
                hc_heat.new_exchange(input=nonf, amount=exc["amount"],
                                      type="biosphere", unit=exc["unit"]).save()

            elif "sulfur dioxide" in name:
                if not so2_written:
                    hc_heat.new_exchange(input=so2_flow, amount=SO2_per_MJ,
                                          type="biosphere", unit="kilogram").save()
                    so2_written = True

            elif "nitrogen oxides" in name:
                if not nox_written:
                    hc_heat.new_exchange(input=nox_flow, amount=NOx_per_MJ,
                                          type="biosphere", unit="kilogram").save()
                    nox_written = True

            elif "particulate" in name:
                hc_heat.new_exchange(input=exc.input, amount=exc["amount"] * PM_scale,
                                      type="biosphere", unit=exc["unit"]).save()


def build_ad_exchanges(ad_process, bios, ei, CH4_mass, NH3_mass, m_AD_in,
                        adm1_mode, PATHWAY, ad_elec, ad_heat,
                        wastewater_treat=None, digestate_mass=None,
                        dilution_water_m3=0.0, tap_water=None):
    f_fug_CH4 = 0.02
    f_fug_NH3 = 0.05
    CH4_fug = CH4_mass * f_fug_CH4
    NH3_fug = NH3_mass * f_fug_NH3

    ch4_air = [a for a in bios if "Methane, non-fossil" in a["name"] and "air" in a["categories"]][0]
    nh3_air = [a for a in bios if "Ammonia" in a["name"] and "air" in a["categories"]][0]

    ad_process.new_exchange(input=ch4_air, amount=CH4_fug, type="biosphere", unit="kilogram").save()
    ad_process.new_exchange(input=nh3_air, amount=NH3_fug, type="biosphere", unit="kilogram").save()

    e_heat_AD = 0.1
    e_elec_AD = 0.006

    Q_heat_AD_MJ = m_AD_in * e_heat_AD
    E_elec_AD_kWh = m_AD_in * e_elec_AD

    ad_process.new_exchange(input=ad_elec, amount=E_elec_AD_kWh, type="technosphere", unit="kilowatt hour").save()
    ad_process.new_exchange(input=ad_heat, amount=Q_heat_AD_MJ, type="technosphere", unit="megajoule").save()

    if dilution_water_m3 > 0 and tap_water is not None:
        ad_process.new_exchange(input=tap_water, amount=dilution_water_m3 * 1000,
                                 type="technosphere", unit="kilogram").save()
        print(f"[AD LCA] Added dilution tap water: {dilution_water_m3*1000:.2f} kg")
    elif dilution_water_m3 > 0 and tap_water is None:
        print(f"⚠️  [AD LCA] dilution_water_m3={dilution_water_m3*1000:.1f} kg "
              f"but tap_water activity not passed — exchange skipped")

    print(f"[AD energy | {adm1_mode}] m_AD_in={m_AD_in:.3f} kg | "
          f"Heat={Q_heat_AD_MJ:.4f} MJ | Elec={E_elec_AD_kWh:.4f} kWh")

    if PATHWAY == "A" and wastewater_treat is not None and digestate_mass is not None:
        ad_process.new_exchange(input=wastewater_treat, amount=digestate_mass / 1000,
                                 type="technosphere", unit="cubic meter").save()

    return CH4_fug, NH3_fug, Q_heat_AD_MJ, E_elec_AD_kWh


def build_dewatering_exchanges(dewatering, elec, wastewater_treat,
                                E_centrifuge_kWh, centrate_m3,
                                ad_process, m_digestate_total):
    dewatering.new_exchange(input=elec, amount=E_centrifuge_kWh,
                             type="technosphere", unit="kilowatt hour").save()
    dewatering.new_exchange(input=wastewater_treat, amount=centrate_m3,
                             type="technosphere", unit="cubic meter").save()
    print(f"[Dewatering] Elec={E_centrifuge_kWh:.3f} kWh  Centrate→WWT={centrate_m3:.4f} m³")


def build_biogas_upgrading_exchanges(biogas_upgrading, psa_ref, ei, biomethane_m3):
    for exc in psa_ref.exchanges():
        exc_type = exc["type"]
        exc_input = exc.input
        exc_amount = exc["amount"]
        exc_unit = exc["unit"]
        name_lower = exc_input["name"].lower()

        if exc_type == "production" and "biomethane" in name_lower:
            continue
        if exc_type == "technosphere" and ("biogas" in name_lower or "anaerobic digestion" in name_lower):
            continue

        if exc_type == "technosphere":
            query = exc_input["name"]
            if "electricity" in query.lower():
                matches = [a for a in ei.search("market for electricity, low voltage")
                           if "CA-QC" in a.get("location", "")]
            else:
                matches = [a for a in ei.search(query) if "CA-QC" in a.get("location", "")]
                if not matches:
                    matches = [a for a in ei.search(query)
                               if any(t in a.get("location", "") for t in ["GLO", "RoW", "RER"])]
            regional = matches[0] if matches else exc_input
            biogas_upgrading.new_exchange(input=regional, amount=exc_amount * biomethane_m3,
                                           type=exc_type, unit=exc_unit).save()
        elif exc_type == "biosphere":
            biogas_upgrading.new_exchange(input=exc_input, amount=exc_amount * biomethane_m3,
                                           type=exc_type, unit=exc_unit).save()


# ─────────────────────────────────────────────────────────────────────────────
# SPIN-UP  (formerly notebook Cell 10, part 1)
# ─────────────────────────────────────────────────────────────────────────────

def spinup_ss_state(ctx: SystemContext, pathway: str,
                     adm1_HRT_days: int = 20,
                     t_spinup: int = 400,
                     verbose: bool = True,
                     TS_target: float = 0.08,
                     ml_module: str = None) -> dict:
    """
    Run a single ADM1 spin-up to get a properly adapted SS state for
    `pathway`, then pass the result as adm1_ss_state to build_system().

    ml_module may be a bare module name (already on sys.path) or a
    filesystem path ending in .py (spaces in the filename are fine).
    """
    BSM2_BIOMASS_ONLY = {k: v for k, v in BSM2_INIT.items()}
    for zero_key in ["X_I", "X_xc", "X_ch", "X_pr", "X_li",
                      "S_su", "S_aa", "S_fa", "S_va", "S_bu",
                      "S_pro", "S_ac", "S_h2", "S_ch4", "S_I"]:
        BSM2_BIOMASS_ONLY[zero_key] = 0.0

    ctx.PATHWAY = pathway

    if verbose:
        print(f"\n{'─'*55}")
        print(f"SPIN-UP — Pathway {pathway} | HRT={adm1_HRT_days} d | "
              f"t_sim={t_spinup} d")
        print(f"{'─'*55}")

    _, ss_out, _, _, _ = build_system(
        ctx,
        ml_module=ml_module,
        adm1_mode="continuous",
        adm1_HRT_days=adm1_HRT_days,
        adm1_q_ad=None,
        adm1_t_sim_days=t_spinup,
        adm1_ss_state=BSM2_BIOMASS_ONLY,
        TS_target=TS_target,
    )

    if ss_out is None:
        print(f"⚠️  Spin-up returned None — falling back to BSM2_BIOMASS_ONLY")
        return BSM2_BIOMASS_ONLY

    ph = ss_out.get("pH", 0)
    xac = ss_out.get("X_ac", 0)
    sac = ss_out.get("S_ac", 0)

    if verbose:
        print(f"Spin-up result: pH={ph:.3f}  X_ac={xac:.4f}  S_ac={sac:.4f}")

    if ph < 6.5 or xac < 0.05:
        print(f"⚠️  Spin-up crashed (pH={ph:.2f}, X_ac={xac:.4f}) "
              f"— falling back to BSM2_BIOMASS_ONLY")
        return BSM2_BIOMASS_ONLY

    print(f"✅ Spin-up converged — using adapted SS state for Pathway {pathway}")
    return ss_out


# ─────────────────────────────────────────────────────────────────────────────
# BUILD SYSTEM  (formerly notebook Cell 10, part 2 — the orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

def build_system(ctx: SystemContext,
                  ml_module,
                  adm1_mode="batch",
                  adm1_HRT_days=None,
                  adm1_q_ad=None,
                  adm1_t_sim_days=None,
                  adm1_ss_state=None,
                  ISR_batch=None,
                  TS_target=0.08):
    """
    Build (or rebuild) the full HTC-AD foreground LCA system.

    ml_module: bare module name OR a path ending in .py (see _load_ml_module).
    ISR_batch: if None, falls back to ctx.ISR_batch.

    Returns
    -------
    waste_treatment, ss_state, adm1_out, feed_inputs, inv
    """
    # ── Unpack context into local names (keeps body identical to the
    #    original notebook logic — only the *source* of these values changed) ──
    PATHWAY = ctx.PATHWAY
    m_feedstock = ctx.m_feedstock
    default_params = ctx.default_params
    ml_params = ctx.ml_params
    T_HTC = ctx.T_HTC
    RT_HTC = ctx.RT_HTC
    target_moisture_HTC = ctx.target_moisture_HTC
    MC_char = ctx.MC_char
    target_MC_char_dry = ctx.target_MC_char_dry
    COD_conc_target = ctx.COD_conc_target
    ei = ctx.ei
    bios = ctx.bios
    elec = ctx.elec
    ad_elec = ctx.ad_elec
    ad_heat = ctx.ad_heat
    drying_heat = ctx.drying_heat
    coal_heat = ctx.coal_heat
    tap_water = ctx.tap_water
    wastewater_treat = ctx.wastewater_treat
    psa_ref = ctx.psa_ref
    ng_heat_market = ctx.ng_heat_market
    biomethane_heat_ref = ctx.biomethane_heat_ref
    co2_air = ctx.co2_air
    co_air = ctx.co_air
    FOREGROUND_DB = ctx.FOREGROUND_DB
    htc_reactor = ctx.htc_reactor
    drying = ctx.drying
    heat_prod = ctx.heat_prod
    sub_heat_coal = ctx.sub_heat_coal
    ad_process = ctx.ad_process
    dewatering = ctx.dewatering
    biogas_upgrading = ctx.biogas_upgrading
    heat_biomethane = ctx.heat_biomethane
    heat_naturalgas = ctx.heat_naturalgas
    waste_treatment = ctx.waste_treatment
    if ISR_batch is None:
        ISR_batch = ctx.ISR_batch

    # ── CLEAR ALL FOREGROUND ACTIVITIES BEFORE REBUILDING ────────
    db = bd.Database(FOREGROUND_DB)
    for act in db:
        for exc in list(act.exchanges()):
            if exc["type"] != "production":
                exc.delete()
    print(f"🧹 Cleared all exchanges in '{FOREGROUND_DB}'")

    # ── Load ML module ──────────────────────────────────────────
    ml = _load_ml_module(ml_module)
    HTC_parametric_ML = ml.HTC_parametric_ML
    print(f"✅ Loaded ML module from: {getattr(ml, '__file__', ml_module)}")
    print(f"🔀 Running PATHWAY {PATHWAY}")
    print(f"⚙️  ADM1 mode: {adm1_mode}" +
          (f" | HRT={adm1_HRT_days} d" if adm1_mode == "continuous" and adm1_HRT_days else "") +
          (f" | q_ad={adm1_q_ad} m³/d" if adm1_mode == "continuous" and adm1_q_ad else ""))

    # =========================================================
    # STEP 1 — PATHWAY-SPECIFIC FIRST STEP
    # =========================================================

    if PATHWAY == "B":
        print("\n" + "=" * 60)
        print("  PATHWAY B — Step 1: AD on raw feedstock")
        print("=" * 60)

        m_feedstock_input = m_feedstock
        dry_fraction_feed = 1 - default_params["WC"] / 100
        ash_fraction_feed = default_params.get("Ash_feed", 0.0) / 100
        m_dry_feedstock = m_feedstock * dry_fraction_feed

        C_feed_mass = m_dry_feedstock * (default_params["C_feed"] / 100)
        H_feed_mass = m_dry_feedstock * (default_params["H_feed"] / 100)
        O_feed_mass = m_dry_feedstock * (default_params["O_feed"] / 100)
        N_feed_mass = m_dry_feedstock * (default_params["N_feed"] / 100)
        S_feed_mass = m_dry_feedstock * (default_params["S_feed"] / 100)
        ash_kg_feed = m_dry_feedstock * ash_fraction_feed

        feed_inputs = translate_feedstock_to_adm1(
            C_feed=C_feed_mass, H_feed=H_feed_mass, O_feed=O_feed_mass,
            N_feed=N_feed_mass, S_feed=S_feed_mass, ash_kg=ash_kg_feed,
            feedstock_wet_kg=m_feedstock, mode=adm1_mode,
            HRT_days=adm1_HRT_days, TS_target=TS_target,
        )

        m_feedstock_slurry = feed_inputs["_m_slurry_kg"]
        dilution_water_AD = feed_inputs["_dilution_water_kg"]
        m_dry_feedstock = feed_inputs["_m_dry_kg"]

        print(f"[Pathway B] Feedstock at gate:  {m_feedstock:.2f} kg")
        print(f"[Pathway B] Dilution water → AD: {dilution_water_AD:.2f} kg")
        print(f"[Pathway B] Total slurry to AD:  {m_feedstock_slurry:.2f} kg")

        if adm1_mode == "continuous":
            adm1_q_ad = feed_inputs["_V_feed_m3"]

            CH4_mass, CO2_mass, NH3_mass, adm1_out, adm1_out_blank = run_adm1_and_correct(
                pw_inputs=feed_inputs, adm1_mode=adm1_mode, adm1_q_ad=adm1_q_ad,
                adm1_HRT_days=adm1_HRT_days, adm1_t_sim_days=adm1_t_sim_days,
                T_HTC=T_HTC, RT_HTC=RT_HTC, disable_tox=True, label="B",
                ss_initial_state=adm1_ss_state,
            )
            adm1_out["_blank_out"] = None

            CH4_mass = adm1_out["CH4_mass_kg"] / adm1_HRT_days
            CO2_mass = adm1_out["CO2_mass_kg"] / adm1_HRT_days
            NH3_mass = adm1_out["NH3_mass_kg"] / adm1_HRT_days

        else:
            CH4_mass, CO2_mass, NH3_mass, adm1_out, adm1_out_blank = run_adm1_and_correct(
                pw_inputs=feed_inputs, adm1_mode=adm1_mode, adm1_q_ad=None,
                adm1_HRT_days=None, adm1_t_sim_days=adm1_t_sim_days,
                T_HTC=T_HTC, RT_HTC=RT_HTC, ISR_batch=ISR_batch,
                disable_tox=True, label="B", ss_initial_state=adm1_ss_state,
            )
            adm1_out["_blank_out"] = adm1_out_blank

            m_feedstock_slurry = feed_inputs["_m_slurry_kg"]
            m_dry_feedstock = feed_inputs["_m_dry_kg"]
            dilution_water_AD = feed_inputs["_dilution_water_kg"]

        print("\n" + "=" * 60)
        print("  PATHWAY B — Step 2: Extract digestate → HTC inputs")
        print("=" * 60)

        print(f"m_feedstock_slurry (per tonne) = {m_feedstock_slurry:.4f} kg")
        print(f"m_dry_feedstock    (per tonne) = {m_dry_feedstock:.4f} kg")
        print(f"CH4_mass           (per tonne) = {CH4_mass:.4f} kg")
        print(f"CO2_mass           (per tonne) = {CO2_mass:.4f} kg")

        if adm1_mode == "continuous":
            CH4_for_dig = adm1_out["CH4_mass_kg"] / adm1_HRT_days
            CO2_for_dig = adm1_out["CO2_mass_kg"] / adm1_HRT_days
            NH3_for_dig = adm1_out["NH3_mass_kg"] / adm1_HRT_days
        else:
            CH4_for_dig = CH4_mass
            CO2_for_dig = CO2_mass
            NH3_for_dig = NH3_mass

        digestate_params = extract_digestate_composition(
            CH4_mass=CH4_for_dig, CO2_mass=CO2_for_dig, NH3_mass=NH3_for_dig,
            C_feed=C_feed_mass, H_feed=H_feed_mass, O_feed=O_feed_mass,
            N_feed=N_feed_mass, S_feed=S_feed_mass,
            m_feedstock_wet=m_feedstock_slurry, m_feedstock_dry=m_dry_feedstock,
        )

        # ── DEWATERING ────────────────────────────────────────────
        WC_target_dewater = target_moisture_HTC
        WC_dig_raw = digestate_params["WC"] / 100

        m_digestate_total = m_feedstock_slurry - CH4_mass - CO2_mass
        dry_matter_dig = m_digestate_total * (1 - WC_dig_raw)

        if WC_dig_raw > WC_target_dewater:
            m_cake = dry_matter_dig / (1 - WC_target_dewater)
            m_centrate = m_digestate_total - m_cake

            e_centrifuge = 0.003
            E_centrifuge_kWh = m_digestate_total * e_centrifuge

            print(f"\n[Pathway B — Dewatering]")
            print(f"  Digestate in:   {m_digestate_total:.2f} kg  (WC={WC_dig_raw*100:.1f}%)")
            print(f"  Cake out:       {m_cake:.2f} kg  (WC={WC_target_dewater*100:.0f}%)")
            print(f"  Centrate out:   {m_centrate:.2f} kg  → WWT")
            print(f"  Centrifuge elec:{E_centrifuge_kWh:.3f} kWh")

            digestate_params["WC"] = WC_target_dewater * 100
            m_digestate_for_htc = m_cake
        else:
            m_centrate = 0.0
            E_centrifuge_kWh = 0.0
            m_cake = m_digestate_total
            m_digestate_for_htc = m_digestate_total
            print(f"[Pathway B] No dewatering needed — WC={WC_dig_raw*100:.1f}% ≤ {WC_target_dewater*100:.0f}%")

        htc_params_B = {**ml_params, **digestate_params}
        print(f"[HTC inputs B] T={htc_params_B['T']} °C, RT={htc_params_B['RT']} min, "
              f"WC={htc_params_B['WC']:.1f}%")

        print("\n" + "=" * 60)
        print("  PATHWAY B — Step 3: HTC ML on digestate")
        print("=" * 60)
        elem_sum = (htc_params_B["C_feed"] + htc_params_B["H_feed"] +
                    htc_params_B["O_feed"] + htc_params_B["N_feed"] +
                    htc_params_B["S_feed"] + htc_params_B["Ash_feed"])
        print(f"  Sum C+H+O+N+S+Ash: {elem_sum:.2f} % (should be ~100)")
        df, inv = run_ml_model(HTC_parametric_ML, params=htc_params_B)
    else:
        print("\n" + "=" * 60)
        print("  PATHWAY A — Step 1: HTC ML on raw feedstock")
        print("=" * 60)
        ml_params_htc = {**ml_params, "WC": target_moisture_HTC * 100}
        df, inv = run_ml_model(HTC_parametric_ML, params=ml_params_htc)

    print("\n[ML model outputs]", inv)

    Y_char_pct = inv.get("Y_char (%)", np.nan)
    HHV_char = inv.get("HHV_char (MJ/kg)", np.nan)

    # =========================================================
    # STEP 2 — MOISTURE CORRECTION & EFFECTIVE FEEDSTOCK MASS
    # =========================================================
    added_water = 0.0

    if PATHWAY == "B":
        m_feedstock_effective = m_digestate_for_htc
        WC_feed = digestate_params["WC"] / 100

        if WC_feed < target_moisture_HTC:
            dry_mass_digestate = m_feedstock_effective * (1 - WC_feed)
            m_new = dry_mass_digestate / (1 - target_moisture_HTC)
            added_water = max(0.0, m_new - m_feedstock_effective)
            m_feedstock_effective = m_new
            WC_feed = target_moisture_HTC
            print(f"[Pathway B] Added water: {added_water:.3f} kg "
                  f"(digestate too dry — adjusted to {target_moisture_HTC*100:.0f}%)")
        else:
            added_water = 0.0

        print(f"\n[Pathway B] m_feedstock_effective = {m_feedstock_effective:.4f} kg")
        print(f"[Pathway B] Digestate WC: {WC_feed*100:.1f}% (post water addition)")
        print(f"[Pathway B] Biogas (per HRT): CH4={CH4_mass:.4f} kg, "
              f"CO2={CO2_mass:.4f} kg, NH3={NH3_mass:.4f} kg")

    else:
        m_feedstock_effective = m_feedstock
        WC_feed = inv.get("WC_feed (%)", np.nan) / 100
        if WC_feed < target_moisture_HTC:
            added_water = (m_feedstock * (1 - WC_feed) - m_feedstock * (1 - target_moisture_HTC)) / (1 - target_moisture_HTC)
            added_water = max(0.0, added_water)
            m_feedstock_effective = m_feedstock + added_water
            WC_feed = target_moisture_HTC
            print(f"Added water: {added_water:.3f} kg (to reach {target_moisture_HTC*100:.0f}% moisture)")
        else:
            added_water = 0.0
            print(f"No water addition needed — WC={WC_feed*100:.1f}% ≥ target {target_moisture_HTC*100:.0f}%")

    # =========================================================
    # STEP 3 — HTC MASS BALANCE & ENERGY
    # =========================================================

    mb = compute_htc_mass_balance(
        m_feed_eff=m_feedstock_effective, WC_feed=WC_feed, Y_char_pct=Y_char_pct,
        T_HTC=T_HTC, MC_char=MC_char, target_MC_char_dry=target_MC_char_dry,
    )
    HC_wet_yield = mb["HC_wet_yield"]
    HC_dry_yield = mb["HC_dry_yield"]
    PW_yield = mb["PW_yield"]
    CO2_yield = mb["CO2_yield"]
    CO_yield = mb["CO_yield"]
    Q_heat_kWh = mb["Q_heat_kWh"]
    elec_MJ = mb["elec_MJ"]
    wastewater_m3 = mb["wastewater_m3"]

    print_mass_balance_diagnostic(m_feedstock_effective, mb, WC_feed)

    HHV_coal = 28.0
    coal_needed_1MJ = 0.04325
    SubstitutionFactor_HC_COAL = HHV_char / HHV_coal
    MJ_hydrochar = (HC_wet_yield - wastewater_m3 * 1000) / coal_needed_1MJ * SubstitutionFactor_HC_COAL

    print(f"Derived yields: HC={HC_wet_yield:.3f} kg, PW={PW_yield:.3f} kg, "
          f"CO2={CO2_yield:.3f} kg, CO={CO_yield:.3f} kg")
    print(f"Energy use: {Q_heat_kWh:.3f} kWh (HTC heating), {elec_MJ:.3f} MJ = "
          f"{elec_MJ/3.6:.3f} kWh (drying), {MJ_hydrochar:.3f} MJ (HC substitution)")

    dry_mass_in = m_feedstock_effective * (1 - WC_feed)
    inv["Y_gas (%)"] = mb["Y_gas_drybasis"]
    inv["Y_liquid (%)"] = PW_yield / dry_mass_in * 100

    # =========================================================
    # PATHWAY A — Per-tonne normalisation
    # =========================================================
    if PATHWAY == "A":
        _scale_A = 1000.0 / m_feedstock

        m_feedstock_effective = m_feedstock_effective * _scale_A
        added_water = added_water * _scale_A
        HC_wet_yield = HC_wet_yield * _scale_A
        HC_dry_yield = HC_dry_yield * _scale_A
        PW_yield = PW_yield * _scale_A
        CO2_yield = CO2_yield * _scale_A
        CO_yield = CO_yield * _scale_A
        Q_heat_kWh = Q_heat_kWh * _scale_A
        elec_MJ = elec_MJ * _scale_A
        wastewater_m3 = wastewater_m3 * _scale_A
        MJ_hydrochar = MJ_hydrochar * _scale_A

        print(f"[per-tonne scaling A] scale={_scale_A:.5f}  FU=1 tonne wet feed")
        print(f"[per-tonne scaling A] PW_yield={PW_yield:.3f} kg/t  "
              f"HC_wet={HC_wet_yield:.3f} kg/t  Q_heat={Q_heat_kWh:.3f} kWh/t")

    # =========================================================
    # STEP 4 — AD ON PROCESS WATER (Pathway A only)
    # =========================================================

    if PATHWAY == "A":
        print("\n" + "=" * 60)
        print("  PATHWAY A — Step 2: AD on HTC process water")
        print("=" * 60)

        C_char = inv.get("C_char (%)", np.nan)
        H_char = inv.get("H_char (%)", np.nan)
        O_char = inv.get("O_char (%)", np.nan)
        N_char = inv.get("N_char (%)", np.nan)
        S_char = inv.get("S_char (%)", np.nan)

        dry_fraction_A = 1 - WC_feed
        _m_feed_actual = m_feedstock

        C_feed_mass_A = _m_feed_actual * dry_fraction_A * (default_params["C_feed"] / 100)
        H_feed_mass_A = _m_feed_actual * dry_fraction_A * (default_params["H_feed"] / 100)
        O_feed_mass_A = _m_feed_actual * dry_fraction_A * (default_params["O_feed"] / 100)
        N_feed_mass_A = _m_feed_actual * dry_fraction_A * (default_params["N_feed"] / 100)
        S_feed_mass_A = _m_feed_actual * dry_fraction_A * (default_params["S_feed"] / 100)

        Y_char_dry = Y_char_pct / 100
        C_char_mass = _m_feed_actual * dry_fraction_A * Y_char_dry * (C_char / 100)
        H_char_mass = _m_feed_actual * dry_fraction_A * Y_char_dry * (H_char / 100)
        O_char_mass = _m_feed_actual * dry_fraction_A * Y_char_dry * (O_char / 100)
        N_char_mass = _m_feed_actual * dry_fraction_A * Y_char_dry * (N_char / 100)
        S_char_mass = _m_feed_actual * dry_fraction_A * Y_char_dry * (S_char / 100)

        C_gas_mass = ((CO2_yield / _scale_A) / (MW_C + 2 * MW_O) +
                      (CO_yield / _scale_A) / (MW_C + MW_O)) * MW_C
        O_gas_mass = ((CO2_yield / _scale_A) * (2 * MW_O) / (MW_C + 2 * MW_O) +
                      (CO_yield / _scale_A) * MW_O / (MW_C + MW_O))

        C_pw_mass = max(0, C_feed_mass_A - (C_char_mass + C_gas_mass))
        H_pw_mass = max(0, H_feed_mass_A - H_char_mass)
        O_pw_mass = max(0, O_feed_mass_A - (O_char_mass + O_gas_mass))
        N_pw_mass = max(0, N_feed_mass_A - N_char_mass)
        S_pw_mass = max(0, S_feed_mass_A - S_char_mass)

        rho_PW = 1.02
        pw_inputs = translate_pw_to_adm1(
            C_pw=C_pw_mass, H_pw=H_pw_mass, O_pw=O_pw_mass,
            N_pw=N_pw_mass, S_pw=S_pw_mass,
            PW_yield_L=(PW_yield / _scale_A) / rho_PW,
            mode=adm1_mode, HRT_days=adm1_HRT_days, T_HTC=T_HTC,
            ISR_batch=ISR_batch, COD_conc_target=COD_conc_target,
        )

        if adm1_mode == "continuous":
            adm1_q_ad = pw_inputs["_V_feed_m3"]

        CH4_mass, CO2_mass, NH3_mass, adm1_out, adm1_out_blank = run_adm1_and_correct(
            pw_inputs=pw_inputs, adm1_mode=adm1_mode, adm1_q_ad=adm1_q_ad,
            adm1_HRT_days=adm1_HRT_days, adm1_t_sim_days=adm1_t_sim_days,
            T_HTC=T_HTC, RT_HTC=RT_HTC, ISR_batch=ISR_batch,
            disable_tox=True, label="A", ss_initial_state=adm1_ss_state,
        )
        adm1_out["_blank_out"] = adm1_out_blank

        if adm1_mode == "continuous":
            _actual_HRT = adm1_out.get("HRT_days", adm1_HRT_days)
            _gas_scale_A = _scale_A / _actual_HRT
            CH4_mass = adm1_out["CH4_mass_kg"] * _gas_scale_A
            CO2_mass = adm1_out["CO2_mass_kg"] * _gas_scale_A
            NH3_mass = adm1_out["NH3_mass_kg"] * _gas_scale_A
            print(f"[Pathway A gas scaling] actual HRT={_actual_HRT:.2f} d  "
                  f"scale={_gas_scale_A:.5f}  "
                  f"CH4={CH4_mass:.4f} kg/t  CO2={CO2_mass:.4f} kg/t")
        else:
            CH4_mass = adm1_out["CH4_mass_kg"] * _scale_A
            CO2_mass = adm1_out["CO2_mass_kg"] * _scale_A
            NH3_mass = adm1_out["NH3_mass_kg"] * _scale_A

    # =========================================================
    # STEP 5 — COMPUTE AD REFERENCE MASS & DIGESTATE MASS
    # =========================================================

    if PATHWAY == "A":
        dilution_water_PW_kg = pw_inputs.get("_dilution_water_m3", 0.0) * 1000.0 * _scale_A
        m_AD_in = PW_yield + dilution_water_PW_kg
        digestate_mass = max(0, PW_yield + dilution_water_PW_kg - CH4_mass - CO2_mass)
    else:
        m_AD_in = m_feedstock_slurry
        digestate_mass = max(0, m_feedstock_slurry - CH4_mass - CO2_mass)

    # =========================================================
    # STEP 6 — BIOGAS UPGRADING (PSA) & BIOMETHANE
    # =========================================================

    PSA_CH4_recovery = 0.97
    LHV_biomethane = 35
    LHV_NG = 35

    CH4_m3 = CH4_mass / CH4_density
    biomethane_m3 = CH4_m3 * PSA_CH4_recovery
    biogas_m3 = (CH4_mass / CH4_density) + (CO2_mass / CO2_density)

    MJ_biomethane = biomethane_m3 * LHV_biomethane
    scaling_factor = MJ_biomethane / LHV_NG

    print(f"Biogas: {biogas_m3:.4f} m³ → biomethane: {biomethane_m3:.4f} m³")
    print(f"Biomethane energy: {MJ_biomethane:.4f} MJ → NG displaced: {scaling_factor:.4f} m³NG")

    # =========================================================
    # STEP 7 — WRITE LCA EXCHANGES
    # =========================================================

    CH4_fug, NH3_fug, Q_heat_AD_MJ, E_elec_AD_kWh = build_ad_exchanges(
        ad_process=ad_process, bios=bios, ei=ei,
        CH4_mass=CH4_mass, NH3_mass=NH3_mass, m_AD_in=m_AD_in,
        adm1_mode=adm1_mode, PATHWAY=PATHWAY,
        ad_elec=ad_elec, ad_heat=ad_heat,
        wastewater_treat=wastewater_treat if PATHWAY == "A" else None,
        digestate_mass=digestate_mass if PATHWAY == "A" else None,
        dilution_water_m3=(pw_inputs.get("_dilution_water_m3", 0.0) * _scale_A if PATHWAY == "A"
                            else feed_inputs.get("_dilution_water_kg", 0.0) / 1000.0),
        tap_water=tap_water,
    )

    if PATHWAY == "B":
        build_dewatering_exchanges(
            dewatering=dewatering, elec=elec, wastewater_treat=wastewater_treat,
            E_centrifuge_kWh=E_centrifuge_kWh, centrate_m3=m_centrate / 1000.0,
            ad_process=ad_process, m_digestate_total=m_digestate_total,
        )

    build_htc_reactor_exchanges(
        htc_reactor=htc_reactor, elec=elec, tap_water=tap_water,
        co2_air=co2_air, co_air=co_air, wastewater_treat=wastewater_treat,
        Q_heat_kWh=Q_heat_kWh, CO2_yield=CO2_yield, CO_yield=CO_yield,
        added_water=added_water, PW_yield=PW_yield, PATHWAY=PATHWAY,
        ad_process=None,
        digestate_mass=digestate_mass if PATHWAY == "A" else None,
        dewatering_activity=dewatering if PATHWAY == "B" else None,
        m_digestate_for_htc=m_digestate_for_htc if PATHWAY == "B" else None,
    )

    build_drying_exchanges(drying, elec, drying_heat, wastewater_treat, elec_MJ, wastewater_m3)
    build_coal_substitution_exchange(sub_heat_coal, coal_heat)

    hc_heat = heat_prod
    for exc in list(hc_heat.exchanges()):
        if exc["type"] != "production":
            exc.delete()
    build_hydrochar_heat_exchanges(
        hc_heat, coal_heat, bios,
        S_char_pct=inv["S_char (%)"], N_char_pct=inv["N_char (%)"],
        Ash_char_pct=inv["Ash_char (%)"], HHV_char=inv["HHV_char (MJ/kg)"],
    )

    for exc in list(biogas_upgrading.exchanges()):
        if exc["type"] != "production":
            exc.delete()
    build_biogas_upgrading_exchanges(biogas_upgrading, psa_ref, ei, biomethane_m3)

    for exc in list(heat_biomethane.exchanges()):
        if exc["type"] != "production":
            exc.delete()
    for exc in biomethane_heat_ref.exchanges():
        if exc["type"] == "biosphere":
            heat_biomethane.new_exchange(input=exc.input, amount=exc["amount"],
                                          type="biosphere", unit=exc["unit"]).save()
        elif exc["type"] == "technosphere":
            heat_biomethane.new_exchange(input=exc.input, amount=0,
                                          type="technosphere", unit=exc["unit"]).save()

    heat_naturalgas.new_exchange(input=ng_heat_market, amount=1,
                                  type="technosphere", unit="megajoule").save()

    waste_treatment.new_exchange(input=htc_reactor, amount=1, type="technosphere", unit="unit").save()
    waste_treatment.new_exchange(input=drying, amount=1, type="technosphere", unit="unit").save()
    waste_treatment.new_exchange(input=heat_prod, amount=MJ_hydrochar, type="technosphere", unit="megajoule").save()
    waste_treatment.new_exchange(input=sub_heat_coal, amount=-MJ_hydrochar, type="technosphere", unit="megajoule").save()
    waste_treatment.new_exchange(input=ad_process, amount=1, type="technosphere", unit="unit").save()
    if PATHWAY == "B":
        waste_treatment.new_exchange(input=dewatering, amount=1, type="technosphere", unit="unit").save()
    waste_treatment.new_exchange(input=biogas_upgrading, amount=1, type="technosphere", unit="unit").save()
    waste_treatment.new_exchange(input=heat_biomethane, amount=MJ_biomethane, type="technosphere", unit="megajoule").save()
    waste_treatment.new_exchange(input=heat_naturalgas, amount=-MJ_biomethane, type="technosphere", unit="megajoule").save()

    print(f"Natural gas substitution: -{MJ_biomethane:.4f} MJ")
    print(f"Heat from biomethane:      {MJ_biomethane:.4f} MJ")

    # =========================================================
    # STEP 8 — REACTOR MASS BALANCES (QA)
    # =========================================================

    if PATHWAY == "A":
        m_in_HTC_A = m_feedstock + added_water
        m_out_HTC_A = HC_wet_yield + CO2_yield + CO_yield + PW_yield
        MB_HTC_A = m_in_HTC_A - m_out_HTC_A

        m_in_AD_A = PW_yield + dilution_water_PW_kg
        m_out_AD_A = CH4_mass + CO2_mass + digestate_mass
        MB_AD_A = m_in_AD_A - m_out_AD_A

        waste_treatment["MB_HTC_A"] = float(MB_HTC_A)
        waste_treatment["MB_AD_A"] = float(MB_AD_A)
    else:
        m_in_AD_B = m_feedstock_input + dilution_water_AD
        digestate_leaving_AD = m_feedstock_slurry - CH4_mass - CO2_mass
        m_out_AD_B = CH4_mass + CO2_mass + digestate_leaving_AD
        MB_AD_B = m_in_AD_B - m_out_AD_B

        m_in_DW_B = m_digestate_total
        m_out_DW_B = m_cake + m_centrate
        MB_DW_B = m_in_DW_B - m_out_DW_B

        m_in_HTC_B = m_feedstock_effective
        m_out_HTC_B = HC_wet_yield + CO2_yield + CO_yield + PW_yield
        MB_HTC_B = m_in_HTC_B - m_out_HTC_B

        waste_treatment["MB_AD_B"] = float(MB_AD_B)
        waste_treatment["MB_DW_B"] = float(MB_DW_B)
        waste_treatment["MB_HTC_B"] = float(MB_HTC_B)

    print("\n========== REACTOR MASS BALANCES ==========")
    print(f"ADM1 mode: {adm1_mode}")
    if PATHWAY == "A":
        print(f"HTC MB (A): {MB_HTC_A:.6f} kg")
        print(f"AD  MB (A): {MB_AD_A:.6f} kg")
    else:
        print(f"AD  MB (B): {MB_AD_B:.6f} kg")
        print(f"DW  MB (B): {MB_DW_B:.6f} kg")
        print(f"HTC MB (B): {MB_HTC_B:.6f} kg")

    # =========================================================
    # STEP 9 — STORE METADATA
    # =========================================================

    waste_treatment["PATHWAY"] = PATHWAY
    waste_treatment["adm1_mode"] = adm1_mode
    _HRT_stored = float(adm1_out.get("HRT_days", 0.0)) if adm1_mode == "continuous" else 0.0
    waste_treatment["adm1_HRT_days"] = _HRT_stored
    waste_treatment["SubstitutionFactor_HC_COAL"] = SubstitutionFactor_HC_COAL
    waste_treatment["MJ_hydrochar"] = MJ_hydrochar
    waste_treatment["m_feedstock"] = m_feedstock_input if PATHWAY == "B" else m_feedstock
    waste_treatment["m_feedstock_effective"] = float(m_feedstock_effective)
    waste_treatment["HC_wet_yield"] = float(HC_wet_yield)
    waste_treatment["HC_dry_yield"] = float(HC_dry_yield)
    waste_treatment["MC_char"] = float(MC_char)
    waste_treatment["target_MC_char_dry"] = float(target_MC_char_dry)
    waste_treatment["PW_yield"] = float(PW_yield)
    waste_treatment["CO2_yield_HTC"] = float(CO2_yield)
    waste_treatment["CO_yield_HTC"] = float(CO_yield)
    waste_treatment["CH4_mass_AD"] = float(CH4_mass)
    waste_treatment["CO2_mass_AD"] = float(CO2_mass)
    waste_treatment["NH3_mass_AD"] = float(NH3_mass)
    waste_treatment["biogas_m3"] = float(biogas_m3)
    waste_treatment["biomethane_m3"] = float(biomethane_m3)
    waste_treatment["scaling_MJ_biomethane"] = float(scaling_factor)
    waste_treatment["wastewater_drying_m3"] = float(wastewater_m3)
    waste_treatment["added_water_HTC"] = float(added_water)
    waste_treatment["m_centrate_dewater_kg"] = float(m_centrate) if PATHWAY == "B" else 0.0
    waste_treatment["m_cake_dewater_kg"] = float(m_cake) if PATHWAY == "B" else 0.0
    waste_treatment["E_centrifuge_kWh"] = float(E_centrifuge_kWh) if PATHWAY == "B" else 0.0
    waste_treatment["dilution_water_AD_kg"] = float(dilution_water_AD) if PATHWAY == "B" else 0.0
    waste_treatment["dilution_water_PW_m3"] = float(pw_inputs.get("_dilution_water_m3", 0.0) * _scale_A) if PATHWAY == "A" else 0.0
    waste_treatment["location_country"] = ctx.location_country
    waste_treatment["location_province"] = ctx.location_province
    waste_treatment["location_code"] = ctx.location_code

    waste_treatment.save()

    print(f"\n✅ System '{FOREGROUND_DB}' built — PATHWAY {PATHWAY} "
          f"— model '{ml_module}' — ADM1 mode '{adm1_mode}'")

    if PATHWAY == "A":
        feed_inputs = pw_inputs

    return waste_treatment, adm1_out.get("ss_state", None), adm1_out, feed_inputs, inv