"""
system_builder.py
=================
HTC-AD system model of HADES: mass balances, ADM1 coupling, heat
integration and Brightway foreground inventory. The case study notebook and
the Streamlit app both call build_system() and spinup_ss_state() from this
module.

ADM1.py and plot_adm1_batch.py are loaded by file path (not `import ADM1`),
because the folder and the module share the name ADM1.

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
import re as _re
import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import bw2data as bd


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ADM1.py AND plot_adm1_batch.py BY FILE PATH
# ─────────────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_by_path(name, relpath):
    """Load a module from a path relative to this folder."""
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
plot_adm1_results = _plot_mod.plot_adm1_results

# Companion modules, loaded by path under p1_* names to avoid name clashes in
# the same Python session.
hi = _load_by_path("p1_heat_integration", "heat_integration.py")
hce = _load_by_path("p1_hydrochar_enduse", "hydrochar_enduse.py")
ac = _load_by_path("p1_ad_characterization", "ad_characterization.py")
bwl = _load_by_path("p1_bw_lookup", "bw_lookup.py")
get_flow = bwl.get_flow
relocate_flow = bwl.relocate_flow
_template_cats = bwl._template_cats
CORE_FLOW_LOCATIONS = bwl.CORE_FLOW_LOCATIONS


def _load_by_exact_path(name, filepath):
    """
    Like _load_by_path, but filepath is used as given (absolute, or relative
    to the current working directory). Used for the ML module, which lives in
    a sibling folder.
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
      - a plain module name already on sys.path, e.g. "htc_surrogate"
      - a filesystem path ending in .py, e.g. "ML training scripts and
        models/htc_surrogate.py" (spaces are fine — loaded by path, not by
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
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MW_C = 12.011     # standard atomic weights (IUPAC), g/mol
MW_H = 1.008
MW_O = 15.999
MW_N = 14.007
MW_S = 32.06
MW_CO2 = MW_C + 2 * MW_O
MW_CO = MW_C + MW_O

CH4_density = 0.716
CO2_density = 1.977

# BSM2 steady-state initial condition (Rosen and Jeppsson, 2006)
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
# SYSTEM CONTEXT: all inputs of build_system(), built once per pathway and
# location.
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
    default_params: dict          # the feedstock (FEED_KEYS), single source
    ml_params: Optional[dict]     # None: built by check_inputs from default_params, T_HTC, RT_HTC
    T_HTC: float                  # single source of the HTC temperature (C)
    RT_HTC: float                 # single source of the HTC residence time (min)
    target_moisture_HTC: float
    MC_char: float
    target_MC_char_dry: float
    COD_conc_target: float = 10.0
    ISR_batch: float = 2.0

    # optional metadata (Streamlit export)
    location_country: str = ""
    location_province: str = ""
    location_code: str = ""
    # dataset location priority for background inputs, e.g. ["CA-QC", "CA", "RoW", "GLO"]
    # (None: location_code, then CA, RoW, GLO)
    location_priority: Optional[list] = None

    # digestate characterization, heat integration and hydrochar end use
    feedstock_name: str = "food_waste"   # ad_characterization key (Pathway B digestate)
    eps_HR: Optional[float] = None       # HTC feed preheat effectiveness (None -> hi.EPS_HR_CENTRAL)
    nox_conv: float = 0.10               # fuel N -> NOx, hydrochar boiler (hce.NOX_CONV_CENTRAL)
    qc_heat_mix: Any = None              # foreground activity: Quebec industrial combustion heat mix
    heat_mix_components_qc: Any = None   # bwl.resolve_heat_mix(ei, hce.HEAT_MIX_DATASETS, ...)
    ash_disposal: Any = None             # ash treatment of the boiler ash (None -> omitted, printed)


# ─────────────────────────────────────────────────────────────────────────────
# INPUT CHECKS AND FEEDSTOCK HELPERS
# build_system() calls check_inputs() before any calculation. An input outside
# the range where a submodel is defined, or two copies of the same input that
# disagree, raise ValueError instead of running a clipped or inconsistent case.
# Statistical extrapolation of the HTC surrogate is not an error: it stays a
# QA flag (qa_checks).
# ─────────────────────────────────────────────────────────────────────────────

# HTC temperature range of the PW-ADM1 S_I composition table (ADM1.SI_TEMPS);
# outside it get_si_composition() and f_slow would be clipped silently, and
# the HTC gas yield correlation reaches 0 at 151.6 C.
T_HTC_RANGE_C = (float(_adm1_core.SI_TEMPS[0]), float(_adm1_core.SI_TEMPS[-1]))

FEED_ELEMENT_KEYS = ("C_feed", "H_feed", "N_feed", "S_feed", "Ash_feed")
FEED_KEYS = FEED_ELEMENT_KEYS + ("O_feed", "VM_feed", "FC_feed", "WC")
FEED_O_TOL = 0.01         # O_feed vs 100 - C - H - N - S - ash (% points)
FEED_PROXIMATE_TOL = 1.0  # VM + FC + ash vs 100 (% points)


def close_feed(feed, tol=FEED_O_TOL):
    """
    Copy of `feed` with O_feed = 100 - C - H - N - S - ash (O by difference,
    % dry basis). If `feed` already has an O_feed that differs from the
    difference by more than `tol` % points, ValueError: the composition was
    changed without updating O.
    """
    f = dict(feed)
    O = 100.0 - sum(float(f[k]) for k in FEED_ELEMENT_KEYS)
    if O <= 0:
        raise ValueError(f"[feed] C + H + N + S + ash = {100 - O:.2f} % of dry matter: "
                         f"no oxygen left by difference")
    if "O_feed" in f and abs(float(f["O_feed"]) - O) > tol:
        raise ValueError(f"[feed] O_feed = {f['O_feed']:.3f} % but 100 - C - H - N - S - ash "
                         f"= {O:.3f} %: update O when the composition changes (close_feed)")
    f["O_feed"] = O
    return f


def perturb_feed(feed, key, factor, balance="O"):
    """
    Multiply one component of `feed` (C_feed, H_feed, N_feed, S_feed or
    Ash_feed) by `factor` and say explicitly what compensates, so that the
    dry composition still sums to 100 %:
      balance="O"    O by difference absorbs the change (Paper 1 OAT). C x 1.2
                     on the base food waste lowers O by 7.65 % points.
      balance="daf"  the other organic elements are rescaled proportionally
                     (element change), or all organic elements are rescaled
                     with the ash-free composition kept (ash change).
    For an ash change, VM and FC are rescaled so that VM + FC + ash = 100.
    Returns the closed feed (close_feed).
    """
    if key not in FEED_ELEMENT_KEYS:
        raise ValueError(f"[perturb_feed] key {key!r} not in {FEED_ELEMENT_KEYS}")
    if balance not in ("O", "daf"):
        raise ValueError(f"[perturb_feed] balance {balance!r} must be 'O' or 'daf'")
    f = close_feed(feed)
    old, new = float(f[key]), float(f[key]) * float(factor)
    organics = ("C_feed", "H_feed", "O_feed", "N_feed", "S_feed")
    if key == "Ash_feed":
        k_pr = (100.0 - new) / (f["VM_feed"] + f["FC_feed"])
        f["VM_feed"], f["FC_feed"] = f["VM_feed"] * k_pr, f["FC_feed"] * k_pr
        if balance == "daf":
            k = (100.0 - new) / (100.0 - old)
            for e in organics:
                f[e] *= k
    elif balance == "daf":
        others = [e for e in organics if e != key]
        rest = sum(f[e] for e in others)
        k = (rest - (new - old)) / rest
        if k <= 0:
            raise ValueError(f"[perturb_feed] {key} x {factor}: no organic matter left to rescale")
        for e in others:
            f[e] *= k
    f[key] = new
    f.pop("O_feed")
    return close_feed(f)


def _check_feed(feed, label):
    missing = [k for k in FEED_KEYS if k not in feed]
    if missing:
        raise ValueError(f"[check_inputs | {label}] missing {missing}"
                         + (" (use close_feed to add O by difference)" if missing == ["O_feed"] else ""))
    for k in FEED_KEYS:
        v = float(feed[k])
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"[check_inputs | {label}] {k} = {feed[k]}")
    if not 0.0 < float(feed["WC"]) < 100.0:
        raise ValueError(f"[check_inputs | {label}] WC = {feed['WC']} % outside (0, 100)")
    close_feed(feed)    # O > 0 and O_feed consistent with the other elements
    prox = float(feed["VM_feed"]) + float(feed["FC_feed"]) + float(feed["Ash_feed"])
    if abs(prox - 100.0) > FEED_PROXIMATE_TOL:
        raise ValueError(f"[check_inputs | {label}] VM + FC + ash = {prox:.2f} % "
                         f"(tolerance {FEED_PROXIMATE_TOL} % points)")


def check_inputs(ctx, adm1_mode="batch", adm1_HRT_days=None, TS_target=0.08):
    """
    Validate the inputs of build_system() and return the HTC surrogate
    parameters.

    Single source: the feedstock is ctx.default_params, T and RT are
    ctx.T_HTC and ctx.RT_HTC. ctx.ml_params may be None, or carry copies of
    these values for backward compatibility; a copy that disagrees with the
    single source raises ValueError instead of feeding the surrogate a
    different case than the rest of the model.
    """
    feed = ctx.default_params
    _check_feed(feed, "default_params")
    T, RT = float(ctx.T_HTC), float(ctx.RT_HTC)
    lo, hi_ = T_HTC_RANGE_C
    if not lo <= T <= hi_:
        raise ValueError(f"[check_inputs] T_HTC = {T} C outside {lo:.0f}-{hi_:.0f} C, the range of "
                         f"the PW-ADM1 S_I composition (values would be clipped silently)")
    if not np.isfinite(RT) or RT <= 0:
        raise ValueError(f"[check_inputs] RT_HTC = {ctx.RT_HTC} min must be > 0")

    ml = dict(ctx.ml_params or {})
    for k in FEED_KEYS:
        if k in ml and abs(float(ml[k]) - float(feed[k])) > 1e-9:
            raise ValueError(f"[check_inputs] ml_params[{k!r}] = {ml[k]} but default_params[{k!r}] = "
                             f"{feed[k]}: one feedstock per run (leave ml_params None)")
    for k, v, name in (("T", T, "T_HTC"), ("RT", RT, "RT_HTC")):
        if k in ml and abs(float(ml[k]) - v) > 1e-9:
            raise ValueError(f"[check_inputs] ml_params[{k!r}] = {ml[k]} but ctx.{name} = {v}: "
                             f"set ctx.{name} only (leave ml_params None)")
    ml.update({k: feed[k] for k in FEED_KEYS})
    ml["T"], ml["RT"] = T, RT

    for name, v in (("target_moisture_HTC", ctx.target_moisture_HTC), ("TS_target", TS_target)):
        if not 0.0 < float(v) < 1.0:
            raise ValueError(f"[check_inputs] {name} = {v} outside (0, 1)")
    mc_cake = hi.effective_MC_char(ctx.MC_char)
    if not 0.0 <= float(ctx.target_MC_char_dry) < mc_cake:
        raise ValueError(f"[check_inputs] target_MC_char_dry = {ctx.target_MC_char_dry} must be in "
                         f"[0, MC_char = {mc_cake})")
    if not float(ctx.m_feedstock) > 0:
        raise ValueError(f"[check_inputs] m_feedstock = {ctx.m_feedstock} kg must be > 0")
    if not float(ctx.COD_conc_target) > 0:
        raise ValueError(f"[check_inputs] COD_conc_target = {ctx.COD_conc_target} must be > 0")
    if adm1_mode == "continuous" and not (adm1_HRT_days is not None and float(adm1_HRT_days) > 0):
        raise ValueError(f"[check_inputs] continuous ADM1 needs adm1_HRT_days > 0, got {adm1_HRT_days}")
    if ctx.PATHWAY not in ("A", "B"):
        raise ValueError(f"[check_inputs] PATHWAY = {ctx.PATHWAY!r} must be 'A' or 'B'")
    if ctx.PATHWAY == "B" and ctx.feedstock_name not in ac.AD_CHARACTERIZATION:
        raise ValueError(f"[check_inputs] feedstock_name = {ctx.feedstock_name!r} not in "
                         f"{sorted(ac.AD_CHARACTERIZATION)}")
    return ml


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
    """Call the HTC surrogate with the parameters it accepts.

    Also computes the carbon recovery in the hydrochar,
    Y_char x C_char / C_feed (C_recovery_raw). Char yield and char C come
    from separate models, so a value above 1 would break the carbon balance;
    it is flagged by the QA checks. Returns (dataframe or None, inventory dict).
    """
    p = params if params is not None else {}
    valid_args = inspect.signature(HTC_parametric_ML).parameters.keys()
    usable_args = {k: v for k, v in p.items() if k in valid_args}
    result = HTC_parametric_ML(**usable_args)
    df, inv = result if isinstance(result, tuple) and len(result) == 2 else (None, result)
    try:
        inv["C_recovery_raw"] = (inv["Y_char (%)"] / 100.0) * inv["C_char (%)"] / p["C_feed"]
    except (KeyError, TypeError, ZeroDivisionError):
        pass
    return df, inv


# ─────────────────────────────────────────────────────────────────────────────
# HTC MASS BALANCE
# ─────────────────────────────────────────────────────────────────────────────

def compute_htc_mass_balance(m_feed_eff, WC_feed, Y_char_pct, T_HTC,
                             MC_char, target_MC_char_dry, eps_HR=None,
                             T_feed_C=None):
    # T_feed_C: temperature of the HTC feed entering the preheater (None:
    # heat_integration.T_AMB_C, raw feedstock in Pathway A; Pathway B passes
    # the temperature of the wet solids leaving the digestate centrifuge).
    # MC_char: moisture of the hydrochar cake leaving the filter press.
    MC_char = hi.effective_MC_char(MC_char)
    dry_fraction = 1.0 - WC_feed
    Y_char_dry = Y_char_pct / 100.0

    # HTC gas: predominantly CO2 (Berge et al., 2011); 95/5 CO2/CO molar
    # split assumed.
    CO2_frac_mol = 0.95
    CO_frac_mol = 0.05

    dry_mass_in = m_feed_eff * dry_fraction
    water_mass_in = m_feed_eff * WC_feed

    # 1. Solid hydrochar produced (dry mass)
    HC_dry_mass = dry_mass_in * Y_char_dry

    # 2. Water retained in the wet cake (limited by the water available)
    HC_water_retained = min(HC_dry_mass * (MC_char / (1.0 - MC_char)),
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

    # 3. Gas formation and process water (PW) yield
    Gas_yield_dry = max(0.0, 0.121 * T_HTC - 18.34) / 100.0
    Gas_yield_total = m_feed_eff * dry_fraction * Gas_yield_dry
    gas_denom = CO2_frac_mol * MW_CO2 + CO_frac_mol * MW_CO
    CO2_yield = Gas_yield_total * (CO2_frac_mol * MW_CO2) / gas_denom
    CO_yield = Gas_yield_total * (CO_frac_mol * MW_CO) / gas_denom
    PW_yield = m_feed_eff - (HC_wet_yield + CO2_yield + CO_yield)

    # 4. Dried product at the target moisture (dry char mass conserved)
    HC_dry_yield = HC_dry_mass / (1.0 - target_MC_char_dry)

    # 5-6. HTC heating with feed preheat, then dryer heat and electricity
    #      (heat_integration.py).
    water_evaporated_kg = HC_wet_yield - HC_dry_yield
    heat_htc = hi.htc_duty(m_slurry_kg=m_feed_eff, m_dry_kg=dry_mass_in,
                           T_HTC_C=T_HTC, eps_HR=eps_HR,
                           T_amb_C=T_feed_C)
    heat_dry = hi.dryer_duty(water_evaporated_kg, m_prod_kg=m_feed_eff,
                             T_prod1_C=heat_htc["T_prod1_C"],
                             m_dry_char_kg=HC_dry_mass)
    Q_heat_kWh = heat_htc["Q_HTC_util_kWh"]
    elec_MJ = heat_dry["Q_dry_util_MJ"]       # dryer heat supplied by the utility (MJ)
    elec_kWh = elec_MJ / 3.6
    dryer_fan_kWh = heat_dry["E_dry_elec_kWh"]   # dryer fans + filter press
    wastewater_m3 = heat_dry["condensate_m3"]
    # product cooler E2: product cooled to T_SEP_C before the letdown and the
    # filter press; the use of this heat is decided in build_system
    heat_E2 = hi.product_cooler_duty(m_feed_eff, heat_dry["T_prod2_C"])

    return dict(
        HC_dry_mass=HC_dry_mass, HC_wet_yield=HC_wet_yield, HC_dry_yield=HC_dry_yield,
        Gas_yield_total=Gas_yield_total, CO2_yield=CO2_yield, CO_yield=CO_yield,
        PW_yield=PW_yield, wastewater_m3=wastewater_m3,
        Q_heat_kWh=Q_heat_kWh, elec_MJ=elec_MJ, elec_kWh=elec_kWh,
        Y_gas_drybasis=Gas_yield_total / dry_mass_in * 100.0,
        dryer_fan_kWh=dryer_fan_kWh, water_evaporated_kg=water_evaporated_kg,
        T_prod1_C=heat_htc["T_prod1_C"], T_prod2_C=heat_dry["T_prod2_C"],
        heat_htc=heat_htc, heat_dry=heat_dry,
        heat_E2=heat_E2, T_SEP_C=heat_E2["T_SEP_C"],
        MC_char_used=MC_char,
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
# ADM1 RUNNER
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
# BRIGHTWAY EXCHANGE BUILDERS
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
        htc_reactor.new_exchange(input=wastewater_treat, amount=_wastewater_amount(wastewater_treat, pw_wastewater_m3, "HTC PW (B)"),
                                  type="technosphere", unit="cubic meter").save()
        print(f"[Pathway B] Dewatered cake -> HTC reactor: {m_digestate_for_htc:.4f} kg")
        print(f"[Pathway B] HTC PW -> wastewater: {pw_wastewater_m3:.5f} m³")


def build_drying_exchanges(drying, elec, drying_heat, wastewater_treat, elec_MJ, wastewater_m3,
                           fan_kWh=0.0):
    """
    elec_MJ: dryer heat supplied by the utility (MJ), from hi.dryer_duty().
    If drying_heat is an electricity market (kWh), the heat is direct
    electric air heating (1 kWh = 3.6 MJ). fan_kWh: dryer fan, belt drive and
    filter press electricity. Wastewater is written only for a condensing
    dryer.
    """
    is_kwh = drying_heat.get("unit", "") == "kilowatt hour"
    if elec_MJ > 0:
        drying.new_exchange(input=drying_heat, amount=elec_MJ / 3.6 if is_kwh else elec_MJ,
                             type="technosphere", unit="kilowatt hour" if is_kwh else "megajoule").save()
    if fan_kWh > 0:
        drying.new_exchange(input=elec, amount=fan_kWh,
                             type="technosphere", unit="kilowatt hour").save()
    if wastewater_m3 > 0:
        drying.new_exchange(input=wastewater_treat, amount=_wastewater_amount(wastewater_treat, wastewater_m3, "dryer condensate"),
                             type="technosphere", unit="cubic meter").save()



def build_ad_exchanges(ad_process, bios, ei, CH4_mass, NH3_mass, m_AD_in,
                        adm1_mode, PATHWAY, ad_elec, ad_heat,
                        wastewater_treat=None, digestate_mass=None,
                        dilution_water_m3=0.0, tap_water=None,
                        T_AD_in_C=None, Q_rec_avail_MJ=0.0):
    f_fug_CH4 = 0.02          # 2 % of CH4 produced, threshold of Liebetrau et al. (2017)
    f_fug_NH3 = 0.05          # 5 % of gaseous NH3, model default (assumption)
    CH4_fug = CH4_mass * f_fug_CH4
    NH3_fug = NH3_mass * f_fug_NH3

    # exact (name, compartment) lookup (bw_lookup.get_flow)
    ch4_air = get_flow("Methane, non-fossil", ("air",), CORE_FLOW_LOCATIONS)
    nh3_air = get_flow("Ammonia", ("air",), CORE_FLOW_LOCATIONS)

    ad_process.new_exchange(input=ch4_air, amount=CH4_fug, type="biosphere", unit="kilogram").save()
    ad_process.new_exchange(input=nh3_air, amount=NH3_fug, type="biosphere", unit="kilogram").save()

    # AD heat: sensible heat of the inflow to T_DIG_C plus losses, minus heat
    # recovered from the HTC product (heat_integration.ad_duty).
    # AD electricity: 0.006 kWh per kg of digester inflow, derived from
    # Berglund and Boerjesson (2006), Table 5.
    e_elec_AD = 0.006
    _ad = hi.ad_duty(m_AD_in, T_in_C=T_AD_in_C, Q_rec_avail_MJ=Q_rec_avail_MJ)
    Q_heat_AD_MJ = _ad["Q_AD_util_MJ"]
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

    print(f"[AD energy | {adm1_mode}] m_AD_in={m_AD_in:.3f} kg | T_in={_ad['T_AD_in_C']} C | "
          f"Heat={Q_heat_AD_MJ:.4f} MJ (recovered {_ad['Q_AD_rec_MJ']:.2f}) | Elec={E_elec_AD_kWh:.4f} kWh")

    if PATHWAY == "A" and wastewater_treat is not None and digestate_mass is not None:
        ad_process.new_exchange(input=wastewater_treat, amount=_wastewater_amount(wastewater_treat, digestate_mass / 1000, "AD effluent (A)"),
                                 type="technosphere", unit="cubic meter").save()

    return CH4_fug, NH3_fug, Q_heat_AD_MJ, E_elec_AD_kWh, _ad


def build_dewatering_exchanges(dewatering, elec, wastewater_treat,
                                E_centrifuge_kWh, centrate_m3,
                                ad_process, m_digestate_total):
    dewatering.new_exchange(input=elec, amount=E_centrifuge_kWh,
                             type="technosphere", unit="kilowatt hour").save()
    dewatering.new_exchange(input=wastewater_treat, amount=_wastewater_amount(wastewater_treat, centrate_m3, "centrate (B)"),
                             type="technosphere", unit="cubic meter").save()
    print(f"[Dewatering] Elec={E_centrifuge_kWh:.3f} kWh  Centrate→WWT={centrate_m3:.4f} m³")


def location_priority(ctx):
    """Dataset location priority of the study: ctx.location_priority, or the
    location code followed by CA (Canadian provinces only), RoW and GLO
    (CA-QC when no code is set)."""
    if ctx.location_priority:
        return list(ctx.location_priority)
    code = ctx.location_code or "CA-QC"
    extra = ("CA", "RoW", "GLO") if code.startswith("CA-") else ("RoW", "GLO")
    return [code] + [c for c in extra if c != code]


def build_biogas_upgrading_exchanges(biogas_upgrading, psa_ref, ei, biomethane_m3, priority):
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
            search_query = "market for electricity, low voltage" if "electricity" in query.lower() else query
            # deterministic exact-name lookup, study location first
            regional = bwl._regional_equivalent(ei, search_query, priority)
            if regional is None:
                regional = exc_input
            biogas_upgrading.new_exchange(input=regional, amount=exc_amount * biomethane_m3,
                                           type=exc_type, unit=exc_unit).save()
        elif exc_type == "biosphere":
            # The dataset's own methane slip is skipped: plant methane losses
            # are counted by f_fug_CH4 in build_ad_exchanges. The 3 % of CH4
            # not recovered by the PSA (PSA_CH4_recovery 0.97) goes to the
            # off-gas oxidizer.
            if name_lower.startswith("methane, non-fossil"):
                continue
            biogas_upgrading.new_exchange(input=relocate_flow(exc_input, CORE_FLOW_LOCATIONS),
                                           amount=exc_amount * biomethane_m3,
                                           type=exc_type, unit=exc_unit).save()


# ─────────────────────────────────────────────────────────────────────────────
# HYDROCHAR END USE: INDUSTRIAL HEAT IN QUEBEC
# ─────────────────────────────────────────────────────────────────────────────

# Flows of the coal furnace template not transferred to hydrochar combustion:
# coal-specific trace elements, radionuclides and organics that HADES cannot
# predict for hydrochar. Matched on whole words.
_COMBUSTION_SKIP_WORDS = [
    "radioactive", "uranium", "radon", "thorium", "lead-210", "polonium",
    "arsenic", "cadmium", "mercury", "antimony", "chromium", "nickel",
    "molybdenum", "vanadium", "thallium", "tin", "beryllium",
    "phosphorus", "dioxin", "dioxins", "benzene", "toluene", "xylene",
    "pah", "polycyclic", "nmvoc",
]


_SKIP_RE = _re.compile(r"\b(" + "|".join(_re.escape(w) for w in _COMBUSTION_SKIP_WORDS) + r")\b")


def build_hydrochar_combustion_exchanges(hc_heat, coal_heat, bios, SO2_per_MJ,
                                          NOx_per_MJ, CO2_bio_per_MJ, Ash_char_pct,
                                          locations=None, pm_scale=None):
    """
    1 MJ of useful heat from dried hydrochar. Template: hard coal industrial
    furnace 1-10 MW, with hydrochar-specific emissions:
      CO2        biogenic, from the hydrochar carbon
      SO2, NOx   from the hydrochar S and N (NOx with the fuel-N conversion of
                 hydrochar_enduse), capped at the emission limit values
      CO, CH4    coal furnace factors, as non-fossil flows
      N2O        coal furnace factor
      PM         coal furnace factors x Ash_char / Ash_coal, capped
    Technosphere inputs of the template (coal supply, etc.) are set to zero:
    the fuel is the foreground hydrochar. Per-MJ factors are computed as (mass
    per FU) / (MJ delivered per FU).
    """
    Ash_coal_pct = 10.0
    PM_scale = (Ash_char_pct / Ash_coal_pct) if Ash_coal_pct > 0 else 1.0
    # SO2, NOx and PM arrive already capped at the emission limit values
    # (hydrochar_enduse); pm_scale is the multiplier on the template PM flows.
    if pm_scale is not None:
        PM_scale = float(pm_scale)
    # Emission compartments are copied from the coal furnace template, so the
    # hydrochar and the displaced fuel are characterized alike; `locations`
    # selects location-named flows in regionalized projects.
    locations = locations or CORE_FLOW_LOCATIONS
    so2_flow = get_flow("Sulfur dioxide", _template_cats(coal_heat, "Sulfur dioxide"), locations)
    nox_flow = get_flow("Nitrogen oxides", _template_cats(coal_heat, "Nitrogen oxides"), locations)
    co2_nf = get_flow("Carbon dioxide, non-fossil", ("air",), locations)
    co_nf = get_flow("Carbon monoxide, non-fossil", _template_cats(coal_heat, "Carbon monoxide, fossil"), locations)
    ch4_nf = get_flow("Methane, non-fossil", _template_cats(coal_heat, "Methane, fossil"), locations)

    hc_heat.new_exchange(input=so2_flow, amount=SO2_per_MJ, type="biosphere", unit="kilogram").save()
    hc_heat.new_exchange(input=nox_flow, amount=NOx_per_MJ, type="biosphere", unit="kilogram").save()
    hc_heat.new_exchange(input=co2_nf, amount=CO2_bio_per_MJ, type="biosphere", unit="kilogram").save()

    for exc in coal_heat.exchanges():
        if exc["type"] == "technosphere":
            hc_heat.new_exchange(input=exc.input, amount=0, type="technosphere", unit=exc["unit"]).save()
            continue
        if exc["type"] != "biosphere":
            continue
        name = exc.input["name"].lower()
        if _SKIP_RE.search(name):
            continue
        if "carbon dioxide" in name or "sulfur dioxide" in name or "nitrogen oxides" in name:
            continue  # written above from hydrochar composition
        if "carbon monoxide, fossil" in name:
            hc_heat.new_exchange(input=co_nf, amount=exc["amount"], type="biosphere", unit=exc["unit"]).save()
        elif "methane, fossil" in name:
            hc_heat.new_exchange(input=ch4_nf, amount=exc["amount"], type="biosphere", unit=exc["unit"]).save()
        elif "dinitrogen monoxide" in name:
            hc_heat.new_exchange(input=relocate_flow(exc.input, locations), amount=exc["amount"],
                                  type="biosphere", unit=exc["unit"]).save()
        elif "particulate" in name:
            # PM flows follow the hydrochar emission location
            hc_heat.new_exchange(input=relocate_flow(exc.input, locations), amount=exc["amount"] * PM_scale,
                                  type="biosphere", unit=exc["unit"]).save()
    print(f"[HC combustion '{hc_heat['name']}'] SO2={SO2_per_MJ:.2e}  NOx={NOx_per_MJ:.2e}  "
          f"CO2bio={CO2_bio_per_MJ:.3f} kg/MJ  PM scale={PM_scale:.2f}  "
          f"flows -> '{so2_flow['name']}' {tuple(so2_flow['categories'])}")


def build_heat_mix_exchanges(mix_act, components, shares):
    """
    1 MJ of an industrial combustion heat mix =
        sum over groups g, datasets d in g of  share_g x weight_d x 1 MJ(d)
    components: {group: [(activity, weight normalized within group)]}
    """
    missing = [k for k in shares if k not in components or components[k] is None]
    if missing:
        raise ValueError(f"Heat mix '{mix_act['name']}': no dataset resolved for {missing}")
    # the rounded shares sum to 0.999; renormalize so that 1 MJ of hydrochar
    # heat displaces exactly 1 MJ of the mix
    total = float(sum(shares.values()))
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Heat mix shares sum to {total:.3f}, not ~1")
    for k, share in shares.items():
        if share <= 0:
            continue
        for act, w in components[k]:
            mix_act.new_exchange(input=act, amount=float(share) / total * float(w),
                                  type="technosphere", unit="megajoule").save()


def _waste_input_amount(waste_act, mass_kg):
    """
    Amount to write for a waste sent to an ecoinvent waste treatment/market.
    ecoinvent waste treatments usually carry a NEGATIVE production amount
    (-1 kg) after import; the consumer's exchange must then be negative too
    for the treatment burden to come out positive. Reading the sign from the
    dataset makes this correct whichever convention your import used.
    """
    prod = [e for e in waste_act.exchanges() if e["type"] == "production"]
    sign = np.sign(prod[0]["amount"]) if prod else 1.0
    return float(mass_kg) * (sign if sign != 0 else 1.0)


def _wastewater_amount(wastewater_treat, volume_m3, where):
    """
    Amount for a wastewater stream sent to ecoinvent "treatment of wastewater,
    average" (production -1 m3), with the sign read from the dataset through
    _waste_input_amount. Checks that the volume is >= 0 and that the written
    amount has the sign of the dataset's production exchange.
    """
    if not np.isfinite(volume_m3) or volume_m3 < 0:
        raise ValueError(f"[_wastewater_amount | {where}] invalid wastewater volume {volume_m3} m3")
    amt = _waste_input_amount(wastewater_treat, volume_m3)
    prod = [e["amount"] for e in wastewater_treat.exchanges() if e["type"] == "production"]
    if prod and volume_m3 > 0 and np.sign(amt) != np.sign(prod[0]):
        raise ValueError(f"[_wastewater_amount | {where}] amount {amt} has the wrong sign "
                         f"for a treatment with production {prod[0]}")
    return amt


def build_hydrochar_enduse_exchanges(ctx, q, bios):
    """
    Hydrochar heat in Quebec. heat_prod = 1 MJ of heat from hydrochar (coal furnace
    template, hydrochar emissions, capped at the emission limit values);
    qc_heat_mix = 1 MJ of the Quebec industrial combustion heat mix.
    waste_treatment demands +MJ_hydrochar of heat_prod and -MJ_hydrochar of
    qc_heat_mix (written in build_system). Boiler ash goes to ctx.ash_disposal
    through waste_treatment when one is configured.
    """
    if ctx.qc_heat_mix is None or ctx.heat_mix_components_qc is None:
        raise ValueError("[hydrochar end use] ctx.qc_heat_mix / ctx.heat_mix_components_qc not set: "
                         "run the notebook cell that creates the Quebec heat mix.")
    for act in (ctx.qc_heat_mix, ctx.heat_prod):
        for exc in list(act.exchanges()):
            if exc["type"] != "production":
                exc.delete()
    build_heat_mix_exchanges(ctx.qc_heat_mix, ctx.heat_mix_components_qc, hce.QC_HEAT_MIX)
    build_hydrochar_combustion_exchanges(
        ctx.heat_prod, ctx.coal_heat, bios, q["SO2_per_MJ"], q["NOx_per_MJ"],
        q["CO2_bio_per_MJ"], q["Ash_char_pct"], locations=CORE_FLOW_LOCATIONS,
        pm_scale=q.get("PM_scale"))

# Centrifuge N split. Molar mass of N taken as 14.0 kg/kmol, consistent with
# ad_characterization._S_IN_equiv_kmol_m3.
N_KG_PER_KMOL = 14.0


def split_centrifuge_N(m_digestate_total, dry_matter_kg, WC_solids,
                       S_IN_kmol_m3, N_dig_kg, max_iter=50, tol=1e-9):
    """
    Split the digestate N between the two outputs of the Pathway B centrifuge:
        * wet solids sent to HTC, at moisture WC_solids (= target_moisture_HTC)
        * liquid sent to wastewater treatment (by difference)

    Particles (and the N bound in them) go with the solids. Dissolved ammonium
    is spread evenly through all the water, so the liquid carries
        N_liquid = S_IN [kmol N/m3] x 14 [kg/kmol] x m_liquid [kg] / 1000

    Small fixed point: removing N_liquid lowers the dry matter of the solids,
    which lowers the solids mass at WC_solids, which raises m_liquid a little,
    which raises N_liquid a little. The update is a strong contraction
    (dN_new/dN = S_IN x 0.014 / (1 - WC_solids) ~ 0.01), so it converges in
    two or three iterations.

    Cap: N_liquid <= N_dig_kg. ADM1 computes S_IN from its own N content of
    amino acids and biomass, not from the elemental N of the feed, so the two
    can differ; the cap keeps the split physical and is printed.

    Mass balance: m_solids + m_liquid = m_digestate_total exactly (the liquid
    is computed by difference, the N mass moves with it).

    Returns (m_solids_kg, m_liquid_kg, dry_matter_solids_kg, N_liquid_kg).
    """
    N_liq = 0.0
    for _ in range(max_iter):
        dry = dry_matter_kg - N_liq
        m_sol = dry / (1.0 - WC_solids)
        m_liq = max(0.0, m_digestate_total - m_sol)
        N_new = min(max(0.0, S_IN_kmol_m3 * N_KG_PER_KMOL * m_liq / 1000.0), N_dig_kg)
        if abs(N_new - N_liq) < tol:
            N_liq = N_new
            break
        N_liq = N_new
    if N_liq >= N_dig_kg - 1e-12 and N_dig_kg > 0:
        print(f"⚠️  [centrifuge N split] ADM1 dissolved N in the liquid "
              f"({S_IN_kmol_m3 * N_KG_PER_KMOL * m_liq / 1000.0:.2f} kg) >= digestate N "
              f"({N_dig_kg:.2f} kg): capped, ALL N leaves with the liquid. ADM1 N "
              f"stoichiometry and elemental N_feed disagree at this point.")
    dry = dry_matter_kg - N_liq
    m_sol = dry / (1.0 - WC_solids)
    m_liq = m_digestate_total - m_sol
    return m_sol, m_liq, dry, N_liq


def dried_basis_for_htc(dp, NH4_kg, m_wet_kg):
    """
    Composition of the wet digestate cake as a dried sample, for the HTC
    surrogate. The surrogate was trained on
    ultimate analyses of dried samples; drying digestate (pH > 7) drives
    80-97 % of its ammonium off as NH3 (Maurer & Mueller 2012, Eng. Life
    Sci. 12:321; Pantelopoulos et al. 2016, Waste Manag. 48:218; Liu et al.
    2021, Nitrogen 2:287). The cake's dissolved ammonium (NH4_kg, kg N) is
    removed from the solids: it is liquid, passes HTC as NH4+ (process water
    pH 4.5-6.5 << pKa 9.25) and is added to the process-water N by the caller.
    dp: digestate dict after the centrifuge split (_C_dig_kg ... _Ash_dig_kg,
    _N_dig_kg = N left in the cake, _fc_ratio); m_wet_kg: wet cake mass.
    Returns dp with dried-basis X_feed %, FC/VM, WC (ammonium now counted
    with the water) and _m_dry_htc_kg, _N_org_htc_kg, _NH4_to_PW_kg.
    """
    kg = {e: float(dp[f"_{e}_dig_kg"]) for e in ("C", "H", "O", "S", "Ash")}
    kg["N"] = float(dp["_N_dig_kg"]) - float(NH4_kg)
    if NH4_kg < 0 or kg["N"] < -1e-6:
        raise ValueError(f"[HTC DRIED BASIS] ammonium {NH4_kg:.3f} kg > cake N {dp['_N_dig_kg']:.3f} kg")
    kg["N"] = max(kg["N"], 0.0)
    total = sum(kg.values())
    if not 0 < total < m_wet_kg:
        raise ValueError(f"[HTC DRIED BASIS] dried mass {total:.2f} kg vs wet cake {m_wet_kg:.2f} kg")
    out = dict(dp)
    for e in ("C", "H", "O", "N", "S", "Ash"):
        out[f"{e}_feed"] = kg[e] / total * 100.0
    fc, vm = ac.proximate_from_fc_ratio(out["Ash_feed"], dp["_fc_ratio"])
    out["FC_feed"], out["VM_feed"] = fc, vm
    out["WC"] = 100.0 * (1.0 - total / m_wet_kg)
    out["_m_dry_htc_kg"] = total
    out["_N_org_htc_kg"] = kg["N"]
    out["_NH4_to_PW_kg"] = float(NH4_kg)
    return out


def _clamp_report(label, **raw_kg):
    """
    The process water elements are computed by difference,
    X_pw = X_feed - X_char - X_gas, and clamped at zero. A negative difference
    means the surrogate put more of that element in the char and gas than the
    feed contained. Returns the clamped elements as "C;N" text (empty if none)
    for the pw_clamped_elements column and prints the size of each clamp.
    """
    neg = {k: v for k, v in raw_kg.items() if v < 0}
    if neg:
        txt = ", ".join(f"{k}={v:.3f} kg" for k, v in neg.items())
        print(f"⚠️  [{label}] PW element(s) clamped to 0 (char + gas > feed): {txt}")
    return ";".join(neg)



# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL QUALITY CHECKS, run at the end of every build_system.
# A failed check is written to waste_treatment["qa_flags"] and printed; it
# never passes silently. Mass and energy balance failures RAISE.
# ─────────────────────────────────────────────────────────────────────────────
# Plausibility ranges: 1st-99th percentiles of the surrogate training targets
# (938 rows, 122 studies).
QA_RANGES_CHAR = {
    "Y_char (%)": (17.7, 90.9),
    "HHV_char (MJ/kg)": (9.7, 33.3),
    "HC_atomic": (0.64, 1.97),
    "OC_atomic": (0.07, 0.83),
}
# Biogas CH4 volume fraction of anaerobic digestion: 40-75 % (Ryckebosch,
# Drouillon & Vervaeren 2011, Biomass and Bioenergy 35:1633-1645, Table 1).
QA_RANGE_CH4_VOL = (0.40, 0.75)
QA_MB_REL_TOL = 1e-6          # reactor mass balances, relative to m_ref
QA_ENERGY_REL_TOL = 1e-6      # heat cascade closure, relative to Q_HTC_gross


def qa_checks(PATHWAY, inv, mb, ad_hi, CH4_mass, CO2_mass, MB, m_ref,
              energy_residual=None):
    """Return a list of flag strings (empty = all checks passed).
    Raises RuntimeError for a mass or energy balance that does not close."""
    flags = []
    # (a) mass balances (exact by construction: a failure is a code error)
    for v in MB:
        if v is not None and abs(float(v)) > QA_MB_REL_TOL * m_ref:
            raise RuntimeError(f"[QA build_system] reactor mass balance residual {v:.3e} kg "
                               f"> {QA_MB_REL_TOL * m_ref:.1e} kg")
    # (b) energy closure of the HTC heat cascade (heat_integration.metadata)
    gross = abs(float((mb.get("heat_htc") or {}).get("Q_HTC_gross_MJ") or 0.0))
    if energy_residual is not None and abs(float(energy_residual)) > QA_ENERGY_REL_TOL * max(gross, 1.0):
        raise RuntimeError(f"[QA build_system] heat cascade closure residual "
                           f"{energy_residual:.3e} MJ > tol")
    # (c) physical bounds
    if mb["PW_yield"] < 0:
        flags.append(f"PW_yield {mb['PW_yield']:.2f} kg < 0")
    for k in ("C_char (%)", "H_char (%)", "O_char (%)", "N_char (%)", "S_char (%)", "Ash_char (%)"):
        v = float(inv.get(k, float("nan")))
        if not 0.0 <= v <= 100.0:
            flags.append(f"{k} = {v:.2f} outside [0, 100]")
    for T in (mb.get("T_prod1_C"), mb.get("T_prod2_C"), (ad_hi or {}).get("T_AD_in_C")):
        if T is not None and not hi.T_AMB_C - 1e-9 <= T <= 374.0:
            flags.append(f"stream temperature {T:.1f} C outside [T_amb, 374]")
    # (d) applicability / plausibility vs the training data
    C, H, O = (float(inv.get(k, float("nan"))) for k in ("C_char (%)", "H_char (%)", "O_char (%)"))
    vals = {"Y_char (%)": float(inv.get("Y_char (%)", float("nan"))),
            "HHV_char (MJ/kg)": float(inv.get("HHV_char (MJ/kg)", float("nan"))),
            "HC_atomic": (H / 1.008) / (C / 12.011) if C > 0 else float("nan"),
            "OC_atomic": (O / 15.999) / (C / 12.011) if C > 0 else float("nan")}
    for k, (lo, hi_) in QA_RANGES_CHAR.items():
        if not lo <= vals[k] <= hi_:
            flags.append(f"{k} = {vals[k]:.3f} outside training p1-p99 [{lo}, {hi_}]")
    if inv.get("in_training_range") is False:
        flags.append(f"HTC surrogate extrapolation: {inv.get('domain_violations', '')}")
    if inv.get("mass_balance_note"):
        flags.append(f"HTC surrogate: {inv['mass_balance_note']}")
    if (inv.get("C_recovery_raw") or 0.0) > 1.0:
        flags.append(f"carbon balance: char C recovery {inv['C_recovery_raw']:.3f} > 1")
    # (e) biogas composition (ideal gas, 22.414 L/mol at 0 C, 1 atm)
    n_ch4, n_co2 = CH4_mass / 16.043, CO2_mass / 44.009
    if n_ch4 + n_co2 > 0:
        y = n_ch4 / (n_ch4 + n_co2)
        if not QA_RANGE_CH4_VOL[0] <= y <= QA_RANGE_CH4_VOL[1]:
            flags.append(f"biogas CH4 {y:.3f} vol outside {QA_RANGE_CH4_VOL} (Ryckebosch et al. 2011)")
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# SPIN-UP
# ─────────────────────────────────────────────────────────────────────────────

def spinup_ss_state(ctx: SystemContext, pathway: str,
                     adm1_HRT_days: int = 20,
                     t_spinup: int = 400,
                     verbose: bool = True,
                     TS_target: float = 0.08,
                     ml_module: str = None) -> dict:
    """
    ADM1 spin-up to an adapted steady state for `pathway`; pass the result
    as adm1_ss_state to build_system().

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
# BUILD SYSTEM
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
    # ── unpack the context ──────────────────────────────────────
    # single source of the feedstock, T and RT; raises on inconsistent or
    # out-of-range inputs (check_inputs)
    ml_params = check_inputs(ctx, adm1_mode=adm1_mode, adm1_HRT_days=adm1_HRT_days,
                             TS_target=TS_target)
    PATHWAY = ctx.PATHWAY
    m_feedstock = ctx.m_feedstock
    default_params = ctx.default_params
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

    # ── clear all foreground activities before rebuilding ────────
    db = bd.Database(FOREGROUND_DB)
    for act in db:
        for exc in list(act.exchanges()):
            if exc["type"] != "production":
                exc.delete()
    print(f"🧹 Cleared all exchanges in '{FOREGROUND_DB}'")

    # ── load the HTC surrogate ──────────────────────────────────
    ml = _load_ml_module(ml_module)
    HTC_parametric_ML = ml.HTC_parametric_ML
    print(f"✅ Loaded ML module: {os.path.basename(getattr(ml, '__file__', ml_module))}")
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

        # Digestate composition from the feed, the ADM1 biogas and measured
        # digestate coefficients (ad_characterization). Ammonium: ADM1 steady
        # state S_IN x inflow. ADM1 computes S_IN from its own N stoichiometry,
        # not from the elemental N of the feed; if the value is outside
        # [0, N_feed] or there is no steady state, the measured ammonium share
        # f_NH4 is used instead and the switch is flagged.
        _NH4_source_B = "ADM1_S_IN"
        try:
            _NH4_B = ac.adm1_ammonium_kg(adm1_out, feed_inputs["_V_feed_m3"])
        except (ValueError, KeyError, TypeError):
            _NH4_B = None
        if _NH4_B is None or not (0.0 <= _NH4_B <= N_feed_mass):
            print(f"⚠️  [DIGESTATE] ADM1 ammonium {_NH4_B} kg N not usable "
                  f"(feed N {N_feed_mass:.2f} kg): measured f_NH4 used instead")
            _NH4_B = None
            _NH4_source_B = "measured_f_NH4"
        digestate_params = ac.digestate_simple(
            ctx.feedstock_name, C_feed_mass, H_feed_mass, O_feed_mass,
            N_feed_mass, S_feed_mass, ash_kg_feed,
            CH4_mass, CO2_mass, m_feedstock_slurry, NH4_N_kg=_NH4_B)
        print(f"[Pathway B | DIGESTATE] dry {digestate_params['_m_dry_kg']:.1f} kg  "
              f"VS destroyed {digestate_params['_dg_VS_destroyed'] * 100:.1f} %  "
              f"NH4-N {digestate_params['_NH4_N_kg']:.2f} kg ({_NH4_source_B})  "
              f"{digestate_params['_dg_warnings']}")
        N_dig_kg_B = digestate_params["_N_dig_kg"]

        # ── DEWATERING ────────────────────────────────────────────
        WC_target_dewater = target_moisture_HTC
        WC_dig_raw = digestate_params["WC"] / 100

        m_digestate_total = m_feedstock_slurry - CH4_mass - CO2_mass
        dry_matter_dig = digestate_params["_m_dry_kg"]   # exact (WC is rounded)

        N_to_centrate_kg_B = 0.0
        if WC_dig_raw > WC_target_dewater:
            # dissolved ammonium leaves with the centrate water (S_IN x its
            # volume); particles and their N go to HTC
            m_cake, m_centrate, dry_matter_dig, N_to_centrate_kg_B = split_centrifuge_N(
                m_digestate_total, dry_matter_dig, WC_target_dewater,
                digestate_params["_S_IN_equiv_kmol_m3"], N_dig_kg_B)
            digestate_params = dict(digestate_params)
            digestate_params["_N_dig_kg"] = N_dig_kg_B - N_to_centrate_kg_B
            print(f"[Pathway B | centrifuge N split] N to centrate {N_to_centrate_kg_B:.2f} kg, "
                  f"N to HTC {digestate_params['_N_dig_kg']:.2f} kg (of {N_dig_kg_B:.2f} kg)")

            e_centrifuge = 0.003   # kWh per kg of digestate, model default (assumption)
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

        # the surrogate was trained on dried samples: the ammonium left in the
        # cake water is removed from the solids and joins the process water
        # (dried_basis_for_htc)
        NH4_cake_kg_B = max(0.0, float(digestate_params["_NH4_N_kg"]) - N_to_centrate_kg_B)
        digestate_params = dried_basis_for_htc(digestate_params, NH4_cake_kg_B,
                                               m_digestate_for_htc)
        print(f"[Pathway B | HTC dried basis] ammonium {NH4_cake_kg_B:.2f} kg N -> process "
              f"water; surrogate N {digestate_params['N_feed']:.2f} %, dry "
              f"{digestate_params['_m_dry_htc_kg']:.1f} kg, WC {digestate_params['WC']:.2f} %")
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
        # use the feedstock's own moisture; drier feeds are diluted with tap
        # water to the HTC moisture target in step 2
        WC_native = default_params["WC"] / 100.0
        WC_reactor = max(WC_native, target_moisture_HTC)
        ml_params_htc = {**ml_params, "WC": WC_reactor * 100.0}
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
        if WC_native < target_moisture_HTC:
            added_water = (m_feedstock * (1 - WC_native) - m_feedstock * (1 - target_moisture_HTC)) / (1 - target_moisture_HTC)
            added_water = max(0.0, added_water)
            m_feedstock_effective = m_feedstock + added_water
            WC_feed = target_moisture_HTC
            print(f"Added water: {added_water:.3f} kg (native WC={WC_native*100:.1f}% -> target {target_moisture_HTC*100:.0f}%)")
        else:
            added_water = 0.0
            m_feedstock_effective = m_feedstock
            WC_feed = WC_native
            print(f"No water addition needed — native WC={WC_native*100:.1f}% >= target {target_moisture_HTC*100:.0f}%")

        # [QA] dry matter is conserved through water addition (rel tol 1e-6)
        _dry_in = m_feedstock * (1 - WC_native)
        _dry_htc = m_feedstock_effective * (1 - WC_feed)
        if abs(_dry_htc - _dry_in) > 1e-6 * max(_dry_in, 1.0):
            raise ValueError(f"[build_system | Pathway A] dry mass not conserved through water "
                             f"addition: {_dry_in:.4f} kg in, {_dry_htc:.4f} kg to HTC (rel tol 1e-6)")

    # =========================================================
    # STEP 3 — HTC MASS BALANCE & ENERGY
    # =========================================================

    # HTC feed temperature. Pathway B: wet solids leaving the digestate
    # centrifuge plus any tap water at T_amb. Pathway A: raw feedstock at
    # T_amb (None).
    _T_feed_HTC_C = (hi.htc_feed_temperature_B(m_digestate_for_htc, added_water)
                     if PATHWAY == "B" else None)
    mb = compute_htc_mass_balance(
        m_feed_eff=m_feedstock_effective, WC_feed=WC_feed, Y_char_pct=Y_char_pct,
        T_HTC=T_HTC, MC_char=MC_char, target_MC_char_dry=target_MC_char_dry,
        eps_HR=ctx.eps_HR, T_feed_C=_T_feed_HTC_C,
    )
    HC_wet_yield = mb["HC_wet_yield"]
    HC_dry_yield = mb["HC_dry_yield"]
    PW_yield = mb["PW_yield"]
    CO2_yield = mb["CO2_yield"]
    CO_yield = mb["CO_yield"]
    Q_heat_kWh = mb["Q_heat_kWh"]
    elec_MJ = mb["elec_MJ"]
    wastewater_m3 = mb["wastewater_m3"]
    dryer_fan_kWh = mb["dryer_fan_kWh"]   # dryer fans + filter press
    HC_dry_mass = mb["HC_dry_mass"]

    print_mass_balance_diagnostic(m_feedstock_effective, mb, WC_feed)

    # Heat delivered = dry char mass x HHV_char / HHV_coal / 0.04325 kg coal
    # per MJ (hydrochar_enduse). HHV_char is a dry-basis value, so it applies
    # to the bone-dry char mass.
    q_enduse = hce.compute_enduse_quantities(inv, HC_dry_mass, nox_conv=ctx.nox_conv)
    HHV_coal = hce.HHV_COAL_MJ_PER_KG
    coal_needed_1MJ = hce.COAL_KG_PER_MJ
    SubstitutionFactor_HC_COAL = q_enduse["SubstitutionFactor_HC_COAL"]
    MJ_hydrochar = q_enduse["MJ_hydrochar"]

    print(f"Derived yields: HC={HC_wet_yield:.3f} kg, PW={PW_yield:.3f} kg, "
          f"CO2={CO2_yield:.3f} kg, CO={CO_yield:.3f} kg")
    print(f"Energy use: {Q_heat_kWh:.3f} kWh (HTC heating), {elec_MJ:.3f} MJ = "
          f"{elec_MJ/3.6:.3f} kWh (drying), {MJ_hydrochar:.3f} MJ (HC substitution)")

    dry_mass_in = m_feedstock_effective * (1 - WC_feed)
    inv["Y_gas (%)"] = mb["Y_gas_drybasis"]
    inv["Y_liquid (%)"] = PW_yield / dry_mass_in * 100

    # =========================================================
    # PATHWAY A: normalization to 1 t of feedstock
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
        dryer_fan_kWh = dryer_fan_kWh * _scale_A
        HC_dry_mass = HC_dry_mass * _scale_A
        q_enduse = hce.compute_enduse_quantities(inv, HC_dry_mass, nox_conv=ctx.nox_conv)

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

        # dry matter of the feedstock at its native moisture (before dilution)
        dry_fraction_A = 1 - default_params["WC"] / 100.0
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
        _pw_clamped = _clamp_report(
            "Pathway A PW",
            C=C_feed_mass_A - (C_char_mass + C_gas_mass), H=H_feed_mass_A - H_char_mass,
            O=O_feed_mass_A - (O_char_mass + O_gas_mass), N=N_feed_mass_A - N_char_mass,
            S=S_feed_mass_A - S_char_mass)

        rho_PW = 1.02
        pw_inputs = translate_pw_to_adm1(
            C_pw=C_pw_mass, H_pw=H_pw_mass, O_pw=O_pw_mass,
            N_pw=N_pw_mass, S_pw=S_pw_mass,
            PW_yield_L=(PW_yield / _scale_A) / rho_PW,
            mode=adm1_mode, HRT_days=adm1_HRT_days, T_HTC=T_HTC,
            RT_HTC=RT_HTC,
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

    PSA_CH4_recovery = 0.97   # model default (assumption)
    # LHV of methane, 35.8 MJ/Nm3. LHV_NG is used only to convert MJ to m3 in
    # the printout; the natural gas credit is in MJ.
    LHV_biomethane = 35.8
    LHV_NG = 35.8

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

    # Digester inlet temperature and recovered heat.
    # Pathway A: process water after E2 and the letdown (<= T_SEP_C), mixed
    # with dilution water at T_amb; E2 heat preheats the dilution water.
    # Pathway B: raw feed at T_amb; residual product heat (down to
    # max(T_dig + DT_min, T_SEP_C)) warms the digester feed.
    _T_prod2_C = mb["T_prod2_C"]
    _T_PW_AD_C = None
    if PATHWAY == "A":
        _T_PW_AD_C = hi.pw_temperature_after_letdown(_T_prod2_C)
        _T_AD_in_C = hi.mix_temperature([(PW_yield, _T_PW_AD_C),
                                         (dilution_water_PW_kg, hi.dilution_temperature())])
        _Q_E2_A_MJ = (mb.get("heat_E2") or {}).get("Q_E2_MJ", 0.0) * _scale_A
        _Q_rec_AD_MJ = hi.e2_heat_to_dilution_MJ(_Q_E2_A_MJ, dilution_water_PW_kg)
    else:
        _T_AD_in_C = hi.T_AMB_C
        _Q_rec_AD_MJ = hi.residual_heat_to_ad_MJ(m_feedstock_effective, _T_prod2_C)

    CH4_fug, NH3_fug, Q_heat_AD_MJ, E_elec_AD_kWh, ad_hi = build_ad_exchanges(
        ad_process=ad_process, bios=bios, ei=ei,
        CH4_mass=CH4_mass, NH3_mass=NH3_mass, m_AD_in=m_AD_in,
        adm1_mode=adm1_mode, PATHWAY=PATHWAY,
        ad_elec=ad_elec, ad_heat=ad_heat,
        wastewater_treat=wastewater_treat if PATHWAY == "A" else None,
        digestate_mass=digestate_mass if PATHWAY == "A" else None,
        dilution_water_m3=(pw_inputs.get("_dilution_water_m3", 0.0) * _scale_A if PATHWAY == "A"
                            else feed_inputs.get("_dilution_water_kg", 0.0) / 1000.0),
        tap_water=tap_water,
        T_AD_in_C=_T_AD_in_C, Q_rec_avail_MJ=_Q_rec_AD_MJ,
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

    build_drying_exchanges(drying, elec, drying_heat, wastewater_treat, elec_MJ, wastewater_m3,
                           fan_kWh=dryer_fan_kWh)
    # hydrochar heat displaces the Quebec industrial combustion heat mix;
    # emissions capped at the emission limit values (Directive (EU) 2015/2193)
    build_hydrochar_enduse_exchanges(ctx, q_enduse, bios)

    for exc in list(biogas_upgrading.exchanges()):
        if exc["type"] != "production":
            exc.delete()
    build_biogas_upgrading_exchanges(biogas_upgrading, psa_ref, ei, biomethane_m3,
                                     location_priority(ctx))

    for exc in list(heat_biomethane.exchanges()):
        if exc["type"] != "production":
            exc.delete()
    for exc in biomethane_heat_ref.exchanges():
        if exc["type"] == "biosphere":
            # biomethane combustion emissions located in Quebec (template is RoW)
            heat_biomethane.new_exchange(input=relocate_flow(exc.input, CORE_FLOW_LOCATIONS),
                                          amount=exc["amount"],
                                          type="biosphere", unit=exc["unit"]).save()
        elif exc["type"] == "technosphere":
            heat_biomethane.new_exchange(input=exc.input, amount=0,
                                          type="technosphere", unit=exc["unit"]).save()

    heat_naturalgas.new_exchange(input=ng_heat_market, amount=1,
                                  type="technosphere", unit="megajoule").save()

    waste_treatment.new_exchange(input=htc_reactor, amount=1, type="technosphere", unit="unit").save()
    waste_treatment.new_exchange(input=drying, amount=1, type="technosphere", unit="unit").save()
    waste_treatment.new_exchange(input=heat_prod, amount=MJ_hydrochar, type="technosphere", unit="megajoule").save()
    waste_treatment.new_exchange(input=ctx.qc_heat_mix, amount=-MJ_hydrochar, type="technosphere", unit="megajoule").save()
    if ctx.ash_disposal is not None and q_enduse["ash_kg"] > 0:
        waste_treatment.new_exchange(input=ctx.ash_disposal,
                                     amount=_waste_input_amount(ctx.ash_disposal, q_enduse["ash_kg"]),
                                     type="technosphere", unit="kilogram").save()
    elif q_enduse["ash_kg"] > 0:
        print(f"⚠️  [END USE] boiler ash {q_enduse['ash_kg']:.1f} kg/FU: no ash treatment configured")
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
    # inputs of this run, so that every stored result says which case it is
    waste_treatment["T_HTC"] = float(T_HTC)
    waste_treatment["RT_HTC"] = float(RT_HTC)
    for _k in FEED_KEYS:
        waste_treatment[f"in_{_k}"] = float(default_params[_k])
    waste_treatment["feedstock_name"] = ctx.feedstock_name
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
    # heat integration, end use, digestate and QA outputs
    _hi_extra = {"T_PW_to_AD_C": _T_PW_AD_C}
    for _k, _v in hi.metadata(mb, ad_hi, scale=(_scale_A if PATHWAY == "A" else 1.0),
                              extra=_hi_extra).items():
        waste_treatment[_k] = _v
    waste_treatment["dryer_fan_press_kWh"] = float(dryer_fan_kWh)
    waste_treatment["E_HTC_elec_kWh"] = float(Q_heat_kWh)          # HTC reactor heating (electric)
    waste_treatment["E_dryer_heat_kWh"] = float(elec_MJ) / 3.6     # electric belt dryer heat
    waste_treatment["Q_heat_AD_MJ"] = float(Q_heat_AD_MJ)
    waste_treatment["E_elec_AD_kWh"] = float(E_elec_AD_kWh)
    waste_treatment["CH4_fugitive_kg"] = float(CH4_fug)
    waste_treatment["HC_dry_mass_kg"] = float(HC_dry_mass)
    waste_treatment["HHV_coal_MJ_kg"] = float(HHV_coal)
    waste_treatment["LHV_biomethane_MJ_Nm3"] = float(LHV_biomethane)
    waste_treatment["C_recovery_raw"] = inv.get("C_recovery_raw")
    for _k in ("NOx_comb_kg", "SO2_comb_kg", "PM_comb_kg", "NOx_capped", "SO2_capped",
               "PM_capped", "NOx_uncapped_mgNm3", "SO2_uncapped_mgNm3", "PM_uncapped_mgNm3",
               "V_fluegas_Nm3", "ash_kg", "CO2_bio_kg"):
        _v = q_enduse[_k]
        waste_treatment[f"enduse_{_k}"] = bool(_v) if isinstance(_v, (bool, np.bool_)) else float(_v)
    waste_treatment["pw_clamped_elements"] = _pw_clamped if PATHWAY == "A" else ""
    if PATHWAY == "B":
        for _k in ("_NH4_N_kg", "_dg_VS_destroyed", "_dg_f_NH4", "_dg_warnings",
                   "_m_dry_htc_kg", "_NH4_to_PW_kg"):
            waste_treatment["dg" + _k] = digestate_params.get(_k)
        waste_treatment["dg_NH4_source"] = _NH4_source_B
        waste_treatment["N_to_centrate_kg"] = float(N_to_centrate_kg_B)
    waste_treatment["qa_flags"] = "; ".join(qa_checks(
        PATHWAY=PATHWAY, inv=inv, mb=mb, ad_hi=ad_hi, CH4_mass=CH4_mass, CO2_mass=CO2_mass,
        MB=[waste_treatment.get(k) for k in ("MB_HTC_A", "MB_AD_A", "MB_AD_B", "MB_DW_B", "MB_HTC_B")],
        m_ref=1000.0, energy_residual=waste_treatment.get("energy_closure_residual_MJ")))
    print(f"[QA] flags: {waste_treatment['qa_flags'] or 'none'}")

    waste_treatment.save()

    print(f"\n✅ System '{FOREGROUND_DB}' built — PATHWAY {PATHWAY} "
          f"— model '{os.path.basename(str(ml_module))}' — ADM1 mode '{adm1_mode}'")

    if PATHWAY == "A":
        feed_inputs = pw_inputs

    return waste_treatment, adm1_out.get("ss_state", None), adm1_out, feed_inputs, inv