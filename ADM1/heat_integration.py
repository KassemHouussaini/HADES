"""
heat_integration.py
===================
Heat and electricity demands of the HTC and AD units, with heat integration
between them. Pure Python and configuration, no Brightway: system_builder.py
calls these functions and writes the resulting duties as exchanges.

HEAT CASCADE (applied in this order in every run)
-------------------------------------------------
The hot HTC product (slurry at T_HTC) is the only significant heat source. It
is cooled in steps, each step feeding a sink at a lower temperature:

  Match 1  hot product -> HTC feed preheat
           Q_rec1  = eps_HR x m x cp x (T_HTC - T_feed)
           T_prod1 = T_HTC - eps_HR x (T_HTC - T_feed)
           T_feed  = T_AMB_C for the raw feedstock (Pathway A) and the
           temperature of the wet solids leaving the digestate centrifuge for
           Pathway B (htc_feed_temperature_B()).
  E2       product cooled to the separation temperature T_SEP_C before the
           pressure letdown and the filter press; the heat released goes to the
           digester side (Pathway A: dilution water; Pathway B: raw-feed
           digester) up to the digester demand, and the rest is rejected.
  Match 2  digester heat demand = sensible heat of the inflow to T_DIG_C plus
           wall losses; recovered heat is used first and the remainder is
           supplied by the AD heat utility (natural gas in system_builder).
  Dryer    the belt dryer is supplied by electricity; it takes no heat from
           the product.

Heat used at a higher temperature is not available lower down, so more feed
preheat leaves less heat for the digester.

UNITS: masses in kg, temperatures in C, heat in MJ unless the name says kWh.
Extensive outputs scale linearly with mass; temperatures are intensive.

REFERENCES
----------
[1] Lucian, M., Fiori, L. (2017). Hydrothermal carbonization of waste biomass:
    process design, modeling, energy efficiency and cost analysis. Energies
    10, 211. Sec. 3.3: the preheated slurry remains 88, 81 and 95 K below the
    reactor temperature at 180, 220 and 250 C, i.e. eps_HR = 0.45 to 0.61 for
    a feed at 10 to 20 C. Appendix A.5: heat of reaction conservatively set to
    zero. Thermal losses about 40 % of the total heat demand in their design.
[2] Khoury, O. et al. (2026). Techno-economic analysis of hydrothermal
    carbonization of sewage sludge for phosphorus retention and carbon
    sequestration. Waste Manag. 115780. Pinch-analysis target of 86 % heat
    recovery (including reaction heat).
[3] Kemp, I.C. (2007). Pinch Analysis and Process Integration, 2nd ed.
    Butterworth-Heinemann. Minimum approach temperatures.
[4] Funke, A., Ziegler, F. (2010). Hydrothermal carbonization of biomass: a
    summary and discussion of chemical mechanisms for process engineering.
    Biofuels Bioprod. Biorefin. 4, 160-177. HTC is slightly exothermic.
[5] Tchobanoglous, G. et al. (2014). Metcalf & Eddy, Wastewater Engineering:
    Treatment and Resource Recovery, 5th ed. Digester heating: sensible heat
    of the feed plus U x A x dT losses through walls, floor and roof.
[6] Model-Based Evaluation of Hydrothermal Treatment for the Energy Efficient
    Dewatering and Drying of Sewage Sludge. Processes 9 (2021) 1346.
    Mechanical dewatering 20-30 kWh per t TS; about 65-70 % DM after
    hydrothermal treatment.
[7] TerraNova Energy, FAQ (https://www.terranova-energy.com/en/faq-en/):
    slurry cooled, expanded to ambient pressure and dewatered in a chamber
    filter press to about 65 % DM.
[8] AHT CleanTec, hydrochar production
    (https://www.aht-cleantec.com/en/produkte/hydrochar-production/):
    hydrochar mechanically dewatered to up to 70 % DM.
[9] Zhang, Y. (2013). Anaerobic digestion system: energy balance. VALORGAS
    summer school, lecture 13, University of Southampton. U values (W/m2/C)
    reproducing the Metcalf & Eddy table: insulated concrete wall 0.6-0.8,
    concrete floor on dry earth 1.7, fixed concrete cover with 25 mm
    insulation 1.2-1.6.
[10] Tanczuk, M., Kostowski, W. (2021). Technical, energetic and economic
    optimization analysis of selection of heat source for municipal sewage
    sludge dryer. Energies 14, 316. Belt dryer: 1 kWh of heat and 0.085 kWh of
    electricity per kg of water evaporated; 0.5-1.2 kWh/kg for belt dryers
    worldwide.
"""

import math

# =============================================================================
# PARAMETERS
# =============================================================================

# Slurry heat capacity (water value, applied to the whole slurry).
CP_KJ_PER_KG_K = 4.18

# Ambient temperature (C): raw feed, tap water and outside of the digester.
# Model default 10 C (Paper 1). Paper 2 province presets set it to the
# province's 1991-2020 annual mean (lca_contexts.ambient_overrides), 2.4 to
# 10.5 C for the ten provinces. Allowed range 0-20 C (lower bound lowered
# from 5 C so that the Prairie provinces fit).
T_AMB_C = 10.0
T_AMB_RANGE_C = (0.0, 20.0)

# Mesophilic digester temperature; must equal ADM1 T_op (checked by
# system_builder at import).
T_DIG_C = 35.0

# Minimum approach temperature of indirect liquid exchangers [3].
DT_MIN_K = 10.0

# ---- Match 1: HTC feed preheat --------------------------------------------
# eps_HR = heat recovered from the product to the feed / maximum sensible heat
# of the slurry. 0.55 lies within the 0.45-0.61 implied by the two-stage flash
# preheating of [1] and below the 86 % pinch target of [2].
EPS_HR_CENTRAL = 0.55
EPS_HR_RANGE = (0.5, 0.6)

# Efficiency of the HTC heat utility (electricity to heat). Model default
# (assumption).
ETA_HTC_UTILITY = 0.9

# Additional HTC thermal losses as a fraction of the utility heat demand.
# 0 in the default configuration; [1] reports about 0.4 for their design.
F_LOSS_HTC = 0.0
F_LOSS_HTC_RANGE = (0.0, 0.4)

# HTC heat of reaction (MJ released per kg dry feed). HTC is slightly
# exothermic [4]; set to zero as in [1], which slightly overestimates the HTC
# heat demand.
DH_RXN_HTC_MJ_PER_KG_DRY = 0.0

# ---- Pathway B: HTC feed temperature ---------------------------------------
# The Pathway B HTC feed is the wet solids stream leaving the digestate
# centrifuge, at T_DIG_C minus a temperature loss in the centrifuge and buffer
# storage (model default 5 K, range 0-10 K). Tap water added to reach the HTC
# moisture target enters at T_AMB_C and is mixed by mass.
DT_DEWATER_LOSS_K = 5.0
DT_DEWATER_LOSS_RANGE_K = (0.0, 10.0)

# ---- Mechanical dewatering (filter press) before the dryer ----------------
# MC_char = water fraction of the wet hydrochar leaving the filter press. The
# liquid removed by the press is part of the process water. Central 0.35
# (65 % DM) from sewage sludge hydrochar data [6-8]; the upper bound 0.50
# covers feedstocks without measured data.
MC_CHAR_CENTRAL = 0.35
MC_CHAR_RANGE = (0.30, 0.50)

# Filter press electricity per tonne of dry solids pressed; 20-30 kWh/t TS [6].
PRESS_KWH_PER_T_DS = 25.0
PRESS_KWH_RANGE = (20.0, 30.0)

# ---- Dryer --------------------------------------------------------------------
# Belt dryer heat per kg of water evaporated: 1.0 kWh/kg = 3.6 MJ/kg, range
# 0.5-1.2 kWh/kg [10]. The latent heat alone is 2.26 MJ/kg.
DRYER_SEC_TH_MJ_PER_KG = 3.6
DRYER_SEC_TH_RANGE = (1.8, 4.32)

# Dryer electricity (fans, belt drive) per kg of water evaporated [10].
DRYER_FAN_KWH_PER_KG = 0.085

# Evaporated water leaves with the exhaust air (no condensate to wastewater).
DRYER_CONDENSATE_TO_WWT = False

# ---- Pressure letdown before the filter press ------------------------------
# The product is expanded to ambient pressure before the filter press [7], so
# the process water leaving the press cannot exceed about 100 C. The flash
# steam mass is neglected in the process water balance.
T_LETDOWN_C = 100.0

# ---- Product cooler E2 and separation temperature --------------------------
# The product is cooled to T_SEP_C (60-80 C) before separation to limit the
# volatilization of organic compounds. Heat below T_SEP_C is not recovered.
T_SEP_C = 70.0
T_SEP_RANGE_C = (60.0, 80.0)

# ---- Digester --------------------------------------------------------------
# Wall, floor and roof losses as a fraction of the sensible heat needed to
# bring the inflow from T_AMB_C to T_DIG_C. 0.2 is what digester_wall_loss()
# (Metcalf & Eddy U values [5, 9], insulated tank) gives for a plant treating
# about 10 t of dry matter per day; the fraction scales with capacity^(-1/3).
# Model default; range 0.1-0.3.
F_LOSS_AD = 0.2
F_LOSS_AD_RANGE = (0.1, 0.3)

# Plant size and tank geometry used by digester_wall_loss(), the U x A x dT
# check behind F_LOSS_AD. The loss per FU depends on the plant size, because
# the tank surface scales with volume^(2/3).
PLANT_CAPACITY_T_DM_PER_DAY = 10.0

# Tank shape: vertical cylinder, height / diameter.
DIGESTER_H_OVER_D = 1.0

# U values (W/m2/K), middle of the ranges in [9]; insulated wall and cover.
U_WALL_W_M2K = 0.7    # concrete wall 300 mm above ground, INSULATED (0.6-0.8)
U_ROOF_W_M2K = 1.4    # fixed concrete cover 100 mm + 25 mm insulation (1.2-1.6)
U_FLOOR_W_M2K = 1.7   # concrete floor 300 mm on dry earth (single value)
# Temperature of the Pathway A dilution water. None: tap water at T_AMB_C.
T_DILUTION_C = None


def dilution_temperature():
    """Temperature of the Pathway A dilution water (C)."""
    return T_AMB_C if T_DILUTION_C is None else float(T_DILUTION_C)


def _mcp_MJ_per_K(m_kg):
    """Heat capacity of m_kg of slurry (MJ/K)."""
    return m_kg * CP_KJ_PER_KG_K / 1000.0


# =============================================================================
# PATHWAY B: HTC FEED TEMPERATURE
# =============================================================================
def htc_feed_temperature_B(m_solids_kg, m_added_water_kg=0.0):
    """
    Inlet temperature of the Pathway B HTC feed = wet solids leaving the
    digestate centrifuge (at T_DIG_C - DT_DEWATER_LOSS_K) mixed with any tap
    water added before HTC (at T_AMB_C). Adiabatic mixing, equal cp.

    Example: 4000 kg solids at 30 C + 0 kg water -> 30 C.
             4000 kg solids at 30 C + 1000 kg water at 10 C -> 26 C.
    """
    T_solids = T_DIG_C - DT_DEWATER_LOSS_K
    return mix_temperature([(m_solids_kg, T_solids), (m_added_water_kg, T_AMB_C)])


# =============================================================================
# MATCH 1: HTC HEATING WITH FEED PREHEAT
# =============================================================================
def htc_duty(m_slurry_kg, m_dry_kg, T_HTC_C, eps_HR=None, T_amb_C=None):
    """
    Heat duty of the HTC reactor per batch of m_slurry_kg (feed + added water).
    Returns a dict; Q_HTC_util_kWh is the electricity written to htc_reactor.

    T_amb_C = temperature of the feed entering the preheater. None: T_AMB_C
    (raw feedstock). Pathway B passes htc_feed_temperature_B().
    """
    eps = EPS_HR_CENTRAL if eps_HR is None else float(eps_HR)
    if not 0.0 <= eps < 1.0:
        raise ValueError(f"eps_HR={eps} outside [0, 1)")
    T_amb = T_AMB_C if T_amb_C is None else float(T_amb_C)
    dT = max(0.0, T_HTC_C - T_amb)

    Q_gross = _mcp_MJ_per_K(m_slurry_kg) * dT
    Q_rec = eps * Q_gross
    Q_rxn = max(0.0, DH_RXN_HTC_MJ_PER_KG_DRY) * m_dry_kg
    Q_net = max(0.0, Q_gross - Q_rec - Q_rxn)
    Q_util_kWh = Q_net / (1.0 - F_LOSS_HTC) / ETA_HTC_UTILITY / 3.6
    # Balanced streams (product mass ~ feed mass, gas < 2 %): the product cools
    # by the same dT as the feed is preheated.
    T_prod1 = T_HTC_C - eps * dT
    # T_amb_C and T_feed_C hold the same value (feed inlet temperature).
    return dict(T_HTC_C=float(T_HTC_C),
                eps_HR=eps, T_amb_C=T_amb, T_feed_C=T_amb, T_prod1_C=T_prod1,
                Q_HTC_gross_MJ=Q_gross, Q_HTC_rec_MJ=Q_rec, Q_HTC_rxn_MJ=Q_rxn,
                Q_HTC_net_MJ=Q_net, Q_HTC_util_kWh=Q_util_kWh)


# =============================================================================
# MECHANICAL DEWATERING
# =============================================================================
def effective_MC_char(MC_char):
    """Moisture of the wet hydrochar leaving the filter press: MC_char (None:
    MC_CHAR_CENTRAL)."""
    mc = MC_CHAR_CENTRAL if MC_char is None else float(MC_char)
    if not 0.0 < mc < 1.0:
        raise ValueError(f"MC_char={mc} outside (0, 1)")
    return mc


def press_duty_kWh(m_dry_char_kg):
    """Filter press electricity (kWh) for m_dry_char_kg of dry solids [6]."""
    return PRESS_KWH_PER_T_DS * max(0.0, m_dry_char_kg) / 1000.0


# =============================================================================
# DRYER
# =============================================================================
def dryer_duty(m_water_evap_kg, m_prod_kg=0.0, T_prod1_C=None, m_dry_char_kg=0.0):
    """
    Belt dryer heat and electricity for m_water_evap_kg evaporated [10], plus
    the filter press electricity. Q_dry_util_MJ is supplied by electricity in
    system_builder (1 kWh per 3.6 MJ); the dryer takes no heat from the
    product, so T_prod2_C = T_prod1_C and Q_dry_rec_MJ = 0.
    """
    m_w = max(0.0, float(m_water_evap_kg))
    Q_th = m_w * DRYER_SEC_TH_MJ_PER_KG
    fan = m_w * DRYER_FAN_KWH_PER_KG
    cond_m3 = m_w / 1000.0 if DRYER_CONDENSATE_TO_WWT else 0.0
    E_press = press_duty_kWh(m_dry_char_kg)
    return dict(Q_dry_th_MJ=Q_th, Q_dry_rec_MJ=0.0, Q_dry_util_MJ=Q_th,
                E_dry_fan_kWh=fan, E_press_kWh=E_press, E_dry_elec_kWh=fan + E_press,
                condensate_m3=cond_m3, T_prod2_C=T_prod1_C)


# =============================================================================
# PRESSURE LETDOWN
# =============================================================================
def pw_temperature_after_letdown(T_prod2_C):
    """
    Temperature of the process water leaving the filter press (Pathway A
    digester feed). The product is expanded to about 1 atm before the press
    [7], so liquid water above T_LETDOWN_C flashes; that heat is not recovered.
    With the E2 cooler, the process water leaves at T_SEP_C.
    Example: 136 C -> 100 C; 90 C -> 90 C.
    """
    if T_prod2_C is None:
        return None
    return min(float(T_prod2_C), T_SEP_C, T_LETDOWN_C)


# =============================================================================
# MATCH 2: DIGESTER
# =============================================================================
def mix_temperature(streams):
    """Adiabatic mixing temperature of [(mass_kg, T_C), ...], equal cp."""
    m = sum(max(0.0, a) for a, _ in streams)
    if m <= 0:
        return T_AMB_C
    return sum(max(0.0, a) * t for a, t in streams) / m


def residual_heat_to_ad_MJ(m_prod_kg, T_from_C):
    """Pathway B: product heat recoverable into the digester feed through an
    indirect exchanger, down to T_DIG_C + DT_MIN_K. The exchanger sits on the
    still pressurized product, before the letdown, so no 100 C cap here."""
    if T_from_C is None:
        return 0.0
    floor = max(T_DIG_C + DT_MIN_K, T_SEP_C)      # E2 stops at T_SEP_C
    return _mcp_MJ_per_K(m_prod_kg) * max(0.0, T_from_C - floor)


# =============================================================================
# PRODUCT COOLER E2, PATHWAY A DILUTION WATER, PROCESS WATER TRIM COOLER
# =============================================================================
def product_cooler_duty(m_prod_kg, T_in_C):
    """
    E2: the product (still pressurized) is cooled from T_in_C (= T_prod2, which
    equals T_prod1) to T_SEP_C, before the
    letdown and the filter press.
    Example: 7140 kg from 97.3 C to 70 C -> 7140 x 0.00418 x 27.3 = 815 MJ.
    """
    if T_in_C is None:
        return dict(Q_E2_MJ=0.0, T_SEP_C=None)
    T_in = float(T_in_C)
    T_out = min(T_in, T_SEP_C)
    return dict(Q_E2_MJ=_mcp_MJ_per_K(m_prod_kg) * (T_in - T_out), T_SEP_C=T_out)


def e2_heat_to_dilution_MJ(Q_E2_MJ, m_dil_kg):
    """
    Pathway A: E2 heat available to preheat the dilution water before it is
    mixed with the process water, capped so that the dilution water leaves E2
    at most at T_SEP_C - DT_MIN_K. ad_duty then uses only what the digester
    needs; the rest is rejected.
    Example: 24938 kg of dilution water -> cap = 24938 x 0.00418 x 50 = 5212 MJ.
    """
    if m_dil_kg is None or m_dil_kg <= 0:
        return 0.0
    cap = _mcp_MJ_per_K(m_dil_kg) * max(0.0, T_SEP_C - DT_MIN_K - dilution_temperature())
    return min(max(0.0, float(Q_E2_MJ)), cap)


def trim_cooler_outlet_T(m_PW_kg, T_PW_C, m_dil_kg, T_dil_C, T_in_req_C):
    """
    Pathway A: temperature the process water leaving the filter press must be
    cooled to (before mixing with the dilution water) so the mixed inflow
    enters the digester at T_in_req_C. Returns T_PW_C if no cooling is needed.
    Example: 21976 kg PW at 70 C + 9044 kg at 10 C, T_in_req 40 C -> 52.3 C.
    """
    if T_PW_C is None or T_in_req_C is None or not m_PW_kg or m_PW_kg <= 0:
        return T_PW_C
    m_dil = max(0.0, m_dil_kg or 0.0)
    T_out = (T_in_req_C * (m_PW_kg + m_dil) - m_dil * T_dil_C) / m_PW_kg
    return min(float(T_PW_C), T_out)


def digester_wall_loss(V_in_m3_per_FU, HRT_days, dry_FU_kg,
                       capacity_t_DM_per_day=None):
    """
    Digester heat loss through wall, roof and floor, MJ per FU, Metcalf & Eddy
    method [5]: Q = sum(U_i x A_i) x (T_DIG_C - T_AMB_C). Not used in the
    heat balance: it is the check behind the default loss fraction F_LOSS_AD
    (see tests/test_heat_integration_wall_loss.py).

    Step by step (all per PLANT, then divided by FU per day):
      n_FU_per_day = capacity (t DM/day) x 1000 / dry_FU_kg
      V_tank  = V_in_m3_per_FU x n_FU_per_day x HRT_days        (m3, CSTR)
      D       = (4 V_tank / (pi x H/D))^(1/3),   H = (H/D) x D
      A_wall  = pi D H,  A_roof = A_floor = pi D^2 / 4
      UA      = U_wall A_wall + U_roof A_roof + U_floor A_floor   (W/K)
      Q_plant = UA x (T_DIG_C - T_AMB_C) x 86400 s/day / 1e6     (MJ/day)
      Q_per_FU = Q_plant / n_FU_per_day

    V_in_m3_per_FU: digester inflow volume per FU (feed + dilution water).
    dry_FU_kg: dry matter per FU (kg), used to convert the capacity to FU per
    day (150 kg for 1 t of food waste at 85 % moisture).

    Returns a dict (Q_loss_MJ per FU, V_tank_m3, A_m2, capacity used).
    """
    if not dry_FU_kg or dry_FU_kg <= 0:
        raise ValueError(f"digester_wall_loss: dry_FU_kg = {dry_FU_kg} must be > 0")
    cap = PLANT_CAPACITY_T_DM_PER_DAY if capacity_t_DM_per_day is None else float(capacity_t_DM_per_day)
    if cap <= 0:
        raise ValueError(f"plant capacity {cap} t DM/day must be > 0")
    n_FU_per_day = cap * 1000.0 / float(dry_FU_kg)
    V_tank = max(0.0, float(V_in_m3_per_FU)) * n_FU_per_day * max(0.0, float(HRT_days))
    if V_tank <= 0:
        return dict(Q_loss_MJ=0.0, V_tank_m3=0.0, A_m2=0.0, capacity_t_DM_d=cap)
    r = DIGESTER_H_OVER_D
    D = (4.0 * V_tank / (math.pi * r)) ** (1.0 / 3.0)
    H = r * D
    A_wall = math.pi * D * H
    A_roof = A_floor = math.pi * D * D / 4.0
    UA_W_per_K = U_WALL_W_M2K * A_wall + U_ROOF_W_M2K * A_roof + U_FLOOR_W_M2K * A_floor
    Q_plant_MJ_per_day = UA_W_per_K * max(0.0, T_DIG_C - T_AMB_C) * 86400.0 / 1e6
    return dict(Q_loss_MJ=Q_plant_MJ_per_day / n_FU_per_day, V_tank_m3=V_tank,
                A_m2=A_wall + A_roof + A_floor, capacity_t_DM_d=cap)


def ad_duty(m_in_kg, T_in_C=None, Q_rec_avail_MJ=0.0):
    """
    Digester heat per m_in_kg of inflow (feed + dilution water).
      sensible: m cp (T_DIG - T_in)          (negative if the feed is hot)
      losses:   F_LOSS_AD x m cp (T_DIG - T_amb)
      utility:  max(0, sensible + losses - recovered)
    A hot inflow (Pathway A process water) first covers the losses; any
    surplus is removed by the trim cooler and rejected without burden.
    Q_AD_util_MJ is supplied by an ecoinvent heat dataset (delivered heat), so
    no boiler efficiency is applied here.
    """
    T_in = T_AMB_C if T_in_C is None else float(T_in_C)
    mcp = _mcp_MJ_per_K(m_in_kg)
    Q_sens = mcp * (T_DIG_C - T_in)
    Q_loss = F_LOSS_AD * mcp * max(0.0, T_DIG_C - T_AMB_C)

    Q_dem = Q_sens + Q_loss
    Q_rec = min(max(Q_dem, 0.0), max(0.0, Q_rec_avail_MJ))
    # Hot inflow (Pathway A): a digester fed above T_in_req would drift above
    # T_DIG_C, so the excess is removed by the trim cooler (rejected, no
    # burden) and the inflow enters at T_in_req, where its sensible heat just
    # covers the wall loss. Q_PW_trim_MJ and Q_AD_cool_MJ hold the same heat.
    # T_AD_inlet_C = inflow temperature after trim cooling, or the equivalent
    # inflow temperature after the recovered heat (E2) is added.
    T_in_req = T_DIG_C + (Q_loss / mcp if mcp > 0 else 0.0)
    Q_trim = max(0.0, -Q_dem)
    if Q_trim > 0:
        T_inlet = T_in_req
    else:
        T_inlet = T_in + (Q_rec / mcp if mcp > 0 else 0.0)
    return dict(T_AD_in_C=T_in, Q_AD_dem_MJ=Q_dem, Q_AD_rec_MJ=Q_rec,
                Q_AD_util_MJ=max(0.0, Q_dem - Q_rec), Q_AD_cool_MJ=max(0.0, -Q_dem),
                Q_AD_loss_MJ=Q_loss, Q_PW_trim_MJ=Q_trim, T_AD_inlet_C=T_inlet,
                T_AD_in_req_C=T_in_req)


# =============================================================================
# DIAGNOSTICS (columns added to the results CSV)
# =============================================================================
HEAT_FIELDS = [
    "eps_HR_used", "T_amb_C", "T_prod1_C", "T_prod2_C",
    "Q_HTC_gross_MJ", "Q_HTC_rec_MJ", "Q_HTC_rxn_MJ", "Q_HTC_net_MJ",
    "Q_dry_th_MJ", "Q_dry_rec_MJ", "Q_dry_util_MJ", "E_dry_fan_kWh",
    "MC_char_used", "E_press_kWh",
    "T_AD_in_C", "Q_AD_dem_MJ", "Q_AD_rec_MJ", "Q_AD_util_MJ", "Q_AD_cool_MJ",
    "T_feed_HTC_C",            # HTC feed inlet temperature
    "T_PW_to_AD_C",            # Pathway A: PW leaving the filter press, after letdown
    "Q_AD_loss_MJ",            # digester wall/roof/floor loss per FU
    "T_SEP_C",                 # separation temperature (product leaving E2)
    "Q_E2_MJ",                 # heat released by the product cooler E2
    "Q_E2_to_AD_MJ",           # E2 heat used on the digester side
    "Q_E2_rejected_MJ",        # E2 heat rejected to ambient (no burden)
    "Q_PW_trim_MJ",            # A: PW trim cooler duty, rejected (no burden)
    "T_PW_trim_out_C",         # A: PW temperature after the trim cooler
    "T_AD_inlet_C",            # digester inflow T after trim / E2 heat (equivalent)
    "Q_surplus_rejected_MJ",   # Q_E2_rejected + Q_PW_trim
    "P_surplus_kW_per_FU_d",   # surplus divided by 86.4 (kW at one FU per day)
    "Q_PW_to_WWT_MJ",          # heat above T_AMB lost with PW sent to WWT (B, HTC-only)
    "energy_closure_residual_MJ",  # must be ~0, see metadata()
]
_EXTENSIVE_HTC = ["Q_HTC_gross_MJ", "Q_HTC_rec_MJ", "Q_HTC_rxn_MJ", "Q_HTC_net_MJ",
                  "Q_dry_th_MJ", "Q_dry_rec_MJ", "Q_dry_util_MJ", "E_dry_fan_kWh",
                  "E_press_kWh"]
_AD_KEYS = ["T_AD_in_C", "Q_AD_dem_MJ", "Q_AD_rec_MJ", "Q_AD_util_MJ", "Q_AD_cool_MJ",
            "Q_AD_loss_MJ"]


def metadata(mb=None, ad=None, scale=1.0, extra=None):
    """
    Flat dict of HEAT_FIELDS for waste_treatment / the CSV.
    mb:    compute_htc_mass_balance() output (holds 'heat_htc', 'heat_dry'),
           UNSCALED for Pathway A -> pass scale=_scale_A.
    ad:    ad_duty() output, already per FU.
    extra: other HEAT_FIELDS computed by the caller, e.g.
           {"T_PW_to_AD_C": ...}. Unknown keys are rejected.
    """
    out = {k: None for k in HEAT_FIELDS}
    if mb is not None:
        h, d = mb.get("heat_htc", {}), mb.get("heat_dry", {})
        out["eps_HR_used"] = h.get("eps_HR")
        out["T_amb_C"] = h.get("T_amb_C")
        out["T_feed_HTC_C"] = h.get("T_feed_C", h.get("T_amb_C"))
        out["T_prod1_C"] = h.get("T_prod1_C")
        out["T_prod2_C"] = d.get("T_prod2_C")
        out["MC_char_used"] = mb.get("MC_char_used")
        for k in _EXTENSIVE_HTC:
            v = h.get(k, d.get(k))
            out[k] = None if v is None else float(v) * scale
        e2 = mb.get("heat_E2") or {}
        out["T_SEP_C"] = e2.get("T_SEP_C")
        if e2.get("Q_E2_MJ") is not None:
            out["Q_E2_MJ"] = float(e2["Q_E2_MJ"]) * scale
    if ad is not None:
        for k in _AD_KEYS:
            v = ad.get(k)
            out[k] = None if v is None else float(v)
        out["Q_PW_trim_MJ"] = ad.get("Q_PW_trim_MJ")
        out["T_AD_inlet_C"] = ad.get("T_AD_inlet_C")
    if extra:
        bad = set(extra) - set(HEAT_FIELDS)
        if bad:
            raise KeyError(f"metadata(extra=...): not in HEAT_FIELDS: {sorted(bad)}")
        for k, v in extra.items():
            out[k] = None if v is None else (float(v) if isinstance(v, (int, float)) else v)
    # derived surplus and energy closure terms
    Q_E2 = float(out["Q_E2_MJ"] or 0.0)
    to_ad = out["Q_E2_to_AD_MJ"]
    if to_ad is None:
        # A and B: the heat ad_duty recovered IS the E2 heat delivered.
        # HTC-only (ad None) and AD-only (mb None): nothing delivered.
        to_ad = float(ad.get("Q_AD_rec_MJ") or 0.0) if (ad is not None and mb is not None) else 0.0
        out["Q_E2_to_AD_MJ"] = to_ad
    out["Q_E2_rejected_MJ"] = max(0.0, Q_E2 - to_ad)
    Q_sur = out["Q_E2_rejected_MJ"] + float(out["Q_PW_trim_MJ"] or 0.0)
    out["Q_surplus_rejected_MJ"] = Q_sur
    # MJ per FU / 86.4 = kW at a throughput of one FU per day.
    out["P_surplus_kW_per_FU_d"] = Q_sur / 86.4
    # Closure: (1) E2 cannot deliver more than it releases; (2) product heat
    # from T_HTC to T_SEP = E1 recovery + E2 (HTC side).
    res = min(0.0, Q_E2 - to_ad)
    if mb is not None:
        h, d = mb.get("heat_htc", {}), mb.get("heat_dry", {})
        T_htc, T_sep = h.get("T_HTC_C"), out["T_SEP_C"]
        T_f = h.get("T_feed_C", h.get("T_amb_C"))
        if None not in (T_htc, T_sep, T_f) and T_htc > T_f and h.get("Q_HTC_gross_MJ"):
            mcp = float(h["Q_HTC_gross_MJ"]) / (T_htc - T_f) * scale
            res += mcp * (T_htc - T_sep) - (
                (float(h.get("Q_HTC_rec_MJ") or 0.0) + float(d.get("Q_dry_rec_MJ") or 0.0)) * scale + Q_E2)
    out["energy_closure_residual_MJ"] = res
    return out
