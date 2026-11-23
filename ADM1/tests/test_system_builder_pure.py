"""Pure functions of system_builder (no Brightway calls)."""
import pytest


# ---------------------------------------------------------------- HTC mass balance
def test_htc_mass_balance_closes(sb):
    mb = sb.compute_htc_mass_balance(1000.0, 0.85, 67.26, 180.0, 0.35, 0.10)
    out = mb["HC_wet_yield"] + mb["CO2_yield"] + mb["CO_yield"] + mb["PW_yield"]
    assert out == pytest.approx(1000.0, rel=1e-12)
    assert mb["HC_dry_yield"] * (1 - 0.10) == pytest.approx(mb["HC_dry_mass"])
    assert mb["HC_dry_mass"] == pytest.approx(150.0 * 0.6726)
    # cake at the filter press moisture
    water = mb["HC_wet_yield"] - mb["HC_dry_mass"]
    assert water / mb["HC_wet_yield"] == pytest.approx(0.35)


def test_gas_yield_correlation_at_180C(sb):
    mb = sb.compute_htc_mass_balance(1000.0, 0.85, 60.0, 180.0, 0.35, 0.10)
    assert mb["Y_gas_drybasis"] == pytest.approx(0.121 * 180 - 18.34)      # 3.44 %


def test_heat_recovery_lowers_htc_duty(sb):
    on = sb.compute_htc_mass_balance(1000.0, 0.85, 67.26, 180.0, 0.35, 0.10)
    off = sb.compute_htc_mass_balance(1000.0, 0.85, 67.26, 180.0, 0.35, 0.10, eps_HR=0.0)
    assert 0 < on["Q_heat_kWh"] < off["Q_heat_kWh"]
    # no recovery: electric duty = m cp (T_HTC - T_feed) / eta
    assert off["Q_heat_kWh"] == pytest.approx(1000.0 * 4.18 * (180 - 10) / 3600 / 0.9, rel=1e-9)
    assert on["T_prod1_C"] < 180.0      # heat only flows hot -> cold


# ---------------------------------------------------------------- carbon recovery
def test_carbon_recovery_reported_and_yield_unchanged(sb):
    def fake_ml(C_feed, T):
        return None, {"Y_char (%)": 60.0, "C_char (%)": 45.0}
    _, inv = sb.run_ml_model(fake_ml, {"C_feed": 50.0, "T": 180})
    assert inv["C_recovery_raw"] == pytest.approx(0.54)      # 0.60 x 45 / 50
    assert inv["Y_char (%)"] == 60.0


def test_run_ml_model_drops_unknown_arguments(sb):
    def fake_ml(C_feed, T):
        return None, {"Y_char (%)": 60.0, "C_char (%)": 45.0}
    _, inv = sb.run_ml_model(fake_ml, {"C_feed": 50.0, "T": 180, "not_an_input": 1})
    assert inv["Y_char (%)"] == 60.0


# ---------------------------------------------------------------- centrifuge N split
def test_centrifuge_split_mass_and_n(sb):
    m_sol, m_liq, dry, N_liq = sb.split_centrifuge_N(1787.0, 53.0, 0.85, 0.08, 5.5)
    assert m_sol + m_liq == pytest.approx(1787.0, rel=1e-12)
    assert m_sol == pytest.approx(dry / 0.15)
    assert dry == pytest.approx(53.0 - N_liq)
    assert N_liq == pytest.approx(0.08 * 14.0 * m_liq / 1000.0, rel=1e-6)
    assert 0 <= N_liq <= 5.5


def test_centrifuge_no_ammonium_limit(sb):
    m_sol, m_liq, dry, N_liq = sb.split_centrifuge_N(1787.0, 53.0, 0.85, 0.0, 5.5)
    assert N_liq == 0.0 and m_sol == pytest.approx(53.0 / 0.15)


def test_centrifuge_cap_at_digestate_n(sb):
    _, _, _, N_liq = sb.split_centrifuge_N(1787.0, 53.0, 0.85, 10.0, 0.5)
    assert N_liq == pytest.approx(0.5)


# ---------------------------------------------------------------- dried basis
DP = {"_C_dig_kg": 18.3, "_H_dig_kg": 2.3, "_O_dig_kg": 12.7, "_N_dig_kg": 3.8,
      "_S_dig_kg": 0.75, "_Ash_dig_kg": 13.65, "_fc_ratio": 0.206}


def test_dried_basis_removes_ammonium(sb):
    out = sb.dried_basis_for_htc(DP, NH4_kg=0.3, m_wet_kg=342.0)
    tot = sum(out[f"{e}_feed"] for e in ("C", "H", "O", "N", "S", "Ash"))
    assert tot == pytest.approx(100.0, rel=1e-12)
    assert out["_N_org_htc_kg"] == pytest.approx(3.5)
    assert out["_m_dry_htc_kg"] == pytest.approx(51.2)
    assert out["WC"] == pytest.approx(100 * (1 - 51.2 / 342.0))
    assert out["FC_feed"] + out["VM_feed"] + out["Ash_feed"] == pytest.approx(100.0)


def test_dried_basis_rejects_excess_ammonium(sb):
    with pytest.raises(ValueError):
        sb.dried_basis_for_htc(DP, NH4_kg=5.0, m_wet_kg=342.0)


# ---------------------------------------------------------------- QA checks
def _mb(sb):
    return sb.compute_htc_mass_balance(1000.0, 0.85, 67.26, 180.0, 0.35, 0.10)


INV_OK = {"Y_char (%)": 67.3, "HHV_char (MJ/kg)": 17.5, "C_char (%)": 41.1, "H_char (%)": 5.0,
          "O_char (%)": 45.0, "N_char (%)": 1.6, "S_char (%)": 0.3, "Ash_char (%)": 6.3}


def test_qa_passes_reference_point(sb):
    flags = sb.qa_checks("A", INV_OK, _mb(sb), None, CH4_mass=5.03, CO2_mass=9.98,
                         MB=[0.0, 0.0], m_ref=1000.0)
    assert flags == []


def test_qa_mass_balance_raises(sb):
    with pytest.raises(RuntimeError):
        sb.qa_checks("A", INV_OK, _mb(sb), None, 5.0, 10.0, MB=[0.5], m_ref=1000.0)


def test_qa_flags_biogas_and_char(sb):
    bad = {**INV_OK, "HHV_char (MJ/kg)": 40.0}
    flags = sb.qa_checks("A", bad, _mb(sb), None, CH4_mass=1.0, CO2_mass=20.0, MB=[0.0], m_ref=1000.0)
    assert any("HHV_char" in f for f in flags)
    assert any("biogas CH4" in f for f in flags)


def test_lhv_and_end_use_present_in_source(sb):
    import inspect
    src = inspect.getsource(sb._build_system)
    assert "LHV_biomethane = 35.8" in src
    assert "build_hydrochar_enduse_exchanges(ctx, q_enduse, bios)" in src
    psa = inspect.getsource(sb.build_biogas_upgrading_exchanges)
    assert 'startswith("methane, non-fossil")' in psa


def test_hi_overrides_set_and_restore(sb):
    hi = sb.hi
    before = {k: getattr(hi, k) for k in sb.SWEEP_HI_PARAMS}
    with sb._HiOverrides({"F_LOSS_AD": 0.3, "DRYER_SEC_TH_MJ_PER_KG": 1.8}):
        assert hi.F_LOSS_AD == 0.3 and hi.DRYER_SEC_TH_MJ_PER_KG == 1.8
        assert hi.dryer_duty(10.0)["Q_dry_th_MJ"] == pytest.approx(18.0)       # 10 kg water x 1.8 MJ/kg
    assert {k: getattr(hi, k) for k in sb.SWEEP_HI_PARAMS} == before          # restored
    with pytest.raises(ValueError):
        with sb._HiOverrides({"F_LOSS_AD": 0.9}):                              # outside (0.1, 0.3)
            pass
    with pytest.raises(ValueError):
        with sb._HiOverrides({"ETA_HTC_UTILITY": 0.8}):                        # not a sweep parameter
            pass
    assert {k: getattr(hi, k) for k in sb.SWEEP_HI_PARAMS} == before
    for k, r in sb.SWEEP_HI_PARAMS.items():                                    # every default inside its range
        lo, up = getattr(hi, r)
        assert lo <= getattr(hi, k) <= up, k
