"""Digester wall loss (heat_integration.digester_wall_loss, Metcalf & Eddy U x A x dT)."""
import pytest


def test_wall_loss_requires_dry_mass_per_fu(sb):
    with pytest.raises(TypeError):
        sb.hi.digester_wall_loss(3.0, 30)
    with pytest.raises(ValueError):
        sb.hi.digester_wall_loss(3.0, 30, dry_FU_kg=0)


def test_wall_loss_fraction_supports_f_loss_ad(sb):
    # 1 t food waste at 85 % moisture (150 kg DM per FU), 10 t DM/day plant, 30 d HRT.
    # Pathway B inflow at 8 % TS is 150 / 0.08 = 1.875 m3 per FU. The loss as a share
    # of the sensible heat from T_AMB_C to T_DIG_C must fall inside F_LOSS_AD_RANGE.
    hi = sb.hi
    m = 1.875
    wl = hi.digester_wall_loss(m, 30, dry_FU_kg=150.0, capacity_t_DM_per_day=10.0)
    frac = wl["Q_loss_MJ"] / (m * 1000 * 4.18e-3 * (hi.T_DIG_C - hi.T_AMB_C))
    lo, up = hi.F_LOSS_AD_RANGE
    assert lo <= frac <= up


def test_wall_loss_zero_volume(sb):
    wl = sb.hi.digester_wall_loss(0.0, 30, dry_FU_kg=150.0, capacity_t_DM_per_day=10.0)
    assert wl["Q_loss_MJ"] == 0.0
