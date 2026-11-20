"""The quasi-steady H2 balance of ADM1 uses the same sub-pool inhibition as the ODE."""
import pytest


def _sv(adm1, si=0.0):
    sv = dict(adm1.BSM2_INIT) if hasattr(adm1, "BSM2_INIT") else {}
    for k in adm1.STATE_NAMES:
        sv.setdefault(k, 0.0)
    base = {"S_su": 0.012, "S_aa": 0.0055, "S_fa": 0.107, "S_va": 0.0123, "S_bu": 0.014,
            "S_pro": 0.0176, "S_ac": 0.089, "S_h2": 2.5e-7, "S_IC": 0.095, "S_IN": 0.094,
            "X_su": 0.31, "X_aa": 0.93, "X_fa": 0.34, "X_c4": 0.34, "X_pro": 0.10,
            "X_ac": 0.68, "X_h2": 0.28, "S_cation": 0.052, "S_anion": 0.0052,
            "S_H_ion": 5.46e-8, "S_gas_h2": 1.1e-5}
    sv.update(base)
    for sp in adm1.SI_SUBPOOL_NAMES:
        sv[f"S_I_{sp}"] = si
    return sv


def _c(adm1):
    return adm1._build_constants()


def test_no_subpools_unchanged(adm1, monkeypatch):
    c = _c(adm1)
    if c is None:
        pytest.skip("constants accessor not found")
    sv = _sv(adm1, 0.0)
    monkeypatch.setattr(adm1, "H2_TOX_CONSISTENT", True)
    a = adm1._dae_solve(dict(sv), c)
    monkeypatch.setattr(adm1, "H2_TOX_CONSISTENT", False)
    b = adm1._dae_solve(dict(sv), c)
    assert a["S_h2"] == pytest.approx(b["S_h2"], rel=1e-12)   # Pathway B unaffected


def test_inhibition_raises_dissolved_h2(adm1, monkeypatch):
    c = _c(adm1)
    if c is None:
        pytest.skip("constants accessor not found")
    sv = _sv(adm1, 5.0)
    monkeypatch.setattr(adm1, "H2_TOX_CONSISTENT", True)
    a = adm1._dae_solve(dict(sv), c)
    monkeypatch.setattr(adm1, "H2_TOX_CONSISTENT", False)
    b = adm1._dae_solve(dict(sv), c)
    # H2 uptake (R12) inhibited -> higher S_h2 at the quasi-steady state
    assert a["S_h2"] > b["S_h2"] > 0
