"""Cap recorder of ADM1.py: bound() gives the same value as np.clip and
records only the bounds that bind."""
import numpy as np
import pytest


def test_bound_equals_clip_and_records_only_binding(adm1):
    adm1.reset_caps()
    for x in (-1.0, 0.2, 0.5, 0.9):
        assert adm1.bound("t", x, 0.0, 0.55) == float(np.clip(x, 0.0, 0.55))
    rep = adm1.caps_report()
    assert rep["t"]["n"] == 2 and rep["t"]["raw"] == -1.0 and rep["t"]["used"] == 0.0
    adm1.reset_caps()
    adm1.bound("t", 0.3, 0.0, 0.55)
    assert adm1.caps_report() == {}


def test_protein_cap_recorded(adm1):
    adm1.reset_caps()
    # N-rich substrate: (N / organic) / 0.16 = 0.625 > 0.55
    adm1.derive_adm1_fractions(C=40, H=6, O=34, N=10, S=0, substrate_type="feedstock")
    assert "protein fraction (feedstock) <= 0.55" in adm1.caps_report()


def test_feedstock_S_IN_floor_is_design_not_cap(adm1):
    adm1.reset_caps()
    adm1.compute_S_IN(5.0, 1.0, "feedstock")          # f_IN = 0: floor 1e-4 by design
    assert adm1.caps_report() == {}


def test_S_IN_upper_cap_recorded(adm1):
    adm1.reset_caps()
    adm1.compute_S_IN(500.0, 1.0, "htc_pw", T_HTC=250, RT_HTC=120)
    assert any(k.startswith("S_IN (htc_pw) <= 0.13") for k in adm1.caps_report())


def test_format_caps(sb):
    txt = sb.format_caps({"x": dict(raw=1.0, used=0.5, kind="correlation", n=3)})
    assert txt == "x (correlation): 1 -> 0.5 [x3]"
