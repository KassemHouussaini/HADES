"""Regression: Base scenario, food waste (T 180 C, RT 60 min, HRT 30 d, COD
target 10 kg/m3, TS 8 %, MC_char 0.35), with the HTC surrogate in
htc_xgb_models/. Runs the whole build_system with a mock Brightway
(dryrun_mock_bw.py, about 40 s per pathway). A failure means the process
model changed: update the frozen values only on purpose."""
import json, os, subprocess, sys
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ADM1_DIR = os.path.dirname(HERE)
SCRIPT = os.path.join(HERE, "dryrun_mock_bw.py")

FROZEN = {
    "A": {"MJ_hydrochar": 1365.1333882001034, "CH4_mass_AD": 5.197704050762951,
          "CO2_mass_AD": 9.60387207573531, "PW_yield": 832.9079321289062,
          "HC_dry_yield": 116.95093790690107, "Q_heat_AD_MJ": 68.50894596491034},
    "B": {"MJ_hydrochar": 483.3086313949063, "CH4_mass_AD": 28.089161013219076,
          "CO2_mass_AD": 60.039970230900344, "PW_yield": 283.66817730534603,
          "HC_dry_yield": 40.59113340156826, "Q_heat_AD_MJ": 195.85549891520893},
}


@pytest.mark.slow
@pytest.mark.parametrize("pathway", ["A", "B"])
def test_base_case_frozen(pathway, tmp_path):
    out = tmp_path / f"base_{pathway}.json"
    r = subprocess.run([sys.executable, SCRIPT, ADM1_DIR, pathway, str(out), "0.35", "htc_surrogate.py"],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    meta = json.load(open(out))["meta"]
    for k, v in FROZEN[pathway].items():
        assert meta[k] == pytest.approx(v, rel=1e-5), k
    # Base-A at 180 C: char H/C and O/C just above the training 99th percentile; flagged, not fatal
    assert ("HC_atomic" in meta["qa_flags"]) == (pathway == "A")


@pytest.mark.slow
@pytest.mark.parametrize("pathway", ["A", "B"])
def test_batch_mode_blank_corrected_per_fu(pathway, tmp_path):
    """ADM1 batch mode (BMP test): blank-corrected methane, scaled from the
    bottle to the FU. Pathway A: the bottle holds COD_conc x V_liq of
    substrate, more than the process water of the FU."""
    out = tmp_path / f"batch_{pathway}.json"
    r = subprocess.run([sys.executable, SCRIPT, ADM1_DIR, pathway, str(out), "0.35", "htc_surrogate.py"],
                       capture_output=True, text=True, timeout=900, env={**os.environ, "HADES_MODE": "batch"})
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    meta = json.load(open(out))["meta"]
    assert meta["adm1_mode"] == "batch"
    if pathway == "A":
        assert meta["CH4_mass_AD"] == pytest.approx(4.6918, rel=1e-3)
        assert meta["CH4_mass_AD"] < FROZEN["A"]["CH4_mass_AD"]    # 30-d BMP below continuous at 30 d HRT
        cod_fed = meta.get("pw_COD_fed_kg")
        if cod_fed:
            assert meta["CH4_mass_AD"] / 0.716 / cod_fed * 1000 < 350
