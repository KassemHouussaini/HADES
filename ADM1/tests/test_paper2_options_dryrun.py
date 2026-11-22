"""Paper 2 options of HADES.ipynb run end to end (mock Brightway): LCA context
preset, heat supply, Pathway B degradability override, Pathway A process-water
methane multiplier. Checks that each option reaches the model, not exact
values (those are covered by the frozen base case)."""
import json, os, subprocess, sys
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ADM1_DIR = os.path.dirname(HERE)
SCRIPT = os.path.join(HERE, "dryrun_mock_bw.py")
BASE_B = {"CH4_mass_AD": 24.402600316096553, "MJ_hydrochar": 583.2053071249042}
BASE_A = {"CH4_mass_AD": 5.197704050762951}


def _run(pathway, out, **env):
    e = {**os.environ, **env}
    r = subprocess.run([sys.executable, SCRIPT, ADM1_DIR, pathway, str(out), "0.35", "htc_surrogate.py"],
                       capture_output=True, text=True, timeout=900, env=e)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return json.load(open(out))


@pytest.mark.slow
def test_pathway_b_options(tmp_path):
    d = _run("B", tmp_path / "b.json", HADES_CONTEXT="ON", HADES_B_FEED='{"fD": 0.80}',
             HADES_HEAT_SUPPLY="htc=ng,dryer=ng,ad=own_biogas")
    m = d["meta"]
    assert m["adm1_fD"] == pytest.approx(0.80)
    assert m["CH4_mass_AD"] < BASE_B["CH4_mass_AD"]           # less degradable feed
    assert m["MJ_hydrochar"] > BASE_B["MJ_hydrochar"]         # more solids left for HTC
    assert m["CH4_own_use_MJ"] > 0 and m["heat_shortfall_ad_MJ"] == 0
    mix = {e["input"]: e["amount"] for e in d["exchanges"]["qc_heat_mix"]}
    assert sum(mix.values()) == pytest.approx(1.0, rel=1e-9)
    assert any("coal coke" in k for k in mix)                 # Ontario steel-industry coke
    ng = {e["input"]: e["amount"] for e in d["exchanges"]["heat_ng"]}   # biomethane credit
    assert len(ng) == 4 and sum(ng.values()) == pytest.approx(1.0, rel=1e-12)
    assert all("natural gas, at" in k for k in ng)          # boilers and furnaces, no CHP


@pytest.mark.slow
def test_pathway_a_options(tmp_path):
    d = _run("A", tmp_path / "a.json", HADES_CONTEXT="AB", HADES_PW_M="1.2",
             HADES_HEAT_SUPPLY="htc=own_biogas,dryer=ng,ad=own_biogas")
    m = d["meta"]
    assert m["pw_ch4_multiplier_effective"] == pytest.approx(1.2)
    assert m["CH4_mass_AD"] == pytest.approx(1.2 * BASE_A["CH4_mass_AD"], rel=1e-6)
    assert m["biomethane_m3"] == pytest.approx(0.0, abs=1e-9)   # all biogas used on site
    assert m["heat_shortfall_htc_MJ"] > 0
    mix = {e["input"]: e["amount"] for e in d["exchanges"]["qc_heat_mix"]}
    assert not any("coal" in k for k in mix)                  # Alberta: no coal or coke in 2023
