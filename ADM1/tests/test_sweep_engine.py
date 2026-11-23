"""sweep_engine: designs, parameter mapping, ADM1 cache with cap replay, and
an end-to-end in-memory run on the mock background (slow)."""
import json, os, subprocess, sys, types
import numpy as np
import pytest
from conftest import _load

HERE = os.path.dirname(os.path.abspath(__file__))
ADM1_DIR = os.path.dirname(HERE)


@pytest.fixture(scope="module")
def se():
    sys.modules.setdefault("bw2calc", types.SimpleNamespace(LCA=object))
    return _load("p2_sweep_engine_test", "sweep_engine.py")


def test_morris_design_one_factor_per_step(se):
    k, r, levels = 5, 7, 4
    X, steps = se.morris_design(k, r, levels, seed=1)
    assert X.shape == (r * (k + 1), k)
    assert X.min() >= 0 and X.max() <= 1 + 1e-12
    delta = levels / (2 * (levels - 1))
    for row, (t, s, i, d) in enumerate(steps):
        if i is None:
            continue
        diff = X[row] - X[row - 1]
        assert np.count_nonzero(np.abs(diff) > 1e-12) == 1 and diff[i] == pytest.approx(d)
        assert abs(d) == pytest.approx(delta)
    moved = [sorted(i for (t, s, i, d) in steps if t == tt and i is not None) for tt in range(r)]
    assert all(m == list(range(k)) for m in moved)        # every factor moves once per trajectory


def test_morris_indices_linear_model(se):
    b = np.array([3.0, -1.0, 0.0, 0.5])
    X, steps = se.morris_design(4, 10, 4, seed=2)
    mi = se.morris_indices(X @ b, steps, ["a", "b", "c", "d"]).set_index("factor")
    for n, v in zip("abcd", b):
        assert mi.loc[n, "mu_star"] == pytest.approx(abs(v))
        assert mi.loc[n, "mu"] == pytest.approx(v)
        assert mi.loc[n, "sigma"] == pytest.approx(0.0, abs=1e-9)


def test_lhs_strata(se):
    U = se.lhs_design(20, 3, seed=0)
    for j in range(3):                                    # one point per stratum
        assert sorted((U[:, j] * 20).astype(int)) == list(range(20))


def test_map_sample_ranges(se, sb):
    core, hce = sb._adm1_core, sb.hce
    space = se.parameter_space(sb, hce, core)
    names = [s["name"] for s in space]
    assert set(sb.SWEEP_HI_PARAMS) <= set(names)
    for u in (np.zeros(len(space)), np.ones(len(space))):
        prm, x = se.map_sample(u, space, "sewage_sludge", core)
        cls = core.PATHWAY_B_FEED_PARAMS["sewage_sludge"]
        end = 0 if u[0] == 0 else 1
        assert prm["b_feed_overrides"]["fD"] == pytest.approx(cls["fD_range"][end])
        assert prm["b_feed_overrides"]["f_sol"] == pytest.approx(cls["f_sol_range"][end])
        assert prm["pw_ch4_multiplier"] == pytest.approx(sb.PW_CH4_MULTIPLIER_RANGE[end])
        for k, r in sb.SWEEP_HI_PARAMS.items():
            assert prm["hi_overrides"][k] == pytest.approx(getattr(sb.hi, r)[end])
    prm, _ = se.map_sample(np.ones(len(space)), space, "dairy_manure", core)
    assert prm["b_feed_overrides"]["f_sol"] == core.PATHWAY_B_FEED_PARAMS["dairy_manure"]["f_sol"]  # no range


def test_adm1_memo_replays_caps(se):
    calls = []
    core = types.SimpleNamespace(_caps=[])
    core.record_cap = lambda name, raw, used, kind="correlation": core._caps.append((name, raw, used, kind))

    def fake_run_adm1(substrate, **kw):
        calls.append(1)
        core.record_cap("x <= 1", 2.0, 1.0)
        return {"CH4": substrate["cod"] * 2, "arr": np.arange(3)}

    sb = types.SimpleNamespace(_adm1_core=core, run_adm1=fake_run_adm1)
    memo = se.Adm1Memo(sb).install()
    a = sb.run_adm1(substrate={"cod": 1.5}, mode="continuous")
    a["CH4"] = -1                                          # caller mutates its copy
    b = sb.run_adm1(substrate={"cod": 1.5}, mode="continuous")
    c = sb.run_adm1(substrate={"cod": 2.0}, mode="continuous")
    assert len(calls) == 2 and memo.hits == 1 and memo.misses == 2
    assert b["CH4"] == 3.0 and c["CH4"] == 4.0             # cache not corrupted by the caller
    assert [x[0] for x in core._caps] == ["x <= 1"] * 3    # cap replayed on the hit


@pytest.mark.slow
def test_engine_end_to_end_mock(tmp_path):
    out = tmp_path / "eng.json"
    r = subprocess.run([sys.executable, os.path.join(HERE, "engine_mock_run.py"), ADM1_DIR, str(out)],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = json.load(open(out))
    frozen = {"A": (1365.1333882001034, 5.197704050762951), "B": (583.2053071249042, 24.402600316096553)}
    wk = d.pop("worker")
    assert wk["status"] == ["ok", "done"] and wk["rows"] == 4 and not wk["err"]   # 2 contexts x 2 heat
    assert wk["fD"] == [0.8] * 4 and wk["x"] == [0.3] * 4
    for P, v in d.items():
        assert v["adm1_misses"] == 2 and v["adm1_hits"] == 6       # one spin-up + one run per point
        caps = {x["caps"] for x in v["runs"]}
        assert len(caps) == 1 and next(iter(caps))                  # same caps on cached builds
        for x in v["runs"]:
            assert x["MJ_hydrochar"] == pytest.approx(frozen[P][0], rel=1e-9)
            assert x["CH4"] == pytest.approx(frozen[P][1], rel=1e-6)
            assert x["total"] == pytest.approx(x["cat_sum"], rel=1e-12) and not x["missing"]
        own = [x for x in v["runs"] if x["heat"] == "own_biogas"][0]
        base = [x for x in v["runs"] if x["heat"] == "paper1"][0]
        assert own["biomethane"] < base["biomethane"]
