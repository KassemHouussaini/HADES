"""Input checks of build_system (check_inputs), feedstock helpers
(close_feed) and the self-invalidating LCA score cache of
lca_contributions. No Brightway needed."""
import importlib.util, os, sys, types
from types import SimpleNamespace
import pytest

FW = dict(WC=85.0, C_feed=38.25, H_feed=5.47, N_feed=3.66, S_feed=0.50, Ash_feed=9.10,
          VM_feed=82.6, FC_feed=8.3)


def _ctx(sb, **kw):
    base = dict(default_params=sb.close_feed(FW), ml_params=None, T_HTC=180, RT_HTC=60,
                target_moisture_HTC=0.85, MC_char=0.35, target_MC_char_dry=0.10,
                m_feedstock=1000.0, COD_conc_target=10.0, PATHWAY="A", feedstock_name="food_waste")
    base.update(kw)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------- feedstock helpers
def test_close_feed_O_by_difference(sb):
    f = sb.close_feed(FW)
    assert f["O_feed"] == pytest.approx(100 - 38.25 - 5.47 - 3.66 - 0.50 - 9.10)


def test_close_feed_rejects_stale_O(sb):
    f = sb.close_feed(FW)
    f["C_feed"] *= 1.2                      # composition changed, O not updated
    with pytest.raises(ValueError, match="update O"):
        sb.close_feed(f)


# ---------------------------------------------------------------- check_inputs
def test_check_inputs_builds_surrogate_params(sb):
    ml = sb.check_inputs(_ctx(sb))
    assert ml["T"] == 180 and ml["RT"] == 60 and ml["C_feed"] == 38.25


def test_check_inputs_accepts_consistent_copies(sb):
    f = sb.close_feed(FW)
    sb.check_inputs(_ctx(sb, ml_params={**f, "T": 180, "RT": 60}))


@pytest.mark.parametrize("ml", [{"T": 220}, {"RT": 120}, {"C_feed": 40.0}])
def test_check_inputs_rejects_disagreeing_copies(sb, ml):
    with pytest.raises(ValueError, match="ml_params"):
        sb.check_inputs(_ctx(sb, ml_params=ml))


@pytest.mark.parametrize("T", [144.0, 159.9, 280.1, 300.0])
def test_check_inputs_rejects_T_outside_pw_adm1_range(sb, T):
    with pytest.raises(ValueError, match="T_HTC"):
        sb.check_inputs(_ctx(sb, T_HTC=T))


@pytest.mark.parametrize("T", [160.0, 280.0])
def test_check_inputs_accepts_range_ends(sb, T):
    sb.check_inputs(_ctx(sb, T_HTC=T))


@pytest.mark.parametrize("kw", [dict(RT_HTC=0), dict(target_moisture_HTC=1.0),
                                dict(target_MC_char_dry=0.35), dict(m_feedstock=0.0),
                                dict(COD_conc_target=0.0), dict(PATHWAY="C"),
                                dict(PATHWAY="B", feedstock_name="garden_waste")])
def test_check_inputs_rejects_bad_settings(sb, kw):
    with pytest.raises(ValueError):
        sb.check_inputs(_ctx(sb, **kw))


def test_check_inputs_rejects_bad_feed(sb):
    f = sb.close_feed(FW)
    for bad in ({k: v for k, v in f.items() if k != "O_feed"},     # O missing
                {**f, "VM_feed": 70.0},                            # VM + FC + ash = 87.4
                {**f, "WC": 100.0}):
        with pytest.raises(ValueError):
            sb.check_inputs(_ctx(sb, default_params=bad))


def test_check_inputs_needs_HRT_in_continuous_mode(sb):
    with pytest.raises(ValueError, match="HRT"):
        sb.check_inputs(_ctx(sb), adm1_mode="continuous", adm1_HRT_days=None)
    sb.check_inputs(_ctx(sb), adm1_mode="continuous", adm1_HRT_days=30)


# ---------------------------------------------------------------- LCA score cache
class _Act:
    def __init__(self, key): self.key = key
    def __hash__(self): return hash(self.key)


@pytest.fixture
def lcc(monkeypatch):
    """lca_contributions with a fake bw2data/bw2calc that counts LCA calls."""
    calls = []
    meta = {"fg": {"modified": "t0", "depends": ["ei"]}, "ei": {"modified": "e0", "depends": []}}
    bd = types.SimpleNamespace(databases=meta, projects=types.SimpleNamespace(current="p1"))

    class FakeLCA:
        def __init__(self, demand, method):
            self.act = next(iter(demand)); calls.append(self.act.key)
        def lci(self): pass
        def lcia(self): self.score = 1.0
    for name, mod in (("bw2data", bd), ("bw2calc", types.SimpleNamespace(LCA=FakeLCA))):
        monkeypatch.setitem(sys.modules, name, mod)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("lcc_test", os.path.join(here, "lca_contributions.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m, bd, meta, calls


def test_cache_reuses_unchanged_scores(lcc):
    m, bd, meta, calls = lcc
    act, meth = _Act(("fg", "mix")), ("IW", "GWP")
    m.unit_score(act, meth); m.unit_score(act, meth)
    assert len(calls) == 1


def test_cache_invalidates_on_foreground_rebuild(lcc):
    m, bd, meta, calls = lcc
    act, meth = _Act(("fg", "mix")), ("IW", "GWP")
    m.unit_score(act, meth)
    meta["fg"]["modified"] = "t1"            # build_system rewrote the foreground
    m.unit_score(act, meth)
    assert len(calls) == 2


def test_cache_invalidates_on_background_change_and_project(lcc):
    m, bd, meta, calls = lcc
    act, meth = _Act(("fg", "mix")), ("IW", "GWP")
    m.unit_score(act, meth)
    meta["ei"]["modified"] = "e1"            # background re-imported or edited
    m.unit_score(act, meth)
    bd.projects.current = "p2"               # other project, same database names
    m.unit_score(act, meth)
    assert len(calls) == 3
