"""Feedstock composition sampler (decision log F15)."""
import importlib.util
import os
from types import SimpleNamespace

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def fs():
    return _load("fs_test", "feedstock_sampler.py")


@pytest.fixture(scope="module")
def draws(fs):
    corr = fs.correlations()[0]
    return {c: fs.sample(c, 150, seed=7, corr=corr[c]) for c in fs.RANGES}


def test_closure_and_ranges(fs, draws):
    for cls, comps in draws.items():
        assert len(comps) == 150
        for c in comps:
            assert fs.inside(c, cls)
            assert c["C_feed"] + c["H_feed"] + c["N_feed"] + c["S_feed"] + c["Ash_feed"] + c["O_feed"] == pytest.approx(100)
            assert c["VM_feed"] + c["FC_feed"] + c["Ash_feed"] == pytest.approx(100)


def test_reproducible(fs, draws):
    corr = fs.correlations()[0]["food_waste"]
    assert fs.sample("food_waste", 150, seed=7, corr=corr) == draws["food_waste"]


def test_ash_dilutes_carbon(draws):
    for cls, comps in draws.items():
        a = np.array([[c["C_feed"], c["Ash_feed"]] for c in comps])
        assert np.corrcoef(a.T)[0, 1] < -0.3, cls


def test_centres_near_f15(fs, draws):
    for cls, comps in draws.items():
        for p in ("C", "Ash"):
            lo, hi, c = fs.RANGES[cls][p]
            med = np.median([d[fs.KEY[p]] for d in comps])
            assert abs(med - c) < 0.15 * (hi - lo), (cls, p, med, c)


def test_central_composition(fs):
    for cls in fs.RANGES:
        c = fs.central(cls)
        assert c["VM_feed"] + c["FC_feed"] + c["Ash_feed"] == pytest.approx(100)


def test_inside_surrogate_domain(fs, draws):
    pytest.importorskip("xgboost")
    for cls, comps in draws.items():
        share, _ = fs.domain_share(comps[:40], T=220, RT=60, WC=max(85.0, fs.RANGES[cls]["WC"][2]))
        assert share >= 0.95, (cls, share)


def test_make_points_uses_one_composition_per_sample(fs):
    se = _load("se_comp", "sweep_engine.py")
    comps = fs.sample("food_waste", 3, seed=1)
    feeds = {"fw": dict(feedstock_type="food_waste", feed={"WC": 80.0}, compositions=comps)}
    pts = se.make_points(np.zeros((3, 0)), feeds, [180], [60], ["A", "B"], [],
                         SimpleNamespace(PATHWAY_B_FEED_PARAMS={}), "t")
    for p in pts:
        assert p["feed"]["C_feed"] == comps[p["sample_id"]]["C_feed"] and p["feed"]["WC"] == 80.0
    with pytest.raises(ValueError):
        se.make_points(np.zeros((4, 0)), feeds, [180], [60], ["A"], [],
                       SimpleNamespace(PATHWAY_B_FEED_PARAMS={}), "t")
