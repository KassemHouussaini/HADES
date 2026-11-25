"""
feedstock_sampler.py
====================
Feedstock compositions for the Paper 2 main sweep (decision log F15).

Each sample of a sweep gets its own composition within its feedstock class.
Elements are not drawn independently: C, H, N, S, O and ash are parts of a
whole and are correlated (more ash means less of everything organic, C and H
move together). Sampling is done in additive log-ratio space (Aitchison 1986,
The Statistical Analysis of Compositional Data, Chapman & Hall):

    y_i = ln(x_i / x_O),   i in C, H, N, S, ash

  location   y at the central composition of the class (F15 table)
  spread     sd_i = (ln(hi_i / O_c) - ln(lo_i / O_c)) / 4, i.e. the F15 range
             of element i is about +-2 sd at the central oxygen
  links      correlation matrix of y measured on the real compositions of
             the class in the HTC training dataset (F14 cleaning); for classes
             with too few compositions (dairy manure, OFMSW) the within-class
             correlation pooled over the four classes (each class centred on
             its own mean, so differences between classes do not count as
             correlation)

A draw is back-transformed (x_O = 100 / (1 + sum exp y), x_i = x_O exp y_i,
so every draw sums to 100 %) and kept only if every element, O, VM and FC
lie in the class ranges. Proximate analysis: the FC share r = FC / (VM + FC)
is drawn on its own (logit-normal, centred and scaled on the F15 VM and FC
ranges), then VM = (100 - ash)(1 - r), FC = (100 - ash) r, so VM + FC + ash
= 100.

Moisture is not drawn here: it is a sweep factor (feed_WC within WC_range).

Usage
    comps = feedstock_sampler.sample("food_waste", n=100, seed=1)   # list of dicts
    se.make_points(..., feeds={"food_waste": dict(feedstock_type="food_waste",
                   feed=dict(WC=80.0), WC_range=(70, 90), compositions=comps)})
"""
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(os.path.dirname(_HERE), "ML training scripts and models", "HTC_dataset_paper.xlsx")

PARTS = ("C", "H", "N", "S", "Ash")           # log-ratio parts, O is the reference
KEY = {"C": "C_feed", "H": "H_feed", "N": "N_feed", "S": "S_feed", "Ash": "Ash_feed",
       "O": "O_feed", "VM": "VM_feed", "FC": "FC_feed"}

# F15 (decision log): (low, high, central), % dry basis; WC % wet basis.
# O is a result (range used only to reject draws); sewage sludge = raw mixed
# (primary + waste activated), undigested.
RANGES = {
    "food_waste": dict(
        WC=(70, 90, 80), C=(37, 54, 47), H=(4.5, 8.0, 6.6), N=(1.7, 6.0, 2.8), S=(0.05, 0.6, 0.25),
        Ash=(2, 15, 8), VM=(68, 85, 79), FC=(5, 21, 13), O=(30, 44, 37)),
    "sewage_sludge": dict(
        WC=(70, 85, 78), C=(28, 45, 35), H=(4.0, 7.0, 5.3), N=(3.0, 7.5, 4.8), S=(0.5, 1.7, 1.0),
        Ash=(18, 42, 30), VM=(50, 75, 62), FC=(3, 13, 8), O=(15, 30, 24)),
    "dairy_manure": dict(
        WC=(82, 90, 86), C=(33, 45, 40), H=(4.5, 6.0, 5.3), N=(1.4, 3.0, 2.3), S=(0.2, 0.6, 0.4),
        Ash=(9, 25, 16), VM=(60, 77, 69), FC=(11, 18, 15), O=(28, 42, 36)),
    "ofmsw": dict(
        WC=(65, 80, 73), C=(42, 52, 47), H=(5.5, 7.0, 6.3), N=(2.0, 3.8, 2.8), S=(0.1, 0.4, 0.25),
        Ash=(5, 25, 12), VM=(65, 82, 75), FC=(8, 24, 13), O=(28, 40, 32)),
}

# real compositions per class in the HTC dataset (column "biomass", lower case)
DATASET_LABELS = {
    "food_waste": ("food waste", "foodwaste"),
    "sewage_sludge": ("sewage sludge", "sewage sludge (ss)", "municipal sewage sludge", "raw sludge", "sludge"),
    "dairy_manure": ("manure", "cow dung"),
    "ofmsw": ("solid waste",),
}
OWN_CORRELATION = ("food_waste", "sewage_sludge")    # enough compositions for their own correlation
S_ZERO = 0.01                                         # % dry, replaces S = 0 before taking logs


def _excluded(r):
    """F14 cleaning (and F6): rows that are not the class or fail closure."""
    ref = str(r["Ref"])
    return (("126395" in ref and r["Ni(%)"] == 10.0)              # duplicate row with N 10.0 for 1.0
            or ("biombioe.2018.11.016" in ref and r["Ashi"] < 1)  # not sludge
            or ("125164" in ref and r["Ashi"] > 44)               # digestate rows
            or ("2015.01.118" in ref and r["Ashi"] > 18.6)        # ash series of one sample
            or r["Ci(%)"] > 70 or r["Si(%)"] > 5 or r["Ni(%)"] < 0.05
            or 100 - (r["Ci(%)"] + r["Hi(%)"] + r["Ni(%)"] + r["Si(%)"] + r["Ashi"]) < 5)


def real_compositions(path=DATASET):
    """{class: array (n, 5) of C, H, N, S, ash (% dry)} from the HTC dataset,
    distinct compositions, F14 cleaning."""
    import pandas as pd
    d = pd.read_excel(path)
    cols = ["Ci(%)", "Hi(%)", "Ni(%)", "Si(%)", "Ashi"]
    d["_b"] = d["biomass"].astype(str).str.strip().str.lower()
    d = d.dropna(subset=cols).drop_duplicates(subset=["Ref", "_b"] + cols)
    out = {}
    for cls, labels in DATASET_LABELS.items():
        g = d[d["_b"].isin(labels)]
        g = g[~g.apply(_excluded, axis=1)]
        out[cls] = g[cols].to_numpy(float)
    return out


def _alr(x5):
    x = np.array(x5, float)
    x[:, 3] = np.maximum(x[:, 3], S_ZERO)
    o = 100.0 - x.sum(1)
    return np.log(x / o[:, None])


def correlations(path=DATASET):
    """{class: 5x5 correlation of the log-ratios}, own data for food waste and
    sludge, pooled within-class for the others; and the number of rows used."""
    real = real_compositions(path)
    y = {c: _alr(v) for c, v in real.items() if len(v) >= 3}
    pooled = np.vstack([v - v.mean(0) for v in y.values()])
    out, n = {}, {}
    for cls in RANGES:
        if cls in OWN_CORRELATION and cls in y and len(y[cls]) >= 10:
            out[cls], n[cls] = np.corrcoef(y[cls], rowvar=False), len(y[cls])
        else:
            out[cls], n[cls] = np.corrcoef(pooled, rowvar=False), -len(pooled)   # negative: pooled
    return out, n


def _location_scale(cls):
    R = RANGES[cls]
    o_c = 100.0 - sum(R[p][2] for p in PARTS)
    mu = np.array([np.log(R[p][2] / o_c) for p in PARTS])
    sd = np.array([(np.log(R[p][1] / o_c) - np.log(R[p][0] / o_c)) / 4.0 for p in PARTS])
    return mu, sd


def _fc_share(cls):
    R = RANGES[cls]
    logit = lambda r: np.log(r / (1 - r))
    c = R["FC"][2] / (R["FC"][2] + R["VM"][2])
    lo = R["FC"][0] / (R["FC"][0] + R["VM"][1])
    hi = R["FC"][1] / (R["FC"][1] + R["VM"][0])
    return logit(c), (logit(hi) - logit(lo)) / 4.0


def inside(comp, cls):
    """True if every element, O, VM and FC lie in the class ranges."""
    R = RANGES[cls]
    return all(R[k][0] - 1e-9 <= comp[KEY[k]] <= R[k][1] + 1e-9 for k in ("C", "H", "N", "S", "Ash", "O", "VM", "FC"))


def sample(cls, n, seed, corr=None, max_tries=200):
    """n compositions of class `cls` (list of dicts with C_feed ... FC_feed,
    % dry, O by difference). Reproducible for a given seed."""
    if cls not in RANGES:
        raise KeyError(f"no composition ranges for {cls!r}; known: {sorted(RANGES)}")
    if corr is None:
        corr = correlations()[0][cls]
    mu, sd = _location_scale(cls)
    cov = corr * np.outer(sd, sd)
    m_r, s_r = _fc_share(cls)
    rng = np.random.default_rng(seed)
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(f"[feedstock_sampler] {cls}: only {len(out)} of {n} draws inside the ranges")
        y = rng.multivariate_normal(mu, cov, size=4 * n)
        r = 1.0 / (1.0 + np.exp(-(m_r + s_r * rng.standard_normal(4 * n))))
        for yi, ri in zip(y, r):
            xo = 100.0 / (1.0 + np.exp(yi).sum())
            x = xo * np.exp(yi)
            comp = {KEY[p]: float(v) for p, v in zip(PARTS, x)}
            comp["O_feed"] = 100.0 - sum(comp[KEY[p]] for p in PARTS)
            comp["VM_feed"] = (100.0 - comp["Ash_feed"]) * (1.0 - ri)
            comp["FC_feed"] = (100.0 - comp["Ash_feed"]) * ri
            if inside(comp, cls):
                out.append(comp)
                if len(out) == n:
                    break
    return out


def central(cls):
    """Central composition of the class (F15), O by difference, VM + FC + ash = 100."""
    R = RANGES[cls]
    comp = {KEY[p]: float(R[p][2]) for p in PARTS}
    comp["O_feed"] = 100.0 - sum(comp[KEY[p]] for p in PARTS)
    r = R["FC"][2] / (R["FC"][2] + R["VM"][2])
    comp["VM_feed"] = (100.0 - comp["Ash_feed"]) * (1 - r)
    comp["FC_feed"] = (100.0 - comp["Ash_feed"]) * r
    return comp


def domain_share(comps, T, RT, WC, ml_module_path=None):
    """Share of compositions inside the HTC surrogate's applicability domain
    (C0b, k-NN distance <= threshold) at T (C), RT (min) and WC = moisture of
    the HTC slurry (% wet; max of feed moisture and the HTC moisture target)."""
    import importlib.util
    p = ml_module_path or os.path.join(os.path.dirname(_HERE), "ML training scripts and models", "htc_surrogate.py")
    spec = importlib.util.spec_from_file_location("htc_surrogate_ad", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if m.AD is None:
        raise RuntimeError("applicability_domain.json not found")
    sv = 1.0 / (100.0 / WC - 1.0)               # water / dry biomass, as in HTC_parametric_ML
    d = []
    for c in comps:
        x = {"time (min)": RT, "temp(C)": T, "VM(%)": c["VM_feed"], "FC(%)": c["FC_feed"],
             "Ci(%)": c["C_feed"], "Hi(%)": c["H_feed"], "Ni(%)": c["N_feed"], "Si(%)": c["S_feed"],
             "Ashi": c["Ash_feed"], "Sv_BL": sv}
        d.append(m.ad_distance(x))
    d = np.array(d)
    return float((d <= m.AD["threshold"]).mean()), d
