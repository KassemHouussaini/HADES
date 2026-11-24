"""
applicability_domain.py
=======================
Applicability domain (AD) of the HTC surrogate, layer 2: distance of a query
to the nearest training point in the standardized joint input space.
Layer 1 (each input within the training 1st-99th percentile) is in
htc_surrogate.py.

Two-layer approach: Netzeva et al. (2005) ATLA 33:155; Sahigara et al.
(2012) Molecules 17:4791; Roy, Kar & Ambure (2015) Chemometr. Intell. Lab.
Syst. 145:22. Distance threshold D_T = mean + Z x sd of the nearest-neighbour
distances of the training set (Sahigara et al. 2012).

Choices made here
  * inputs standardized with the training mean and sd (the 10 surrogate
    inputs, as trained);
  * NaN-tolerant Euclidean distance: over the inputs both points have,
    rescaled to all inputs, d = sqrt(sum_common dz^2 x p / n_common); at least
    MIN_COMMON inputs in common (26 % of training rows lack VM and FC);
  * k = 1 (nearest neighbour);
  * the threshold uses, for each training point, its nearest neighbour from
    ANOTHER study. Rows of one study are the same feedstock at other
    conditions; within-study neighbours make the data look denser than it is
    for a new feedstock. With grouped (by study) 5-fold CV the out-of-fold
    error rises with this distance (validate() reproduces the table).

Run `python applicability_domain.py` after retraining the surrogate: it
writes htc_xgb_models/applicability_domain.json and prints the validation.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "HTC_dataset_paper.xlsx")
OUT = os.path.join(HERE, "htc_xgb_models", "applicability_domain.json")
Z_THRESHOLD = 2.0
MIN_COMMON = 6


def _standardize(X, mean, sd):
    return ((X - mean) / sd).to_numpy(float)


def nn_distance(z, Ztrain, exclude=None, min_common=MIN_COMMON):
    """Distance from standardized point z (1-D, may hold NaN) to its nearest
    row of Ztrain (rows may hold NaN); rows where `exclude` is True are skipped."""
    p = Ztrain.shape[1]
    common = ~np.isnan(Ztrain) & ~np.isnan(z)
    d2 = np.where(common, (Ztrain - z) ** 2, 0.0).sum(1)
    nc = common.sum(1)
    d = np.sqrt(d2 * p / np.maximum(nc, 1))
    d[nc < min_common] = np.inf
    if exclude is not None:
        d[exclude] = np.inf
    return float(d.min())


def build(z_threshold=Z_THRESHOLD, path=OUT):
    meta = json.load(open(os.path.join(HERE, "htc_xgb_models", "metadata.json")))
    feats = meta["features"]
    x = pd.read_excel(DATA, "data")
    X = x[feats].astype(float)
    mean, sd = X.mean(), X.std()
    Z = _standardize(X, mean, sd)
    refs = x["Ref"].astype(str).to_numpy()
    d_other = np.array([nn_distance(Z[i], Z, exclude=(refs == refs[i])) for i in range(len(Z))])
    d_other = d_other[np.isfinite(d_other)]
    m, s = float(d_other.mean()), float(d_other.std())
    ad = dict(
        method="nearest neighbour (k=1), standardized inputs, NaN-tolerant Euclidean; "
               "threshold from nearest neighbours in other studies",
        features=feats, mean=mean.tolist(), sd=sd.tolist(), min_common=MIN_COMMON,
        train_z=[[None if np.isnan(v) else round(float(v), 6) for v in row] for row in Z],
        nn_other_study_mean=m, nn_other_study_sd=s, z_threshold=z_threshold,
        threshold=m + z_threshold * s,
        share_training_above=float(np.mean(d_other > m + z_threshold * s)),
        n_rows=int(len(Z)), n_studies=int(len(set(refs))))
    json.dump(ad, open(path, "w"))
    print(f"written {path}: D_T = {ad['threshold']:.3f} (mean {m:.3f} + {z_threshold} x sd {s:.3f}), "
          f"{ad['share_training_above']:.1%} of training points beyond")
    return ad


def validate(n_splits=5, z_values=(1.0, 1.5, 2.0, 3.0)):
    """Grouped (by study) CV: out-of-fold absolute error of each target vs the
    distance of the held-out row to the training folds."""
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBRegressor
    meta = json.load(open(os.path.join(HERE, "htc_xgb_models", "metadata.json")))
    feats = meta["features"]
    x = pd.read_excel(DATA, "data")
    X = x[feats].astype(float)
    Z = _standardize(X, X.mean(), X.std())
    refs = x["Ref"].astype(str).to_numpy()
    folds = list(GroupKFold(n_splits=n_splits).split(X, groups=refs))
    dist = np.full(len(x), np.nan)
    for tr, te in folds:
        for i in te:
            dist[i] = nn_distance(Z[i], Z[tr])
    err = {}
    for t, m in meta["targets"].items():
        y = x[t].astype(float).to_numpy()
        e = np.full(len(x), np.nan)
        for tr, te in folds:
            tr = tr[np.isfinite(y[tr])]
            mdl = XGBRegressor(**m["params"], random_state=0).fit(X.iloc[tr], y[tr])
            ok = te[np.isfinite(y[te])]
            e[ok] = np.abs(mdl.predict(X.iloc[ok]) - y[ok])
        err[t] = e
    E = pd.DataFrame(err)
    E["rel_err"] = E.div(E.median()).mean(axis=1, skipna=True)
    E["dist"] = dist
    E["bin"] = pd.qcut(E["dist"], 10)
    table = E.groupby("bin", observed=True).agg(
        n=("dist", "size"), d_max=("dist", "max"), rel_err=("rel_err", "median"),
        yield_abs_err=("Bio_char(%)", "median"), C_abs_err=("Co(%)", "median"),
        HHV_abs_err=("HHVo(Mj/kg)", "median"))
    d = E["dist"]
    rows = []
    for z in z_values:
        th = d.mean() + z * d.std()
        a, b = E[d <= th], E[d > th]
        rows.append(dict(Z=z, D_T=th, flagged=float(np.mean(d > th)),
                         rel_err_inside=a["rel_err"].median(), rel_err_beyond=b["rel_err"].median()))
    return table, pd.DataFrame(rows)


if __name__ == "__main__":
    build()
    tab, zs = validate()
    pd.set_option("display.width", 200)
    print(tab.round(2).to_string())
    print(zs.round(3).to_string(index=False))
