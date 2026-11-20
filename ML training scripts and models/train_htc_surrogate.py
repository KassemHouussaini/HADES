"""Training of the HTC surrogate.

One XGBRegressor per output (Chen and Guestrin, 2016) with the
hyperparameters of htc_xgb_best_params.json (random search,
5-fold CV, following Katongtung et al., 2024), refitted on all rows of
HTC_dataset_paper.xlsx (sheet "data", 938 rows, 122 studies). Inputs: 10
features; outputs: 7 targets (O is closed by difference in the wrapper).
Missing values use XGBoost native handling (tree_method="hist").

Models are saved in XGBoost JSON format, with metadata.json holding the
training feature ranges (applicability check) and the 1st-99th percentiles
of each output (plausibility check).

Usage: python "train_htc_surrogate.py"   (run from this folder)
"""
import json, os, platform
import numpy as np, pandas as pd
import xgboost, sklearn
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "HTC_dataset_paper.xlsx")
PARAMS = os.path.join(HERE, "htc_xgb_best_params.json")
OUTDIR = os.path.join(HERE, "htc_xgb_models")
FEATURES = ["time (min)", "temp(C)", "VM(%)", "FC(%)", "Ci(%)", "Hi(%)",
            "Ni(%)", "Si(%)", "Ashi", "Sv_BL"]
TARGETS = ["Bio_char(%)", "HHVo(Mj/kg)", "Co(%)", "Ho(%)", "No(%)", "So(%)", "Asho"]
BASE = dict(tree_method="hist", n_jobs=4, random_state=0)   # random_state as in the search
BOUNDS = {"Bio_char(%)": (0, 100), "HHVo(Mj/kg)": (0, 40), "Co(%)": (0, 100),
          "Ho(%)": (0, 15), "No(%)": (0, 20), "So(%)": (0, 15), "Asho": (0, 100)}


def load_corrected(path=DATA):
    """The training set, as stored (no filtering here)."""
    return pd.read_excel(path, "data")


def main():
    df = load_corrected()
    assert len(df) == 938, f"corrected dataset has {len(df)} rows, expected 938"
    best = json.load(open(PARAMS))
    os.makedirs(OUTDIR, exist_ok=True)
    X_all = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    meta = {"n_rows": len(df), "n_studies": int(df["Ref"].nunique()), "targets": {},
            "features": FEATURES,
            "feature_range": {c: [float(X_all[c].min()), float(X_all[c].max())] for c in FEATURES},
            "feature_p1_p99": {c: [float(X_all[c].quantile(.01)), float(X_all[c].quantile(.99))]
                               for c in FEATURES},
            "environment": f"Python {platform.python_version()}, xgboost {xgboost.__version__}, "
                           f"scikit-learn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__}"}
    for t in TARGETS:
        y = pd.to_numeric(df[t], errors="coerce"); m = y.notna()
        p = best[t]["params"]
        mdl = XGBRegressor(**BASE, **p).fit(X_all[m], y[m])
        fn = f"xgb_{t.split('(')[0].strip().replace('%', '')}.json"
        mdl.save_model(os.path.join(OUTDIR, fn))
        fit = mdl.predict(X_all[m])
        lo, hi = BOUNDS[t]
        if ((fit < lo) | (fit > hi)).any():
            print(f"[FLAG] train_htc_surrogate: {t} training-set predictions outside {BOUNDS[t]}")
        meta["targets"][t] = {"file": fn, "params": p, "n_train": int(m.sum()),
                              "p1_p99": [float(y[m].quantile(.01)), float(y[m].quantile(.99))],
                              "search_R2_5fold": best[t]["search_R2_5fold"]}
        print(f"{t:13s} n={int(m.sum()):4d}  params={p}")
    json.dump(meta, open(os.path.join(OUTDIR, "metadata.json"), "w"), indent=1)
    print("saved ->", OUTDIR)


if __name__ == "__main__":
    main()
