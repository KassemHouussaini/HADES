"""HTC surrogate: one tuned XGBoost regressor per output.

Models are trained by "train_htc_surrogate.py" on HTC_dataset_paper.xlsx
(938 rows, 122 studies) and loaded from htc_xgb_models/. Predicted outputs:
hydrochar yield, HHV and C, H, N, S and ash contents; O is closed by
difference.

Checks: every input is compared with the 1st-99th percentile range of the
training data (flag and in_training_range), negative predictions are clipped at 0 with a note, and
C + H + N + S + ash > 100 % is flagged.
"""
import json, os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIR = os.path.join(_HERE, "htc_xgb_models")
META = json.load(open(os.path.join(_DIR, "metadata.json")))
feature_cols = META["features"]
target_cols = list(META["targets"])
models = {}
for _t, _m in META["targets"].items():
    models[_t] = XGBRegressor()
    models[_t].load_model(os.path.join(_DIR, _m["file"]))


def HTC_parametric_ML(T, RT, WC, VM_feed, FC_feed,
                      C_feed, H_feed, N_feed, O_feed, S_feed, Ash_feed, Sv_BL=None):
    """Hydrochar inventory for T (degC), RT (min), moisture WC (%, wet basis)
    and feed proximate and ultimate analysis (%, dry basis)."""
    if Sv_BL is None:
        if WC is None or not 0 < WC < 100:
            raise ValueError(f"[HTC surrogate] WC={WC} % must be in (0, 100)")
        Sv_BL = 1.0 / ((1.0 / (WC / 100.0)) - 1.0)          # water / dry biomass
    x = {"time (min)": RT, "temp(C)": T, "VM(%)": VM_feed, "FC(%)": FC_feed,
         "Ci(%)": C_feed, "Hi(%)": H_feed, "Ni(%)": N_feed, "Si(%)": S_feed,
         "Ashi": Ash_feed, "Sv_BL": Sv_BL}

    # applicability domain: 1st-99th percentile of each training input
    violations = []
    for f, v in x.items():
        lo, hi = META["feature_p1_p99"][f]
        if v is not None and np.isfinite(v) and not lo <= v <= hi:
            violations.append(f"{f}={v:.3g} outside training P1-P99 [{lo:.3g}, {hi:.3g}]")
    for v in violations:
        print(f"[HTC surrogate] ⚠️ EXTRAPOLATION {v}")

    X = pd.DataFrame([[x[c] for c in feature_cols]], columns=feature_cols, dtype=float)
    raw = {t: float(models[t].predict(X)[0]) for t in target_cols}
    notes = []
    p = {}
    for t, v in raw.items():
        if v < 0:
            notes.append(f"{t} predicted {v:.3f} clipped to 0")
            v = 0.0
        p[t] = v
    Y, HHV = p["Bio_char(%)"], p["HHVo(Mj/kg)"]
    C, H, N, S, Ash = p["Co(%)"], p["Ho(%)"], p["No(%)"], p["So(%)"], p["Asho"]
    if not 0 < Y <= 100:
        raise ValueError(f"[HTC surrogate] yield {Y:.2f} % outside (0, 100]")
    O = 100.0 - (C + H + N + S + Ash)
    mass_balance_ok = O >= 0
    if not mass_balance_ok:
        notes.append(f"C+H+N+S+Ash = {100 - O:.2f} % > 100, O_char clipped to 0")
        O = 0.0

    df = pd.DataFrame({
        "Parameter": ["Yield (%)", "C (%)", "H (%)", "N (%)", "O (%)", "S (%)", "Ash (%)", "HHV (MJ/kg)"],
        "Hydrochar": [Y, C, H, N, O, S, Ash, HHV],
        "Feedstock": [100, C_feed, H_feed, N_feed, O_feed, S_feed, Ash_feed, None]}).round(3)
    inventory = {
        "Y_char (%)": Y, "HHV_char (MJ/kg)": HHV, "C_char (%)": C, "H_char (%)": H,
        "O_char (%)": O, "N_char (%)": N, "S_char (%)": S, "Ash_char (%)": Ash,
        "WC_feed (%)": WC,
        "in_training_range": len(violations) == 0,
        "domain_violations": "; ".join(violations),
        "mass_balance_ok": mass_balance_ok,
        "mass_balance_note": "; ".join(notes),
        "ml_model": "xgboost_tuned",
    }
    return df, inventory
