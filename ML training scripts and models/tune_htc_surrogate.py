"""Hyperparameter search for the HTC surrogate.

For each output, 32 XGBoost settings are scored by random 5-fold
cross-validation (R2): 30 drawn at random from SPACE, plus two fixed
references (the earlier HADES settings and those of Katongtung et al., 2024). The best setting of each output is written to
htc_xgb_best_params.json and used by "train_htc_surrogate.py".

The winner and the two references are then scored by random 10-fold
cross-validation with three other shuffles (seeds 100-102), the validation
procedure of Katongtung et al. (2024); results in
htc_xgb_tuning_results.csv. The search is not nested, so the 10-fold
R2 of the winner is slightly optimistic (Cawley and Talbot, 2010, J. Mach.
Learn. Res. 11, 2079-2107); the two references are scored the same way.

Data: HTC_dataset_paper.xlsx, sheet "data" (938 rows, 122 studies), as is.

Usage: python "tune_htc_surrogate.py"   (run from this folder)
"""
import os, json, platform
import numpy as np, pandas as pd
import xgboost, sklearn
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, ParameterSampler, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "HTC_dataset_paper.xlsx")
FEATURES = ["time (min)", "temp(C)", "VM(%)", "FC(%)", "Ci(%)", "Hi(%)",
            "Ni(%)", "Si(%)", "Ashi", "Sv_BL"]
TARGETS = ["Bio_char(%)", "HHVo(Mj/kg)", "Co(%)", "Ho(%)", "No(%)", "So(%)", "Asho"]
BASE = dict(tree_method="hist", n_jobs=4)
V4 = dict(n_estimators=600, learning_rate=0.05, max_depth=4, subsample=0.8,       # earlier HADES settings
          colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.5)
KAT = dict(n_estimators=850, learning_rate=0.08, max_depth=90, subsample=0.8,     # Katongtung et al. (2024)
           colsample_bytree=0.9, colsample_bylevel=0.95, colsample_bynode=0.4,
           min_child_weight=7, gamma=2e-05)
SPACE = dict(n_estimators=[300, 600, 1000], max_depth=[3, 4, 5, 6, 8, 10],
             learning_rate=[0.03, 0.05, 0.08, 0.1], subsample=[0.7, 0.8, 0.9, 1.0],
             colsample_bytree=[0.6, 0.8, 1.0], min_child_weight=[1, 3, 5, 7],
             reg_lambda=[0.0, 1.0, 3.0, 10.0], reg_alpha=[0.0, 0.5])
N_ITER = 30


def main():
    df = pd.read_excel(DATA, "data")
    assert len(df) == 938, f"dataset has {len(df)} rows, expected 938"
    X_all = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    cands = [("v4", V4), ("katongtung", KAT)] + \
            [(f"rs{i:02d}", p) for i, p in enumerate(ParameterSampler(SPACE, N_ITER, random_state=7))]
    rows, best = [], {}
    for t in TARGETS:
        y = pd.to_numeric(df[t], errors="coerce"); m = y.notna()
        X, yy = X_all[m], y[m]
        cv5 = KFold(5, shuffle=True, random_state=11)
        scores = {name: cross_val_score(XGBRegressor(**BASE, **p, random_state=0), X, yy,
                                        cv=cv5, scoring="r2").mean() for name, p in cands}
        win = max(scores, key=scores.get)
        best[t] = dict(name=win, params=dict(cands)[win], search_R2_5fold=scores[win])
        for name in ["v4", "katongtung", win]:
            p = dict(cands)[name]; r2s, rm = [], []
            for s in (100, 101, 102):
                pred = cross_val_predict(XGBRegressor(**BASE, **p, random_state=s), X, yy,
                                         cv=KFold(10, shuffle=True, random_state=s))
                r2s.append(r2_score(yy, pred)); rm.append(mean_squared_error(yy, pred) ** 0.5)
            rows.append(dict(target=t, candidate=name if name != win or name in ("v4", "katongtung") else "tuned",
                             is_winner=name == win, n=len(yy), R2_10fold=np.mean(r2s),
                             R2_sd_seeds=np.std(r2s), RMSE=np.mean(rm), search_R2_5fold=scores[name]))
        print(t, "winner:", win, round(scores[win], 3), "| v4", round(scores["v4"], 3),
              "| katongtung", round(scores["katongtung"], 3), flush=True)

    res = pd.DataFrame(rows).drop_duplicates(["target", "candidate"])
    res.to_csv(os.path.join(HERE, "htc_xgb_tuning_results.csv"), index=False)
    with open(os.path.join(HERE, "htc_xgb_best_params.json"), "w") as fh:
        json.dump(best, fh, indent=2)
    print(f"Python {platform.python_version()}, xgboost {xgboost.__version__}, "
          f"scikit-learn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__}")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
