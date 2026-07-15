# === HTC Parametric Model (Reduced XGB – Tippayawong, updated) ===
import pandas as pd
import numpy as np
import joblib

import os

# --- Load trained model and metadata (relative to this file's location) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(_HERE, "xgb_tippayawong_reduced_model.pkl"))
feature_cols = joblib.load(os.path.join(_HERE, "xgb_tippayawong_reduced_features.pkl"))
target_cols  = joblib.load(os.path.join(_HERE, "xgb_tippayawong_reduced_targets.pkl"))

def HTC_parametric_ML(
    T, RT, WC, VM_feed, FC_feed,
    C_feed, H_feed, N_feed, O_feed, S_feed, Ash_feed,
    Sv_BL=None
):
    """
    Predicts hydrochar properties and yields using the reduced-feature
    Tippayawong-trained XGBoost model.

    Notes:
    - If Sv_BL not provided, it is computed from WC (%):
         WC = W/(W+B) → W/B = 1 / ((1/WC) - 1)
    - Model does not predict N(%); N_char is taken from feedstock.
    Returns both a readable DataFrame and a standardized 'inventory' dictionary.
    """

    # --- 1️⃣ Compute Sv_BL if not given ---
    if Sv_BL is None and WC is not None:
        try:
            WC_frac = WC / 100.0
            if 0 < WC_frac < 1:
                # W/B = 1 / ((1/WC) - 1)
                Sv_BL = 1.0 / ((1.0 / WC_frac) - 1.0)
            else:
                Sv_BL = np.nan
        except Exception:
            Sv_BL = np.nan

    # --- 2️⃣ Build input vector (order must match training) ---
    x_dict = {
        "time (min)": RT,
        "temp(C)": T,
        "VM(%)": VM_feed,
        "FC(%)": FC_feed,
        "Ci(%)": C_feed,
        "Hi(%)": H_feed,
        "Ni(%)": N_feed,
        "Si(%)": S_feed,
        "Ashi": Ash_feed,
        "Sv_BL": Sv_BL,
    }
    X_new = pd.DataFrame([[x_dict.get(col, np.nan) for col in feature_cols]], columns=feature_cols)

    # --- 3️⃣ Predict outputs ---
    preds = model.predict(X_new)[0]
    preds = dict(zip(target_cols, preds))

    # --- 4️⃣ Map predictions to standardized nomenclature ---
    Y_char   = preds.get("Bio_char(%)", np.nan)
    HHV_char = preds.get("HHVo(Mj/kg)", np.nan)
    C_char   = preds.get("Co(%)", np.nan)
    H_char   = preds.get("Ho(%)", np.nan)
    #O_char   = preds.get("Oo(%)", np.nan)
    Ash_char = preds.get("Asho", np.nan)
    f_N_char = 0.30   # 20% of feedstock N goes to char, ASSUMPTION, GOOD FOR SENSITIVITY ANALYSIS 
    f_S_char = 0.50   # 40% of feedstock S goes to char, ASSUMPTION, GOOD FOR SENSITIVITY ANALYSIS 
    
    Y_char_dry_mass = (Y_char / 100)   # dry basis yield fraction
    N_char = (N_feed * f_N_char) / max(Y_char_dry_mass, 1e-6)
    S_char = (S_feed * f_S_char) / max(Y_char_dry_mass, 1e-6)
    O_char   = 100- C_char - H_char - N_char -S_char - Ash_char

            

    # --- 5️⃣ Readable summary DataFrame ---
    df = pd.DataFrame({
        "Parameter": ["Yield (%)", "C (%)", "H (%)", "N (%)", "O (%)", "S (%)", "Ash (%)", "HHV (MJ/kg)"],
        "Hydrochar": [Y_char, C_char, H_char, N_char, O_char, S_char, Ash_char, HHV_char],
        "Feedstock": [100, C_feed, H_feed, N_feed, O_feed, S_feed, Ash_feed, None],
    }).round(3)

    # --- 6️⃣ Standardized inventory dictionary (same naming as before) ---
    inventory = {
        "Y_char (%)": Y_char,
        "HHV_char (MJ/kg)": HHV_char,
        "C_char (%)": C_char,
        "H_char (%)": H_char,
        "O_char (%)": O_char,
        "N_char (%)": N_char,
        "S_char (%)": S_char,
        "Ash_char (%)": Ash_char,
        "WC_feed (%)": WC
    }

    return df, inventory

'''
# ============================================================
# === TEST RUN ===
# ============================================================
if __name__ == "__main__":
    print("\n=== 🔬 HTC Parametric (Reduced Feature, updated) Test Run ===")

    df, inv = HTC_parametric_ML(
        T=200, RT=60, WC=85,
        C_feed=47.5, H_feed=6.1, N_feed=2.0, O_feed=43.0,
        S_feed=0.2, Ash_feed=1.2
    )

    print("\n--- 📘 Prediction Table ---")
    print(df.to_string(index=False))

    print("\n--- 📊 Inventory Dictionary ---")
    for k, v in inv.items():
        print(f"{k:20s}: {v}")''' 
