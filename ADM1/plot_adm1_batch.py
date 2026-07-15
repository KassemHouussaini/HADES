import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_adm1_results(adm1_out, adm1_out_blank, pw_inputs, V_liq, V_gas):
    df     = adm1_out["simulate_results"].copy()
    df_blk = adm1_out_blank["simulate_results"].copy()

    # 1. TOTAL COD FED
    S_COD_total = (df["S_su"] + df["S_aa"] + df["S_fa"] +
                   df["S_va"] + df["S_bu"] + df["S_pro"] + df["S_ac"] +
                   df["X_ch"] + df["X_pr"] + df["X_li"] + df["X_xc"] +
                   df["S_I_furan"] + df["S_I_alcohol"] + df["S_I_phenol"] + df["S_I_acid"] +
                   df["S_I_ketone"] + df["S_I_humic"]).iloc[0]

    df["VFA_total"] = df["S_ac"] + df["S_pro"] + df["S_bu"] + df["S_va"]

    # 2. BLANK-CORRECTED BMP
    CH4_sub     = df["S_gas_ch4"] * V_gas
    CH4_blk     = df_blk["S_gas_ch4"] * V_gas
    CH4_net_mL  = np.maximum(0, (CH4_sub - CH4_blk)) * 350_000
    COD_fed_g   = S_COD_total * V_liq * 1000
    df["BMP"]   = CH4_net_mL / COD_fed_g if COD_fed_g > 0 else 0.0

    # Convenience aggregates for plotting
    df["S_I_fast"] = df["S_I_furan"] + df["S_I_alcohol"] + df["S_I_phenol"] + df["S_I_acid"]
    df["S_I_slow"] = df["S_I_ketone"] + df["S_I_humic"]

    # 3. pH
    if "pH" in df.columns and df["pH"].iloc[1] < 2:
        df["pH_plot"] = df["pH"]
    else:
        df["pH_plot"] = -np.log10(df["S_H_ion"].clip(lower=1e-14))

    t = df["time"]

    # Plot 1: BMP
    plt.figure(figsize=(10, 4))
    plt.plot(t, df["BMP"], color="steelblue", linewidth=2)
    plt.axhline(350, color="black", linestyle=":", label="Theoretical Max (STP)")
    plt.xlabel("Time (days)")
    plt.ylabel("BMP (mL CH₄ @STP / g COD fed)")
    plt.title("Biochemical Methane Potential — ADM1 batch")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Plot 2: pH
    plt.figure(figsize=(10, 4))
    plt.plot(t, df["pH_plot"], color="darkorange", linewidth=2)
    plt.axhline(6.5, color="red",   linestyle="--", alpha=0.5, label="Inhibition limit (6.5)")
    plt.axhline(7.5, color="green", linestyle="--", alpha=0.5, label="Upper limit (7.5)")
    plt.xlabel("Time (days)")
    plt.ylabel("pH")
    plt.title("pH Stability (Batch)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Plot 3: Recalcitrant pools — aggregated fast/slow + individual sub-pools
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(t, df["S_I_fast"], label="Fast total (furan+alcohol+phenol+acid)", color="steelblue", linewidth=2)
    axes[0].plot(t, df["S_I_slow"], label="Slow total (ketone+humic)", color="darkorange", linewidth=2)
    axes[0].set_xlabel("Time (days)")
    axes[0].set_ylabel("kgCOD / m³")
    axes[0].set_title("Recalcitrant Pools — Aggregated")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(t, df["S_I_furan"],   label="Furan-like",   linestyle="-")
    axes[1].plot(t, df["S_I_alcohol"], label="Alcohol-like", linestyle="--")
    axes[1].plot(t, df["S_I_phenol"],  label="Phenol-like",  linestyle="-.")
    axes[1].plot(t, df["S_I_acid"],    label="Acid-like",    linestyle=":")
    axes[1].plot(t, df["S_I_ketone"],  label="Ketone/pyrazine-like", linestyle="-",  linewidth=2)
    axes[1].plot(t, df["S_I_humic"],   label="Humic-like",   linestyle="--", linewidth=2)
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("kgCOD / m³")
    axes[1].set_title("Recalcitrant Pools — Individual Sub-pools")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

    # Summary
    final = df.iloc[-1]
    print("\n" + "="*40)
    print(f"FINAL BMP:     {df['BMP'].iloc[-1]:.2f} mL/g COD")
    print(f"FINAL pH:      {df['pH_plot'].iloc[-1]:.2f}")
    print(f"Acetate:       {final['S_ac']:.4f} kgCOD/m³")
    print(f"S_I_fast (SS): {final['S_I_fast']:.4f} kgCOD/m³")
    print(f"S_I_slow (SS): {final['S_I_slow']:.4f} kgCOD/m³")
    print("="*40 + "\n")