"""Figure S5: heat integration of Pathways A and B, Base scenario (food waste,
180 degC, HRT 30 d), per tonne of wet feedstock.

Every number is read from case_study_results.csv (columns written by
heat_integration.metadata()); nothing is recalculated except the slurry
temperature after E1, T_feed + Q_HTC_rec / (m cp), with cp from
heat_integration.CP_KJ_PER_KG_K.
Writes FigureS5_heat_integration.pdf and .png (600 dpi), plus the single-panel
FigureS5a/S5b_heat_integration.png, in this folder.

usage: python figureS_heat_integration.py [results.csv] [scenario]
"""
import os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "case_study_results.csv")
SCEN = sys.argv[2] if len(sys.argv) > 2 else "S1 Base FW"
LABEL = "Base scenario, food waste at 180 °C" if SCEN == "S1 Base FW" else SCEN
CP = 4.18e-3          # MJ/(kg K), heat_integration.CP_KJ_PER_KG_K / 1000
mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42})
INK, GREY = "#1a1a1a", "#555555"
HEAT, NG, ELEC = "#c0392b", "#d68910", "#1f5fbf"
F_HX, F_HTC, F_U, F_AD, F_OUT = "#fbe7dc", "#f6dcc8", "#ffffff", "#d9e8f6", "#eeeeee"

df = pd.read_csv(CSV)
R = {p: df[(df.scenario == SCEN) & (df.pathway == p)].iloc[0] for p in ("A", "B")}


def f(v, d=0):
    return f"{float(v):,.{d}f}"


def box(ax, x, y, w, h, title, lines=(), fc=F_U, ec=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.1))
    n = len(lines)
    ax.text(x, y + 0.15 * n, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
    for i, s in enumerate(lines):
        ax.text(x, y + 0.15 * n - 0.31 * (i + 1), s, ha="center", va="center", fontsize=9.4, color=INK)


def arrow(ax, p0, p1, label=None, color=INK, ls="-", dy=0.14, dx=0.0, lpos=0.5, ha="center", lw=1.3):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                                      mutation_scale=10, shrinkA=0, shrinkB=0))
    if label:
        ax.text(p0[0] + (p1[0] - p0[0]) * lpos + dx, p0[1] + (p1[1] - p0[1]) * lpos + dy, label,
                ha=ha, va="center", fontsize=9, color=color if color != INK else GREY)


def poly(ax, pts, label=None, color=INK, ls="-", lxy=None, ha="center"):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=1.3, ls=ls)
    arrow(ax, pts[-2], pts[-1], color=color, ls=ls)
    if label:
        ax.text(*lxy, label, ha=ha, va="center", fontsize=9, color=color)


def htc_train(ax, r, y, x0, feed_label):
    """Feed -> E1 -> electric heater -> reactor; product -> E1 -> E2 -> separation -> dryer."""
    m = float(r.m_feedstock_effective)
    T_f, T_h, T_p1, T_s = float(r.T_feed_HTC_C), float(r.T_HTC), float(r.T_prod1_C), float(r.T_SEP_C)
    T_pre = T_f + float(r.Q_HTC_rec_MJ) / (m * CP)
    W, H = 2.2, 1.2
    xs = [x0 + 2.75 * i for i in range(6)]
    box(ax, xs[0], y, W, H, feed_label, [f"{f(m)} kg slurry", f"{f(T_f)} °C"], fc=F_OUT)
    box(ax, xs[1], y, W, H, "E1 feed preheater", [f"ε = {float(r.eps_HR_used):.2f}",
                                                 f"recovers {f(r.Q_HTC_rec_MJ)} MJ"], fc=F_HX, ec=HEAT)
    box(ax, xs[2], y, W, H, "Electric heater", [f"net {f(r.Q_HTC_net_MJ)} MJ",
                                               f"{f(r.E_HTC_elec_kWh, 1)} kWh (η = 0.90)"])
    box(ax, xs[3], y, W, H, "HTC reactor", [f"{f(T_h)} °C",
                                           f"gross duty {f(r.Q_HTC_gross_MJ)} MJ"], fc=F_HTC)
    arrow(ax, (xs[0] + W / 2, y), (xs[1] - W / 2, y), f"{f(T_f)} °C")
    arrow(ax, (xs[1] + W / 2, y), (xs[2] - W / 2, y), f"{f(T_pre)} °C")
    arrow(ax, (xs[2] + W / 2, y), (xs[3] - W / 2, y), f"{f(T_h)} °C")
    arrow(ax, (xs[2] + 1.15, y + H / 2 + 0.35), (xs[2] + 0.5, y + H / 2), "electricity", color=ELEC,
          lpos=0.0, dy=0.0, dx=0.08, ha="left")
    # hot product back through E1, then E2
    yb = y - 1.7
    box(ax, xs[1], yb, W, H, "E2 product cooler", [f"{f(T_p1, 1)} → {f(T_s)} °C",
                                                    f"releases {f(r.Q_E2_MJ, 1)} MJ"], fc=F_HX, ec=HEAT)
    poly(ax, [(xs[3], y - H / 2), (xs[3], y - 0.85), (xs[1] + 0.35, y - 0.85), (xs[1] + 0.35, y - H / 2)],
         f"hot product {f(T_h)} °C", color=HEAT, lxy=(xs[2] + 0.2, y - 0.69))
    poly(ax, [(xs[1] - 0.35, y - H / 2), (xs[1] - 0.35, yb + H / 2)], None, color=HEAT)
    ax.text(xs[1] - 0.45, y - 0.85, f"{f(T_p1, 1)} °C", ha="right", va="center", fontsize=9, color=HEAT)
    box(ax, xs[2], yb, W, H, "Letdown, filter press", [f"at {f(T_s)} °C",
                                                       f"{f(r.E_press_kWh, 1)} kWh"])
    box(ax, xs[3], yb, W, H, "Belt dryer (electric)", [f"{f(r.Q_dry_util_MJ, 1)} MJ heat",
                                           f"+ {f(r.E_dry_fan_kWh, 1)} kWh fan"])
    box(ax, xs[4], yb, W, H, "Dried hydrochar", [f"{f(r.HC_dry_yield, 1)} kg (10 % moisture)",
                                                "to combustion"], fc=F_OUT)
    arrow(ax, (xs[1] + W / 2, yb), (xs[2] - W / 2, yb))
    arrow(ax, (xs[2] + W / 2, yb), (xs[3] - W / 2, yb))
    arrow(ax, (xs[3] + W / 2, yb), (xs[4] - W / 2, yb))
    return xs, yb, W, H


def table(ax, r, x, y, rows):
    ax.add_patch(FancyBboxPatch((x, y - 0.32 * len(rows) - 0.2), 3.6, 0.32 * len(rows) + 0.55,
                                boxstyle="round,pad=0.02,rounding_size=0.08", fc="#fafafa", ec="#999999", lw=0.8))
    ax.text(x + 0.15, y + 0.12, "Heat balance (MJ per t wet feed)", fontsize=8.2, fontweight="bold",
            va="center", color=INK)
    for i, (k, v) in enumerate(rows):
        yy = y - 0.32 * (i + 1)
        ax.text(x + 0.15, yy, k, fontsize=9, va="center", color=INK)
        ax.text(x + 3.45, yy, v, fontsize=9, va="center", ha="right", color=INK, fontweight="bold")


fig, axes = plt.subplots(2, 1, figsize=(13.5, 12.5))
for ax in axes:
    ax.set_xlim(-0.6, 14.6); ax.set_ylim(-5.0, 1.8); ax.axis("off")

# ---------------------------------------------------------------- Pathway A
r, ax = R["A"], axes[0]
ax.set_title(f"(a) Pathway A, {LABEL}: hot product preheats the feed (E1) and warms the digester inflow (E2)",
             loc="left", fontsize=12, fontweight="bold", color=INK)
xs, yb, W, H = htc_train(ax, r, 0.45, 0.6, "Feedstock")
yd = yb - 1.85
m_pw, m_tap = float(r.PW_yield), float(r.dilution_water_PW_m3) * 1000
box(ax, xs[2], yd, W, H, "Process water", [f"{f(m_pw)} kg", f"{f(r.T_PW_to_AD_C)} °C"], fc=F_OUT)
box(ax, xs[3], yd, W, H, "Digester, 35 °C", [f"inflow at {f(r.T_AD_in_C, 1)} °C",
                                             f"losses {f(r.Q_AD_loss_MJ, 1)} MJ"], fc=F_AD)
box(ax, xs[4], yd, W, H, "Dilution water", [f"{f(m_tap)} kg tap water", f"{f(r.T_amb_C)} °C"], fc=F_OUT)
arrow(ax, (xs[2], yb - H / 2), (xs[2], yd + H / 2), "process water", dx=0.1, ha="left", dy=0.0)
arrow(ax, (xs[2] + W / 2, yd), (xs[3] - W / 2, yd))
arrow(ax, (xs[4] - W / 2, yd), (xs[3] + W / 2, yd))
poly(ax, [(xs[1], yb - H / 2), (xs[1], yd), (xs[1] + 0.6, yd), (xs[1] + 0.6, yd - 0.75),
          (xs[3] - 0.4, yd - 0.75), (xs[3] - 0.4, yd - H / 2)],
     f"E2 heat to digester {f(r.Q_E2_to_AD_MJ, 1)} MJ", color=HEAT, ls="--", lxy=(xs[2] - 0.2, yd - 0.88))
arrow(ax, (xs[3] + 0.4, yd - H / 2 - 0.65), (xs[3] + 0.4, yd - H / 2),
      f"natural gas {f(r.Q_AD_util_MJ, 1)} MJ", color=NG, lpos=0.0, dy=-0.12, dx=0.1, ha="left")

# ---------------------------------------------------------------- Pathway B
r, ax = R["B"], axes[1]
ax.set_title(f"(b) Pathway B, {LABEL}: hot product preheats the digestate cake (E1) and heats the digester (E2)",
             loc="left", fontsize=12, fontweight="bold", color=INK)
xs, yb, W, H = htc_train(ax, r, 0.45, 0.6, "Digestate cake")
yd = yb - 1.85
m_in = float(r.m_feedstock) + float(r.dilution_water_AD_kg)
box(ax, xs[0], yd, W, H, "Feedstock + water", [f"{f(m_in)} kg", f"{f(r.T_AD_in_C)} °C"], fc=F_OUT)
box(ax, xs[1] + 1.2, yd, W, H, "Digester, 35 °C", [f"inflow at {f(r.T_AD_in_C)} °C",
                                                    f"losses {f(r.Q_AD_loss_MJ, 1)} MJ"], fc=F_AD)
box(ax, xs[3], yd, W, H, "Process water", [f"{f(r.PW_yield, 1)} kg at {f(r.T_SEP_C)} °C",
                                           "to wastewater treatment"], fc=F_OUT)
arrow(ax, (xs[0] + W / 2, yd), (xs[1] + 1.2 - W / 2, yd))
poly(ax, [(xs[2], yb - H / 2), (xs[2], yd + 0.15), (xs[3] - W / 2, yd + 0.15)], "process water",
     lxy=(xs[2] + 0.1, yd + 0.6), ha="left")
poly(ax, [(xs[1] + 1.2, yd + H / 2), (xs[1] + 1.2, yd + H / 2 + 0.25), (xs[0], yd + H / 2 + 0.25),
          (xs[0], 0.45 - 0.52)], "digestate (centrifuge)", lxy=(xs[0] + 0.1, yd + H / 2 + 0.38), ha="left")
poly(ax, [(xs[1] - 0.45, yb - H / 2), (xs[1] - 0.45, yd - 0.8), (xs[1] + 0.8, yd - 0.8), (xs[1] + 0.8, yd - H / 2)],
     f"E2 heat to digester {f(r.Q_E2_to_AD_MJ, 1)} MJ", color=HEAT, ls="--", lxy=(xs[1] - 0.35, yd - 0.95), ha="left")
arrow(ax, (xs[1] + 1.6, yd - H / 2 - 0.65), (xs[1] + 1.6, yd - H / 2),
      f"natural gas {f(r.Q_AD_util_MJ, 1)} MJ", color=NG, lpos=0.0, dy=-0.12, dx=0.1, ha="left")

h = [mpl.lines.Line2D([], [], color=INK, lw=1.3, label="mass flow"),
     mpl.lines.Line2D([], [], color=HEAT, lw=1.3, label="hot product / recovered heat"),
     mpl.lines.Line2D([], [], color=ELEC, lw=1.3, label="electricity"),
     mpl.lines.Line2D([], [], color=NG, lw=1.3, label="natural gas heat")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=10)
fig.subplots_adjust(left=0.01, right=0.995, top=0.96, bottom=0.05, hspace=0.12)
fig.savefig(os.path.join(HERE, "FigureS5_heat_integration.pdf"), bbox_inches="tight")
# one file per pathway (legend included in each), for landscape pages
for ax, tag in zip(axes, "ab"):
    others = [o for o in axes if o is not ax]
    for o in others: o.set_visible(False)
    bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    leg_bb = fig.legends[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    from matplotlib.transforms import Bbox
    box_ = Bbox.union([bb, Bbox([[bb.x0, leg_bb.y0], [bb.x1, bb.y0]])]) if tag == "b" else bb
    fig.savefig(os.path.join(HERE, f"FigureS5{tag}_heat_integration.png"), dpi=600, bbox_inches=box_.padded(0.1))
    for o in others: o.set_visible(True)
fig.savefig(os.path.join(HERE, "FigureS5_heat_integration.png"), dpi=600, bbox_inches="tight")
print("Saved FigureS5_heat_integration.pdf / .png ->", HERE)