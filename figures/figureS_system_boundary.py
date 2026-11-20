"""Figure S4: system boundaries of Pathways A and B (case study).

Foreground units modelled by HADES are inside the dashed boundary; avoided
products are outside. Background inputs come from ecoinvent 3.12 (cut-off).
Functional unit: treatment of 1 t of wet feedstock.
Writes FigureS4_system_boundary.pdf and .png (600 dpi) in this folder.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.dirname(os.path.abspath(__file__))
mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42})
INK, GREY = "#1a1a1a", "#555555"
HEAT, NG, ELEC = "#c0392b", "#d68910", "#1f5fbf"
F = {"htc": "#f6dcc8", "hx": "#fbe7dc", "ad": "#d9e8f6", "unit": "#ffffff",
     "use": "#eeeeee", "avoid": "#e3efe3", "bg": "#f7f7f7"}
W, H = 1.85, 1.0


def box(ax, x, y, text, fc, ls="-", w=W, h=H, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=INK, lw=1.1, ls=ls))
    ax.text(x, y, text, ha="center", va="center", fontsize=10.5, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, p0, p1, label=None, color=INK, ls="-", lpos=0.5, dy=0.13, dx=0.0, ha="center", lw=1.2):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=10,
                                shrinkA=0, shrinkB=0))
    if label:
        x = p0[0] + (p1[0] - p0[0]) * lpos + dx
        y = p0[1] + (p1[1] - p0[1]) * lpos + dy
        ax.text(x, y, label, ha=ha, va="center", fontsize=9, color=color if color != INK else GREY)


def poly(ax, pts, label=None, color=INK, ls="-", lpos=None, ha="center", dy=0.12):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=1.2, ls=ls)
    arrow(ax, pts[-2], pts[-1], color=color, ls=ls)
    if label and lpos is not None:
        ax.text(lpos[0], lpos[1], label, ha=ha, va="center", fontsize=9,
                color=color if color != INK else GREY)


def boundary(ax, x0, y0, x1, y1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="none", ec=GREY, lw=1.2, ls=(0, (5, 3))))
    ax.text(x0 + 0.62 * (x1 - x0), y1 - 0.15, "system boundary", ha="center", va="center", fontsize=9,
            style="italic", color=GREY)


# x positions of the columns
X = [0.0, 2.55, 5.1, 7.65, 10.2, 12.75, 15.3, 18.6]
fig, axes = plt.subplots(2, 1, figsize=(17, 11))
for ax in axes:
    ax.set_xlim(-1.75, 19.6); ax.set_ylim(-3.5, 1.75); ax.axis("off")

# ---------------------------------------------------------------- Pathway A
ax = axes[0]
ax.set_title("(a) Pathway A: HTC, then anaerobic digestion of the process water",
             loc="left", fontsize=13, fontweight="bold", color=INK)
boundary(ax, -1.15, -3.35, 16.55, 1.45)
y1, y2 = 0.45, -1.55
ax.text(-1.25, y1, "Wet feedstock\n1 t (FU)", ha="right", va="center", fontsize=10.5, color=INK)
arrow(ax, (-1.2, y1), (X[0] - W / 2, y1))
box(ax, X[0], y1, "E1\nfeed preheater", F["hx"])
box(ax, X[1], y1, "HTC reactor\n(electric heating)", F["htc"], bold=True)
box(ax, X[2], y1, "E2\nproduct cooler", F["hx"])
box(ax, X[3], y1, "Letdown and\nfilter press", F["unit"])
box(ax, X[4], y1, "Belt dryer\n(electric)", F["unit"])
box(ax, X[5], y1, "Hydrochar\ncombustion\n(industrial boiler)", F["use"])
box(ax, X[7], y1, "Avoided: Quebec\nindustrial heat mix", F["avoid"], ls="--")
arrow(ax, (X[0] + W / 2, y1 + 0.12), (X[1] - W / 2, y1 + 0.12))
arrow(ax, (X[1] - W / 2, y1 - 0.18), (X[0] + W / 2, y1 - 0.18), color=HEAT, dy=-0.13)
arrow(ax, (X[1] + W / 2, y1), (X[2] - W / 2, y1))
arrow(ax, (X[2] + W / 2, y1), (X[3] - W / 2, y1))
arrow(ax, (X[3] + W / 2, y1), (X[4] - W / 2, y1))
arrow(ax, (X[4] + W / 2, y1), (X[5] - W / 2, y1))
arrow(ax, (X[5] + W / 2, y1), (X[7] - W / 2, y1), "heat", ls="--", dy=0.13)
arrow(ax, (X[1], y1 + H / 2), (X[1], y1 + 0.95), "CO$_2$, CO", ls=":", dy=0.0, dx=0.12, ha="left", lpos=1.0)
arrow(ax, (X[5], y1 + H / 2), (X[5], y1 + 0.95), "SO$_2$, NO$_x$, PM, CO$_2$", ls=":", dy=0.0, dx=0.12, ha="left", lpos=1.0)
box(ax, X[6], y1 + 0.62, "Ash disposal", F["use"], w=1.6, h=0.55)
arrow(ax, (X[5] + W / 2, y1 + 0.3), (X[6] - 0.8, y1 + 0.62), "ash", dy=0.12)
# digester line
box(ax, X[3], y2, "Dilution to\nCOD target", F["unit"])
box(ax, X[4], y2, "Digester, 35 °C\n(PW-ADM1)", F["ad"], bold=True)
box(ax, X[5], y2, "PSA\nupgrading", F["unit"])
box(ax, X[6], y2, "Biomethane\nboiler", F["use"])
box(ax, X[7], y2, "Avoided:\nnatural gas heat", F["avoid"], ls="--")
box(ax, X[4], -2.8, "Wastewater\ntreatment", F["use"])
arrow(ax, (X[3], y1 - H / 2), (X[3], y2 + H / 2), "process water", dx=0.1, ha="left", dy=0.0)
arrow(ax, (X[2] - 0.6, y2), (X[3] - W / 2, y2), "tap water", dy=0.13)
arrow(ax, (X[3] + W / 2, y2), (X[4] - W / 2, y2))
arrow(ax, (X[4] + W / 2, y2), (X[5] - W / 2, y2), "biogas", dy=0.13)
arrow(ax, (X[5] + W / 2, y2), (X[6] - W / 2, y2))
arrow(ax, (X[6] + W / 2, y2), (X[7] - W / 2, y2), "heat", ls="--", dy=0.13)
arrow(ax, (X[4], y2 - H / 2), (X[4], -2.8 + H / 2), "effluent", dx=0.1, ha="left", dy=0.0)
arrow(ax, (X[4] + 0.45, y2 - H / 2), (X[4] + 1.2, y2 - 0.95), "CH$_4$, NH$_3$ (fugitive)", ls=":",
      lpos=1.0, dx=0.05, ha="left", dy=-0.08)
poly(ax, [(X[2], y1 - H / 2), (X[2], -0.75), (X[4] - 0.25, -0.75), (X[4] - 0.25, y2 + H / 2)],
     "E2 heat", color=HEAT, ls="--", lpos=(X[2] + 0.1, -0.62), ha="left")
arrow(ax, (X[5] - 0.2, -0.85), (X[4] + 0.3, y2 + H / 2), "natural gas heat", color=NG, lpos=0.0, dx=0.05,
      ha="left", dy=0.08)
arrow(ax, (X[0] - 0.3, -0.75), (X[0] - 0.3, y1 - H / 2), "tap water (only if\nmoisture < 85 %)", lpos=0.0, dy=-0.25, dx=0.0, ha="center")

# ---------------------------------------------------------------- Pathway B
ax = axes[1]
ax.set_title("(b) Pathway B: anaerobic digestion, then HTC of the digestate",
             loc="left", fontsize=13, fontweight="bold", color=INK)
boundary(ax, -1.15, -3.35, 16.55, 1.45)
y1, y2 = 0.45, -1.55
ax.text(-1.25, y1, "Wet feedstock\n1 t (FU)", ha="right", va="center", fontsize=10.5, color=INK)
arrow(ax, (-1.2, y1), (X[0] - W / 2, y1))
box(ax, X[0], y1, "Dilution to\nTS target", F["unit"])
box(ax, X[1], y1, "Digester, 35 °C\n(ADM1)", F["ad"], bold=True)
box(ax, X[2], y1, "PSA\nupgrading", F["unit"])
box(ax, X[6], y1, "Biomethane\nboiler", F["use"])
box(ax, X[7], y1, "Avoided:\nnatural gas heat", F["avoid"], ls="--")
arrow(ax, (X[0] - 1.0, y1 + 0.62), (X[0] - 0.4, y1 + H / 2), "tap water", lpos=0.0, dy=0.1, ha="center")
arrow(ax, (X[0] + W / 2, y1), (X[1] - W / 2, y1))
arrow(ax, (X[1] + W / 2, y1), (X[2] - W / 2, y1), "biogas", dy=0.13)
arrow(ax, (X[2] + W / 2, y1), (X[6] - W / 2, y1), "biomethane", dy=0.13)
arrow(ax, (X[6] + W / 2, y1), (X[7] - W / 2, y1), "heat", ls="--", dy=0.13)
arrow(ax, (X[1] + 0.4, y1 + H / 2), (X[1] + 0.8, y1 + 0.75), "CH$_4$, NH$_3$ (fugitive)", ls=":",
      lpos=1.0, dx=0.05, ha="left", dy=0.05)
box(ax, X[0], y2, "Centrifuge", F["unit"])
box(ax, X[1], y2, "E1\nfeed preheater", F["hx"])
box(ax, X[2], y2, "HTC reactor\n(electric heating)", F["htc"], bold=True)
box(ax, X[3], y2, "E2\nproduct cooler", F["hx"])
box(ax, X[4], y2, "Letdown and\nfilter press", F["unit"])
box(ax, X[5], y2, "Belt dryer\n(electric)", F["unit"])
box(ax, X[6], y2, "Hydrochar\ncombustion\n(industrial boiler)", F["use"])
box(ax, X[7], y2, "Avoided: Quebec\nindustrial heat mix", F["avoid"], ls="--")
box(ax, X[2], -2.8, "Wastewater\ntreatment", F["use"])
poly(ax, [(X[1] - 0.3, y1 - H / 2), (X[1] - 0.3, -0.55), (X[0], -0.55), (X[0], y2 + H / 2)],
     "digestate", lpos=(X[0] + 0.1, -0.42), ha="left")
arrow(ax, (X[0] + W / 2, y2), (X[1] - W / 2, y2))
arrow(ax, (X[1] + W / 2, y2 + 0.12), (X[2] - W / 2, y2 + 0.12))
arrow(ax, (X[2] - W / 2, y2 - 0.18), (X[1] + W / 2, y2 - 0.18), color=HEAT, dy=-0.13)
arrow(ax, (X[2] + W / 2, y2), (X[3] - W / 2, y2))
arrow(ax, (X[3] + W / 2, y2), (X[4] - W / 2, y2))
arrow(ax, (X[4] + W / 2, y2), (X[5] - W / 2, y2))
arrow(ax, (X[5] + W / 2, y2), (X[6] - W / 2, y2))
arrow(ax, (X[6] + W / 2, y2), (X[7] - W / 2, y2), "heat", ls="--", dy=0.13)
arrow(ax, (X[2] + 0.35, y2 + H / 2), (X[2] + 0.35, y2 + 0.85), "CO$_2$, CO", ls=":", lpos=1.0, dx=0.1, ha="left", dy=0.0)
arrow(ax, (X[6] + 0.35, y2 + H / 2), (X[6] + 0.35, y2 + 0.85), "SO$_2$, NO$_x$, PM, CO$_2$", ls=":", lpos=1.0, dx=0.1, ha="left", dy=0.0)
box(ax, X[6], -2.75, "Ash disposal", F["use"], w=1.6, h=0.55)
arrow(ax, (X[6], y2 - H / 2), (X[6], -2.75 + 0.275), "ash", dx=0.1, ha="left", dy=0.0)
poly(ax, [(X[0], y2 - H / 2), (X[0], -2.75), (X[2] - W / 2, -2.75)], "centrate", lpos=(X[1], -2.62))
poly(ax, [(X[4], y2 - H / 2), (X[4], -2.75), (X[2] + W / 2, -2.75)], "process water", lpos=(X[3], -2.62))
poly(ax, [(X[3], y2 + H / 2), (X[3], -0.25), (X[1] + 0.3, -0.25), (X[1] + 0.3, y1 - H / 2)],
     "E2 heat", color=HEAT, ls="--", lpos=(X[3] - 0.1, -0.12), ha="right")
arrow(ax, (X[2] + 0.9, 1.15), (X[1] + 0.55, y1 + H / 2), "natural gas heat", color=NG, lpos=0.0, dx=0.05, ha="left", dy=0.08)

fig.text(0.02, 0.012,
         "Functional unit: treatment of 1 t of wet feedstock (Quebec). Background inputs from ecoinvent 3.12 (cut-off), not drawn: "
         "Quebec grid electricity (HTC heating, filter press, dryer,\ncentrifuge, digester, PSA), natural gas heat (digester), "
         "tap water, wastewater treatment and ash disposal. Dashed boxes: avoided products (system expansion); hydrochar heat = 0.76 × HHV. "
         "Red: hot product and recovered heat.", fontsize=9.5, color=GREY, va="bottom")
fig.subplots_adjust(left=0.01, right=0.995, top=0.96, bottom=0.07, hspace=0.12)
fig.savefig(os.path.join(OUT, "FigureS4_system_boundary.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "FigureS4_system_boundary.png"), dpi=600, bbox_inches="tight")
print("Saved FigureS4_system_boundary.pdf / .png ->", OUT)
