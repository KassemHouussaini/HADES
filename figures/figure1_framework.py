"""Figure 1: HADES modelling framework (vector PDF, 600 dpi TIFF and PNG)."""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.dirname(os.path.abspath(__file__))     # this folder

mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42})
C = {"in": "#e8eef7", "ml": "#f6e3c6", "bal": "#e3efe3", "ad": "#d9e8f6", "lca": "#efe2f2",
     "out": "#f2f2f2", "edge": "#2b2b2b", "txt": "#1a1a1a", "note": "#4d4d4d"}
TFS, DFS = 9.0, 7.8
FIG_W, X_UNITS = 14.0, 140.0
U_PER_PT = X_UNITS / (FIG_W * 72)

fig, ax = plt.subplots(figsize=(FIG_W, 2.3))
ax.set_xlim(0, 140); ax.set_ylim(0, 23); ax.set_aspect("equal"); ax.axis("off")

def box(x, y, w, h, title, detail, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=fc, ec=C["edge"], lw=1.0))
    th, dh = 1.25 * TFS * U_PER_PT, 1.22 * DFS * U_PER_PT
    nt, nd = len(title.split("\n")), len(detail.split("\n"))
    top = y + h / 2 + (nt * th + nd * dh + 0.35) / 2
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=TFS, fontweight="bold",
            color=C["txt"], linespacing=1.15)
    ax.text(x + w / 2, top - nt * th - 0.35, detail, ha="center", va="top", fontsize=DFS,
            color=C["txt"], linespacing=1.15)

def arrow(x0, y0, x1, y1, text=None, ty=0.3, va="bottom"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C["edge"], lw=1.1, shrinkA=0, shrinkB=0,
                                mutation_scale=11))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + ty, text, ha="center", va=va, fontsize=7.8,
                style="italic", color=C["note"])

y0, h = 3.5, 12.5
box(0.5, y0, 14.5, h, "User inputs",
    "feedstock, process\nconditions, pathway,\nlocation, LCIA method", C["in"])
arrow(15, y0 + h / 2, 18, y0 + h / 2)

ax.add_patch(FancyBboxPatch((18, 1.5), 47.5, 19, boxstyle="round,pad=0,rounding_size=1.0",
                            fc="none", ec="0.45", lw=1.0, ls="--"))
ax.text(19, 19.8, "Process models (order set by the pathway)", fontsize=8.2, fontweight="bold",
        color=C["note"], va="top")
box(20, 4.5, 14.5, 11, "HTC surrogate", "XGBoost,\n7 outputs", C["ml"])
box(49, 4.5, 14.5, 11, "AD model", "PW-ADM1 (A)\nADM1 (B)", C["ad"])
arrow(34.5, 13, 49, 13, "A: process water")
arrow(49, 7, 34.5, 7, "B: digestate", ty=-0.3, va="top")
ax.text(41.75, 10, "translators", ha="center", va="center", fontsize=7.2, color="0.45")
arrow(65.5, y0 + h / 2, 68.5, y0 + h / 2)

rest = [("Mass and energy\nbalances", "heat integration", C["bal"]),
        ("Inventory builder", "foreground flows,\nsubstitutions", C["lca"]),
        ("LCA (Brightway)", "ecoinvent,\nselected LCIA method", C["lca"]),
        ("Outputs", "process indicators,\nimpacts", C["out"])]
x, bw, step = 68.5, 15.5, 18
for i, (t, d, fc) in enumerate(rest):
    box(x, y0, bw, h, t, d, fc)
    if i < len(rest) - 1:
        arrow(x + bw, y0 + h / 2, x + step, y0 + h / 2)
    x += step

fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
fig.savefig(os.path.join(OUT, "Figure1_HADES_framework.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Figure1_HADES_framework.png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Figure1_HADES_framework.tiff"), dpi=600, bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"})
print("Saved Figure1_HADES_framework.pdf / .png / .tiff (600 dpi) ->", OUT)
plt.show()
