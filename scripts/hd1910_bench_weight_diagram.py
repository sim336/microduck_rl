"""Weight-end assembly detail diagram (side view, y-z plane).

Shows how the M8 rod, steel discs and nuts fix the pendulum weight at the arm
end. Run: uv run --with matplotlib --no-project python scripts/hd1910_bench_weight_diagram.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

OUT = Path(__file__).resolve().parent.parent / "assets" / "hd1910_bench"
OUT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 7))

# ---- coordinate sketch (side view, y horizontal, z vertical) ----
# arm end boss (vertical bar at z=-120 area), M8 rod horizontal along y
boss_y0, boss_y1 = 31, 49          # boss spans y (arm width)
bore_z = -120                      # M8 bore center z
rod_r = 4.0                        # M8 rod radius

# arm end boss cross-section (x-thick 12, drawn as vertical slab)
ax.add_patch(Rectangle((boss_y0, bore_z - 10), boss_y1 - boss_y0, 20,
                       fc="#cfe2f3", ec="k", lw=1.5, label="arm end boss (Ø8.4 bore)"))

# M8 rod (horizontal line through the bore, extends both sides)
rod_yl, rod_yr = boss_y0 - 6, boss_y1 + 6
ax.plot([rod_yl, rod_yr], [bore_z, bore_z], "k-", lw=5, solid_capstyle="butt",
        label="M8 rod")

# weight discs on the rod (right side), steel washers Ø40 x 5
discs_r = 3
for i in range(discs_r):
    cy = boss_y1 + 4 + i * 6
    ax.add_patch(Circle((cy, bore_z), 20, fc="#d97b29", ec="k", lw=1, alpha=0.85))
    ax.add_patch(Circle((cy, bore_z), 4.2, fc="white", ec="k", lw=1))
discs_l = 2
for i in range(discs_l):
    cy = boss_y0 - 4 - i * 6
    ax.add_patch(Circle((cy, bore_z), 20, fc="#d97b29", ec="k", lw=1, alpha=0.7))
    ax.add_patch(Circle((cy, bore_z), 4.2, fc="white", ec="k", lw=1))

# nuts + washers at both rod ends
for ey in (rod_yl, rod_yr):
    ax.add_patch(Rectangle((ey - 3, bore_z - 7), 6, 14, fc="#888", ec="k",
                           lw=1, label="M8 nut + washer" if ey == rod_yl else None))

# annotations
ax.annotate("", xy=(boss_y1 + 4 + discs_r * 6 + 6, bore_z),
            xytext=(boss_y1 + 4, bore_z),
            arrowprops=dict(arrowstyle="<->", color="r", lw=1.2))
ax.text(boss_y1 + 4 + discs_r * 3, bore_z + 8, "steel discs on rod\n(each Ø40x5 ≈ 47 g)",
        fontsize=8, color="r", ha="center")
ax.annotate("", xy=(boss_y1, bore_z), xytext=(boss_y0, bore_z),
            arrowprops=dict(arrowstyle="<->", color="b", lw=1.2))
ax.text((boss_y0 + boss_y1) / 2, bore_z + 9, "Ø8.4 bore\n(arm)", fontsize=8, color="b", ha="center")
ax.text(rod_yl - 4, bore_z + 12, "nut locks the rod\nin the bore", fontsize=7, ha="right", color="#666")
ax.text(rod_yr + 4, bore_z - 14, "end nut presses\nthe discs tight", fontsize=7, ha="left", color="#666")

ax.axhline(bore_z - 26, color="k", lw=0.5, ls=":")
ax.text(boss_y1 + 10, bore_z - 28, "pendulum CoM sits at 120 mm from the shaft axis",
        fontsize=8, color="#333")

ax.set_xlim(boss_y0 - 18, boss_y1 + 40)
ax.set_ylim(bore_z - 34, bore_z + 34)
ax.set_aspect("equal")
ax.set_xlabel("y (mm, shaft direction)")
ax.set_ylabel("z (mm, up)")
ax.set_title("Weight-end fixation: M8 rod + steel discs + nuts")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "hd1910_bench_weight_detail.png", dpi=150)
print(f"saved {OUT / 'hd1910_bench_weight_detail.png'}")
