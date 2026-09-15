"""HD-1910 pendulum bench assembly diagram (matplotlib).

Draws a front view (XZ plane, looking along the shaft axis) and a side view
(YZ plane, looking along the board) with key dimensions annotated, so the
printed parts and assembly are unambiguous.

Run: uv run --with matplotlib --with numpy --no-project python scripts/hd1910_bench_diagram.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrow, Rectangle

OUT = Path(__file__).resolve().parent.parent / "assets" / "hd1910_bench"
OUT.mkdir(parents=True, exist_ok=True)

# geometry (mm) — matches hd1910_bench_parts.py
SERVO_X, SERVO_Y, SERVO_Z = 20, 34, 23
PLATE_H, PLATE_W, PLATE_T = 78, 78, 6
ARM_LEN = 120
HUB_OD, HORN_OD = 22, 17.2
SHAFT_OFF = 0
BASE_H, BASE_D = 14, 40
HOLES = [(4.90, -22.90), (-8.40, -19.40), (-8.40, 4.50), (8.40, 4.50)]


def draw_front(ax):
    """Looking along the shaft axis (from the +y side). Board = rectangle."""
    ax.set_title("Front view (along shaft axis)")
    ax.set_xlabel("x (mm, servo width direction)"); ax.set_ylabel("z (mm, up)")
    # board outline (78 wide x 78 tall)
    ax.add_patch(Rectangle((-PLATE_H / 2, -PLATE_W / 2), PLATE_H, PLATE_W,
                           fill=False, ec="k", lw=1.5, label="mount plate"))
    # shaft axis cross at board center
    ax.plot([0], [0], "k+", ms=12, mew=2)
    # pendulum arm at rest (hanging down), and swing range +/-110 deg
    ax.plot([0, 0], [0, -ARM_LEN], "C1-", lw=4, label="arm (q=0)")
    for ang in (110, -110):
        th = math.radians(90 - ang)  # from +x axis, arm pointing down = 270deg
        dx = math.cos(math.radians(ang + 90)) * 0
        # draw arm at angle: from top (0) rotated
    # swing arcs
    arc = Arc((0, 0), 2 * ARM_LEN, 2 * ARM_LEN, angle=0, theta1=90 - 110,
              theta2=90 + 110, ec="C2", ls="--", lw=1.2)
    ax.add_patch(arc)
    ax.text(ARM_LEN * 0.72, -ARM_LEN * 0.25, r"$\pm 110^\circ$ swing", color="C2")
    # weight bore at arm end
    ax.add_patch(Circle((0, -ARM_LEN), 12, fill=False, ec="C3", lw=1.5,
                        label="M8 weight (0.1/0.3/0.6 kg)"))
    ax.text(16, -ARM_LEN - 2, "120 mm", fontsize=8)
    # mounting holes (x, y) mapped to (x, z) = (x, y)
    for hx, hy in HOLES:
        ax.plot(hx, hy, "ro", ms=4)
    ax.text(12, 20, "4x M2.5 holes\n(from STEP)", fontsize=7, color="r")
    ax.set_xlim(-60, 60); ax.set_ylim(-135, 50)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)


def draw_side(ax):
    """Looking along +x (board edge-on). Shows plate thickness, servo, shaft."""
    ax.set_title("Side view (board edge-on)")
    ax.set_xlabel("y (mm, shaft direction)"); ax.set_ylabel("z (mm, up)")
    # board edge-on: thickness 6 at y=0
    ax.add_patch(Rectangle((-3, -PLATE_W / 2), 6, PLATE_W, fill=False, ec="k",
                           lw=1.5, label="plate (6 mm)"))
    # servo: back face against board (+y side), z from -23..0 (drawn up from 0)
    ax.add_patch(Rectangle((3, 0), SERVO_Z, SERVO_Y, fill="#cfe2f3", ec="b",
                           lw=1.5, label="HD-1910 servo (34x20x23)"))
    # shaft out of servo, +y
    ax.annotate("", xy=(3 + SERVO_Z + 30, SERVO_Y / 2), xytext=(3 + SERVO_Z, SERVO_Y / 2),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="C1"))
    ax.text(3 + SERVO_Z + 34, SERVO_Y / 2 - 10, "output shaft (25T, OD4.95)",
            fontsize=8, color="C1")
    # table stand
    ax.add_patch(Rectangle((-3, -PLATE_W / 2), 3 + BASE_D, BASE_H,
                           fill="#e2e2e2", ec="k", lw=1, label="table stand"))
    ax.text(10, -PLATE_W / 2 - 2, "stands on the table", fontsize=8)
    # ground line
    ax.plot([-5, 60], [-PLATE_W / 2 - BASE_H, -PLATE_W / 2 - BASE_H], "k-", lw=2)
    # mount screws through servo + plate
    ax.annotate("", xy=(3, 18), xytext=(3 + SERVO_Z + 4, 18),
                arrowprops=dict(arrowstyle="<->", color="r", lw=1.2))
    ax.text(6, 24, "4x M2.5 screws\nservo -> plate,\nnut on back", fontsize=7, color="r")
    ax.set_xlim(-10, 70); ax.set_ylim(-55, 50)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
draw_front(ax1)
draw_side(ax2)
for ax in (ax1, ax2):
    ax.legend(loc="upper left", fontsize=7)
fig.suptitle("HD-1910-C001 BAM pendulum bench — assembly", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "hd1910_bench_assembly.png", dpi=150, bbox_inches="tight")
print(f"saved {OUT / 'hd1910_bench_assembly.png'}")
