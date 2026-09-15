"""HD-1910 pendulum bench 3D assembly render (cadquery geometry + matplotlib).

Reuses the exact part builders from hd1910_bench_parts.py, assembles them at
their true relative positions (vertical plate, servo against the plate with a
horizontal output shaft, pendulum arm hanging down, M8 weights at 120 mm),
and renders an isometric view with color-coded parts and annotations.

Run: uv run --with cadquery --with matplotlib --with numpy --no-project \
         python scripts/hd1910_bench_assembly3d.py
"""
from __future__ import annotations

from pathlib import Path

import cadquery as cq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hd1910_bench_parts import (
    ARM_LEN, PLATE_T, WEIGHT_BORE, build_arm, build_mount,
)

OUT = Path(__file__).resolve().parent.parent / "assets" / "hd1910_bench"

# servo envelope at the assembled pose: (x, y, z) = (34 wide, 23 thick along
# the plate normal, 20 tall). Back face (y=0 of the servo) sits on the plate.
SERVO = dict(x=34.0, y=23.0, z=20.0)


def poly3d(shape: cq.Shape, color, alpha=1.0, label=None):
    import numpy as np
    verts, faces = shape.tessellate(0.15)
    v = np.array([[p.x, p.y, p.z] for p in verts])  # (N, 3)
    f = np.asarray(faces)                            # (M, 3)
    tris = v[f]                                      # (M, 3, 3)
    pc = Poly3DCollection(list(tris), facecolor=color, edgecolor="none",
                          alpha=alpha, label=label)
    return pc


def assemble():
    """Return a list of (Poly3DCollection, bbox) color-coded by part.

    Assembly frame: vertical mount plate in the XZ plane (thin along y),
    centered at origin. The plate's +y face (y = PLATE_T/2) carries the servo;
    the output shaft is horizontal along +y; the pendulum arm hangs along -z.
    """
    mount = build_mount().val()
    arm = build_arm().val()

    # plate +y face (NOT bbox.ymax — the table stand pollutes the bbox).
    plate_y_face = PLATE_T / 2.0            # = 3.0
    servo_end_y = plate_y_face + SERVO["y"]  # = 26 (servo front face)
    shaft_len = 14.0
    shaft_end_y = servo_end_y + shaft_len    # = 40 (shaft tip)

    servo = (
        cq.Workplane("XY").box(SERVO["x"], SERVO["y"], SERVO["z"])
        .translate((0.0, plate_y_face + SERVO["y"] / 2.0, 0.0)).val()
    )
    shaft = (
        cq.Workplane("XY").circle(4.95 / 2).extrude(shaft_len)
        .rotate((0, 0, 0), (1, 0, 0), -90)   # z-axis cylinder -> +y (horizontal)
        .translate((0.0, servo_end_y, 0.0)).val()
    )
    # build_arm(): hub at origin, shaft along y, arm along -z.
    arm = arm.translate((0.0, shaft_end_y, 0.0))
    disc = (
        cq.Workplane("XY").circle(20).extrude(5)
        .faces(">Z").workplane().hole(WEIGHT_BORE)
        .translate((0.0, shaft_end_y, -ARM_LEN)).val()
    )

    parts = [
        (poly3d(mount, "#9db4c0", label="mount plate + stand"),
         mount.BoundingBox()),
        (poly3d(servo, "#e6b800", alpha=0.95, label="HD-1910 servo"),
         servo.BoundingBox()),
        (poly3d(shaft, "#666666", label="25T output shaft"), shaft.BoundingBox()),
        (poly3d(arm, "#4c9f70", label="pendulum arm (120 mm)"), arm.BoundingBox()),
        (poly3d(disc, "#d97b29", alpha=0.9, label="M8 weight disc"),
         disc.BoundingBox()),
    ]
    return parts


def main():
    parts = assemble()
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    all_bb = []
    for pc, bb in parts:
        ax.add_collection3d(pc)
        all_bb.append(bb)
    # unify bounds
    xs = [b.xmin for b in all_bb] + [b.xmax for b in all_bb]
    ys = [b.ymin for b in all_bb] + [b.ymax for b in all_bb]
    zs = [b.zmin for b in all_bb] + [b.zmax for b in all_bb]
    ax.set_xlim(min(xs) - 5, max(xs) + 5)
    ax.set_ylim(min(ys) - 5, max(ys) + 5)
    ax.set_zlim(min(zs) - 10, max(zs) + 5)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm, shaft)"); ax.set_zlabel("z (mm, up)")
    # axes equal-ish
    ax.set_box_aspect((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
    ax.view_init(elev=18, azim=-55)
    ax.set_title("HD-1910-C001 BAM pendulum bench — 3D assembly")
    # legend from collections
    handles = [pc for pc, _ in parts]
    labels = [pc.get_label() for pc in handles]
    ax.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.9)

    # annotation: swing +-110 deg arc (qualitative)
    import numpy as np
    th = np.linspace(np.deg2rad(90 - 110), np.deg2rad(90 + 110), 40)
    R = ARM_LEN * 0.92
    arc_x = R * np.cos(th)
    arc_z = R * np.sin(th) - 0  # in the xz plane at the shaft y
    y_arc = parts[3][1].center.y
    ax.plot(arc_x, np.full_like(th, y_arc), arc_z, "--", color="crimson", lw=1.2)
    ax.text(0, y_arc + 2, -ARM_LEN * 0.72, r"$\pm 110^\circ$ swing", color="crimson",
            fontsize=9)

    plt.tight_layout()
    out = OUT / "hd1910_bench_assembly3d.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
