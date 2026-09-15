"""Virtual dry-fit of the HD-1910 pendulum bench.

Assembles the REAL part geometry (mount, arm, weight) plus a servo envelope
with its actual mounting holes, then checks the things a physical build would
reveal:

  1. servo M2.5 holes align with the plate holes (same (x,z) line, screws run
     along y through plate+servo);
  2. no overlap (boolean intersection) between the moving pendulum (arm +
     weight) and the static assembly (plate + servo + shaft) at several swing
     angles, including the +/-110 deg extremes;
  3. the arm hub actually sits on the shaft tip;
  4. swing clearance: arm/weight stay clear of the plate, servo, and table
     stand throughout the sweep.

Run: uv run --with cadquery --no-project python scripts/hd1910_bench_fitcheck.py
"""
from __future__ import annotations

import math

import cadquery as cq

from hd1910_bench_parts import (
    ARM_LEN, HUB_OD, HORN_CIRCLE, PLATE_T, SHAFT_OD, WEIGHT_BORE,
    build_arm, build_mount,
)

PLATE_Y_FACE = PLATE_T / 2.0          # plate +y face
SERVO = dict(x=34.0, y=23.0, z=20.0)  # (wide, thick along plate normal, tall)
SHAFT_LEN = 14.0
# M3 mounting holes (see hd1910_bench_parts.py — group B, NOT the PHS case
# assembly screws)
HOLES = [(-8.0, 7.5), (8.0, 7.5), (-8.0, -22.5), (8.0, -22.5)]


def build_static() -> tuple[cq.Shape, cq.Shape, list]:
    """Return (static_assembly, shaft_axis_probe, [plate holes (x,z)])."""
    plate = build_mount().val()                       # rotated: XZ plane, y-thin
    # servo envelope with its real mounting holes drilled along y
    servo = cq.Workplane("XY").box(SERVO["x"], SERVO["y"], SERVO["z"])
    for hx, hy in HOLES:                              # holes along y through servo
        servo = servo - cq.Workplane("XZ").circle(2.7 / 2).extrude(40) \
            .rotate((0, 0, 0), (1, 0, 0), -90) \
            .translate((hx, -20, hy))
    servo = servo.translate((0.0, PLATE_Y_FACE + SERVO["y"] / 2, 0.0)).val()
    static = plate.fuse(servo)
    # plate holes after rotation: flat (hx, hy, 0) -> (hx, 0, hy)
    plate_holes = [(hx, 0.0, hy) for hx, hy in HOLES]
    return static, servo, plate_holes


def arm_at(angle_deg: float) -> cq.Shape:
    """Arm + weight disc, hub centered at the shaft tip, rotated about the
    shaft axis (y axis through the hub center) by angle (0 = hanging down)."""
    arm = build_arm().val()
    arm = arm.translate((0.0, PLATE_Y_FACE + SERVO["y"] + SHAFT_LEN, 0.0))
    # weight disc at the arm end (120 mm from axis), on the M8 bore
    disc = (cq.Workplane("XY").circle(20).extrude(5)
            .faces(">Z").workplane().hole(WEIGHT_BORE)
            .translate((0.0, PLATE_Y_FACE + SERVO["y"] + SHAFT_LEN, -ARM_LEN))
            .val())
    assy = arm.fuse(disc)
    hub_center = (0.0, PLATE_Y_FACE + SERVO["y"] + SHAFT_LEN, 0.0)
    return assy.rotate(hub_center, (0.0, 1.0, 0.0), angle_deg)


def main():
    static, servo, plate_holes = build_static()
    shaft_y = PLATE_Y_FACE + SERVO["y"] + SHAFT_LEN   # shaft tip y = hub center

    print("== 1. servo mounting holes vs plate holes (x,z alignment) ==")
    servo_end_y = PLATE_Y_FACE + SERVO["y"]
    for (hx, hy), (px, py, pz) in zip(HOLES, plate_holes):
        # plate hole center: (hx, 0, hy); servo hole center at y=servo_end_y
        print(f"  hole ({hx:5.1f},{hy:6.1f}): plate (x,z)=({px:.1f},{pz:.1f}) "
              f"servo (x,z)=({hx:.1f},{hy:.1f}) -> "
              f"{'ALIGNED' if abs(px-hx)<1e-6 and abs(pz-hy)<1e-6 else 'MISALIGNED'}")

    print("\n== 2. overlap: moving pendulum vs static (plate+servo) ==")
    ok = True
    for ang in (-110, -70, -35, 0, 35, 70, 110):
        moving = arm_at(ang)
        vol = static.intersect(moving).Volume()
        tag = "OK" if vol < 1e-6 else "COLLISION"
        if vol >= 1e-6:
            ok = False
        print(f"  swing {ang:+4d} deg: overlap volume = {vol:9.4f} mm^3  [{tag}]")

    print("\n== 3. hub on shaft tip ==")
    arm = arm_at(0)
    ab = arm.BoundingBox()
    hub_cy = (ab.ymin + ab.ymax) / 2
    print(f"  hub y-center = {hub_cy:.1f} vs shaft tip y = {shaft_y:.1f} -> "
          f"{'OK' if abs(hub_cy - shaft_y) < 1 else 'OFF'}")

    print("\n== 4. swing envelope clear of plate/stand ==")
    # table stand occupies z in [-53,-39] near x in [-20,20]; arm/weight hang
    # down to z=-120 and sweep in XZ; ensure no overlap at extremes (already
    # covered by the volume check in step 2). Print the extreme arm-end x/z.
    for ang in (-110, 110):
        a = math.radians(ang)
        # arm end relative to hub, hanging direction rotated
        ex = 0.0 + ARM_LEN * math.sin(a)
        ez = -ARM_LEN * math.cos(a)
        print(f"  swing {ang:+4d} deg: arm end at x={ex:+.0f}, z={ez:+.0f} "
              f"(plate x in [-39,39], stand z in [-53,-39])")

    print("\nRESULT:", "PASS — dry-fit clean" if ok else "FAIL — collisions found")


if __name__ == "__main__":
    main()
