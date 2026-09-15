"""HD-1910-C001 pendulum-bench 3D printable parts (cadquery).

Generates STL files for the BAM identification rig (see
src/mjlab_microduck/robot/hd1910_testbench_plan.md):

  hd1910_bench_mount.stl   mount plate + table stand. The servo's z=-23 mm
                           face sits against the plate; 4x M2.5 cap screws pass
                           through the servo (z direction) and the plate,
                           nutted on the back. Output shaft is free on the
                           +z side — nothing blocks the arm swing. Rotated so
                           the shaft axis is horizontal on the printed part.
  hd1910_bench_arm.stl     120 mm pendulum arm: 25T horn hub (OD4.95 bore +
                           M3 grub screw) at one end, M8 weight-bore at the
                           other (CoM 120 mm from shaft axis). Shaft axis is
                           horizontal after the rotation.
  hd1910_bench_weight.stl  reference M8 weight disc (PLA is too light for the
                           real masses — buy steel washers; this disc is a
                           prototype/spacer).

Servo frame: 34 (y) x 20 (x) x 23 (z) mm; output shaft at (x,y)=(0,0) on the
z=0 face; mounting screws run along z. Hole coordinates measured from the
HL-2915 STEP (cad/HD-1910-C001-20260902.stp, same mechanical interface):

  HOLES = (x, y) = (4.90,-22.90), (-8.40,-19.40), (-8.40,4.50), (8.40,4.50)

VERIFY the hole pattern against the real servo before printing.

All bores are boolean subtractions (no face-selection ambiguity). Parts are
built flat in the XY plane (thickness along z) then rotated +90 deg about X
so the shaft axis ends up horizontal, matching the pendulum rig.

Run:  uv run --with cadquery --no-project python scripts/hd1910_bench_parts.py
"""
from __future__ import annotations

from pathlib import Path

import cadquery as cq

OUT = Path(__file__).resolve().parent.parent / "assets" / "hd1910_bench"
OUT.mkdir(parents=True, exist_ok=True)

# ---- servo / mounting geometry (mm) ----
SHAFT_OD = 4.95           # 25T horn bore
# SERVO MOUNTING HOLES (M3, through the full 23 mm shell) — measured from the
# HL-2915 STEP cylindrical faces, group B: perfectly symmetric 4-hole pattern
# at (±8.0, 7.5 / -22.5), Ø3.0 through-hole with a top counterbore.
# NOTE: the earlier pattern (4.90,-22.90) etc. was the PHS CASE-ASSEMBLY screw
# holes (group A, asymmetric) — NOT the mounting holes. Fixed on 2026-09-09
# after a cylindrical-face audit; still VERIFY against the real servo.
HOLES = [(-8.0, 7.5), (8.0, 7.5), (-8.0, -22.5), (8.0, -22.5)]
HOLE_M3 = 3.4             # M3 clearance
CBORE_D = 6.5             # M3 countersink
CBORE_DEPTH = 2.5

# ---- mount plate (flat: x width, y height, z thickness) ----
PLATE_W = 78.0            # x (wide)
PLATE_H = 78.0            # y (tall)
PLATE_T = 6.0             # z (thick)
SHAFT_CLEAR = 14.0        # shaft + horn shoulder clearance
BASE_H = 14.0             # table stand height (y)
BASE_D = 40.0             # table stand depth (x)

# ---- arm (flat: shaft along z in build, rotated to horizontal) ----
ARM_LEN = 120.0           # shaft axis -> weight bore CoM
ARM_W = 18.0              # z width of the arm body (flat build)
ARM_T = 6.0               # x thickness of the arm body
HUB_OD = 36.0             # adapter-disc diameter (covers the M3 hole circle)
HUB_W = 18.0              # z width of the hub disc == ARM_W (flush joint, no
                          # ledge sticking out of the servo-facing face)
GRUB_M3 = 3.2             # M3 grub screw (locks directly on the 25T shaft)
HORN_M3 = 3.4             # M3 clearance for mounting to a 25T metal horn
HORN_CIRCLE = 28.0        # M3 hole circle diameter (common cross-horn pattern)
WEIGHT_BORE = 8.4         # M8 clearance
BOSS_T = 12.0             # x thickness of the weight boss (>= bore dia)
BOSS_LEN = 25.0           # boss length along the arm


def cutter_z(d: float, cx: float = 0.0, cy: float = 0.0,
             length: float = 120.0) -> cq.Workplane:
    """Cylinder along +z for boolean subtraction, centered at (cx, cy)."""
    return (
        cq.Workplane("XY").circle(d / 2).extrude(length)
        .translate((cx, cy, -length / 2))
    )


def cutter_y(d: float, cx: float = 0.0, cz: float = 0.0,
             length: float = 80.0) -> cq.Workplane:
    """Cylinder along +y for boolean subtraction, centered at (cx, cz).

    Built from a +z cylinder rotated -90 deg about X (XY-plane extrude goes
    along +z; XZ-plane extrude goes along -y, which silently misplaced the
    grub bore before this fix).
    """
    return (
        cq.Workplane("XY").circle(d / 2).extrude(length)
        .rotate((0, 0, 0), (1, 0, 0), -90)   # z-axis -> +y axis
        .translate((cx, -length / 2, cz))
    )


def build_mount() -> cq.Workplane:
    """Mount plate (flat build) + table stand; rotated so it stands up."""
    plate = cq.Workplane("XY").box(PLATE_W, PLATE_H, PLATE_T)
    # shaft clearance at the origin (through the plate, along z)
    plate = plate - cutter_z(SHAFT_CLEAR, 0, 0)
    # counterbored M2.5 holes at the measured offsets
    for hx, hy in HOLES:
        plate = plate - cutter_z(HOLE_M3, hx, hy)
        # counterbore pocket from the +z (servo-facing) side
        cb = (
            cq.Workplane("XY").circle(CBORE_D / 2).extrude(CBORE_DEPTH)
            .translate((hx, hy, PLATE_T / 2 - CBORE_DEPTH / 2))
        )
        plate = plate - cb
    # table stand under the plate: wide along x (matching the plate width),
    # deep (front-back) 40 mm centered on the plate so it stays clear of the
    # pendulum (arm lives at y >= 31; stand spans y in [-20, 20] after rotation).
    stand = cq.Workplane("XY").box(PLATE_W, BASE_H, BASE_D).translate(
        (0, -PLATE_H / 2 - BASE_H / 2, 0)
    )
    # rotate +90 about X: flat build -> vertical plate, thickness now along y,
    # shaft axis horizontal (along z).
    return plate.union(stand).rotate((0, 0, 0), (1, 0, 0), 90)


def build_arm() -> cq.Workplane:
    """120 mm pendulum arm; rotated so the shaft axis is horizontal.

    Flat-build layout (before the +90 rotation about X):
      - hub: cylinder along z (shaft axis), centered at origin;
      - body: centered on the axis (x-symmetric), hangs along -y;
      - weight boss: thicker (x) block at the arm end so the M8 bore is a
        FULL circle (bore dia 8.4 > arm thickness 6).
    After rotation: shaft axis along y (horizontal), arm hangs along -z.
    """
    hub = cq.Workplane("XY").circle(HUB_OD / 2).extrude(HUB_W)
    hub = hub.translate((0, 0, -HUB_W / 2))
    # x-symmetric arm body: x in [-ARM_T/2, +ARM_T/2], y from 0 down to -ARM_LEN
    body = cq.Workplane("XY").box(ARM_T, ARM_LEN, ARM_W).translate(
        (0, -ARM_LEN / 2, 0)
    )
    # weight boss at the arm end, x-thick enough for a full M8 bore
    boss = cq.Workplane("XY").box(BOSS_T, BOSS_LEN, ARM_W).translate(
        (0, -ARM_LEN, 0)
    )
    part = hub.union(body).union(boss)
    # 25T bore along the shaft axis (z in flat build) -> -y after rotation
    part = part - cutter_z(SHAFT_OD, 0, 0, length=HUB_W + 30)
    # M3 grub-screw bore through the hub, perpendicular to the shaft
    part = part - cutter_y(GRUB_M3, 0, 0)
    # 4x M3 horn-mounting holes on HORN_CIRCLE (to bolt the printed arm onto
    # a 25T metal cross-horn; countersunk on the +z face). Common cross-horn
    # patterns put the screws on the 4 diagonals.
    import math as _math
    for k in range(4):
        th = _math.radians(45 + 90 * k)
        cx, cy = HORN_CIRCLE / 2 * _math.cos(th), HORN_CIRCLE / 2 * _math.sin(th)
        part = part - cutter_z(HORN_M3, cx, cy, length=HUB_W + 20)
        cs = (
            cq.Workplane("XY").circle(6.5 / 2).extrude(1.5)
            .translate((cx, cy, HUB_W / 2 - 1.5))
        )
        part = part - cs
    # M8 weight bore at the arm end (120 mm from the shaft axis) -> -y after rotation
    part = part - cutter_z(WEIGHT_BORE, 0, -ARM_LEN, length=50)
    return part.rotate((0, 0, 0), (1, 0, 0), 90)


def build_weight_disc(t: float = 5.0, od: float = 40.0) -> cq.Workplane:
    """Reference M8 weight disc (PLA density makes real masses impractical —
    use steel washers; this disc is a spacer/prototype)."""
    return (
        cq.Workplane("XY").circle(od / 2).extrude(t)
        .faces(">Z").workplane().hole(WEIGHT_BORE)
    )


if __name__ == "__main__":
    mount = build_mount()
    arm = build_arm()
    disc = build_weight_disc()

    cq.exporters.export(mount, str(OUT / "hd1910_bench_mount.stl"))
    cq.exporters.export(arm, str(OUT / "hd1910_bench_arm.stl"))
    cq.exporters.export(disc, str(OUT / "hd1910_bench_weight.stl"))

    for name, part in (("mount", mount), ("arm", arm), ("disc", disc)):
        bb = part.val().BoundingBox()
        sx, sy, sz = bb.xlen, bb.ylen, bb.zlen
        print(f"{name}: size x={sx:.1f} y={sy:.1f} z={sz:.1f}")
    print(f"STL written to {OUT}")
