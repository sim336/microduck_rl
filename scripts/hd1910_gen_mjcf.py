# -*- coding: utf-8 -*-
"""Generate OpenMicroDuck MJCF from the official model.

- keeps the official body tree / joints / joint limits / mesh geoms
  (structure meshes are already the OpenMicroDuck parts, same filenames);
- recomputes every body inertial with:
    * printed parts  -> shell volume x resin density 1.15 g/cm3
    * servo          -> HD-1910-C001 21 g each (was XL330 18 g)
    * other parts    -> kept as in the official model
- inertia tensor scaled by the mass ratio (first-order approximation).
"""
import math
import sys
from pathlib import Path
import os
import re
import xml.etree.ElementTree as ET

import trimesh

ROBOT = Path(__file__).resolve().parents[1] / "src" / "mjlab_microduck" / "robot" / "microduck"
SRC_XML = str(ROBOT / "robot_walk.xml")
OUT_XML = str(ROBOT / "robot_openmicroduck.xml")
PARTS = None  # set via --parts-dir or env OPENMICRODUCK_PARTS

SERVO_MESHES = {"xl330", "hd1910"}   # mesh names used for the servo bodies
RESIN_DENS = 1.15       # g/cm3 (high-toughness resin)
XL330_MASS = 18.0       # g
HD1910_MASS = 21.0      # g
OFFICIAL_PART_DENS = 1.05  # assumed official print density (ABS) for the
                           # "other mass" back-calculation


def shell_volume(name):
    p = os.path.join(PARTS, f"{name}.stl")
    if not os.path.exists(p):
        return None
    m = trimesh.load(p)
    v = abs(m.volume) if m.volume is not None else 0.0
    return v / 1000.0  # mm3 -> cm3


def set_inertial(body, mass, inertia, pos, scale=1.0):
    inert = body.find("inertial")
    if inert is None:
        inert = ET.SubElement(body, "inertial")
    inert.set("mass", f"{mass:.6f}")
    # scale existing inertia tensor by mass ratio (approximation)
    if inertia:
        vals = [float(x) * scale for x in inertia]
        inert.set("fullinertia", " ".join(f"{v:.9g}" for v in vals))
    if pos:
        inert.set("pos", pos)


def main(parts_dir, output=None):
    global PARTS, OUT_XML
    PARTS = parts_dir
    if output:
        OUT_XML = output
    else:
        OUT_XML = str(ROBOT / "robot_openmicroduck.xml")
    ET.register_namespace("", "")
    tree = ET.parse(SRC_XML)
    root = tree.getroot()

    total_old, total_new = 0.0, 0.0
    servo_cnt = 0
    for body in root.iter("body"):
        name = body.get("name", "?")
        inert = body.find("inertial")
        old_mass = float(inert.get("mass")) if inert is not None else 0.0
        old_ia = None
        old_pos = None
        if inert is not None:
            old_ia = inert.get("fullinertia", "").split()
            old_pos = inert.get("pos")

        part_mass, servo_n = 0.0, 0
        for g in body.findall("geom"):
            mesh = g.get("mesh")
            if not mesh:
                continue
            if mesh in SERVO_MESHES:
                servo_n += 1
            else:
                v = shell_volume(mesh)
                if v is not None:
                    part_mass += v * RESIN_DENS
        # "other" mass kept from official: old - official parts - old servos
        off_part = 0.0
        for g in body.findall("geom"):
            mesh = g.get("mesh")
            if mesh and mesh not in SERVO_MESHES:
                v = shell_volume(mesh)
                if v is not None:
                    off_part += v * OFFICIAL_PART_DENS
        other = max(0.0, old_mass - off_part / 1000.0 - servo_n * XL330_MASS / 1000.0)
        new_mass = part_mass / 1000.0 + servo_n * HD1910_MASS / 1000.0 + other
        scale = new_mass / old_mass if old_mass > 0 else 1.0

        set_inertial(body, new_mass, old_ia, old_pos, scale)
        total_old += old_mass
        total_new += new_mass
        servo_cnt += servo_n
        print(f"{name:22s} old={old_mass*1000:7.1f}g new={new_mass*1000:7.1f}g "
              f"(parts {part_mass:.1f}g, servo x{servo_n})")

    print(f"\nservos: {servo_cnt} | total old={total_old*1000:.0f}g new={total_new*1000:.0f}g")
    tree.write(OUT_XML, encoding="utf-8", xml_declaration=True)
    print("saved:", OUT_XML)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-dir", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    pd = args.parts_dir or os.environ.get("OPENMICRODUCK_PARTS")
    if not pd:
        raise SystemExit("set --parts-dir or OPENMICRODUCK_PARTS to the OpenMicroDuck cad/parts dir")
    main(pd, args.output)
