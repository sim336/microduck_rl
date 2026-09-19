# -*- coding: utf-8 -*-
"""Decompose per-body mass: XML value vs generator formula, body by body.

Shows each mesh volume, servo count, 'other' back-calc, to locate the
source of mismatch between committed XML and a re-run of hd1910_gen_mjcf.py.
"""
import os
import xml.etree.ElementTree as ET
import trimesh

ROBOT = r"e:\optiDuck\microduck_rl\src\mjlab_microduck\robot\microduck"
SRC = os.path.join(ROBOT, "robot_walk.xml")
NEW = os.path.join(ROBOT, "robot_openmicroduck_groundcontact.xml")
PARTS = r"e:\optiDuck\OpenMicroDuck\cad\parts"

RESIN = 1.15
OFF_DENS = 1.05
XL = 18.0
HD = 21.0

_cache = {}


def vol(name):
    if name in _cache:
        return _cache[name]
    p = os.path.join(PARTS, f"{name}.stl")
    if not os.path.exists(p):
        _cache[name] = None
        return None
    m = trimesh.load(p)
    v = abs(m.volume) if m.volume is not None else 0.0
    _cache[name] = v / 1000.0
    return _cache[name]


def parse(xml):
    t = ET.parse(xml)
    out = {}
    for b in t.getroot().iter("body"):
        inert = b.find("inertial")
        if inert is None:
            continue
        out[b.get("name")] = {
            "mass": float(inert.get("mass")) * 1000,
            "geoms": [(g.get("mesh"), g.get("type")) for g in b.findall("geom")],
        }
    return out


old_b = parse(SRC)
new_b = parse(NEW)

print(f"{'body':<18}{'old_g':>7}{'xml_g':>8}{'recalc_g':>9}{'diff':>7}  details")
for name, nb in new_b.items():
    ob = old_b[name]
    part_mass, servo_n, skip = 0.0, 0, []
    per_mesh = []
    for mesh, gtype in nb["geoms"]:
        if not mesh:
            continue
        if mesh == "xl330":
            servo_n += 1
            per_mesh.append(f"xl330")
        else:
            v = vol(mesh)
            if v is None:
                skip.append(mesh)
                per_mesh.append(f"{mesh}:NO_STL")
            else:
                part_mass += v * RESIN
                per_mesh.append(f"{mesh}:{v:.1f}cm3")
    off_part = sum((vol(m) or 0.0) * OFF_DENS for m, _ in nb["geoms"]
                   if m and m != "xl330")
    other = max(0.0, ob["mass"] - off_part - servo_n * XL)
    recalc = part_mass + servo_n * HD + other
    d = nb["mass"] - recalc
    print(f"{name:<18}{ob['mass']:7.1f}{nb['mass']:8.1f}{recalc:9.1f}{d:+7.1f}  "
          f"servo={servo_n} parts={part_mass:.1f}g other={other:.1f}g")
    print(f"{'':<18}{'':<25}{' '.join(per_mesh)}")
    if skip:
        print(f"{'':<18}  MISSING STL: {skip}")
