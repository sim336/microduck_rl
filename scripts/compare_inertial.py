# -*- coding: utf-8 -*-
"""Compare inertial (mass / COM / inertia) between official and OpenMicroDuck XML."""
import sys
import xml.etree.ElementTree as ET

BASE = r"e:\optiDuck\microduck_rl\src\mjlab_microduck\robot\microduck"
OFFICIAL = BASE + r"\robot_groundcontact.xml"
OPENMICRODUCK = BASE + r"\robot_openmicroduck_groundcontact.xml"

ET.register_namespace("", "")


def parse(xml_path):
    """Return {name: dict(mass, pos, fullinertia)} for every body."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bodies = {}
    for b in root.iter("body"):
        name = b.get("name", "?")
        inert = b.find("inertial")
        if inert is None:
            bodies[name] = None
            continue
        bodies[name] = {
            "mass": float(inert.get("mass")),
            "pos": [float(x) for x in inert.get("pos", "0 0 0").split()],
            "fullinertia": [float(x) for x in inert.get("fullinertia", "").split()] or None,
        }
    return bodies


def quat_to_mat(q):
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat_vec(m, v):
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def vec_add(a, b):
    return [a[i] + b[i] for i in range(3)]


def collect(xml_path):
    """Build {name: (mass, global_com)} using body transform chain."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out = {}

    def walk(body, R, t):
        name = body.get("name", "?")
        bp = [float(x) for x in body.get("pos", "0 0 0").split()]
        bq = [float(x) for x in body.get("quat", "1 0 0 0").split()]
        Rb = quat_to_mat(bq)
        # body origin in parent frame
        o = mat_vec(R, bp)
        global_o = vec_add(t, o)
        # body frame rotation in world
        Rw = mat_mul(R, Rb)
        inert = body.find("inertial")
        if inert is not None:
            mass = float(inert.get("mass"))
            com = [float(x) for x in inert.get("pos", "0 0 0").split()]
            global_com = vec_add(global_o, mat_vec(Rw, com))
            out[name] = (mass, global_com)
        for child in body.findall("body"):
            walk(child, Rw, global_o)

    worldbody = root.find("worldbody")
    for b in worldbody.findall("body"):
        walk(b, quat_to_mat([1.0, 0.0, 0.0, 0.0]), [0.0, 0.0, 0.0])
    return out


def fmt(v):
    return f"{v * 1000:8.1f}"


def main():
    off = parse(OFFICIAL)
    omd = parse(OPENMICRODUCK)
    off_g = collect(OFFICIAL)
    omd_g = collect(OPENMICRODUCK)

    names = [n for n in omd if omd[n] is not None]
    print(f"{'body':<22}{'off_g':>8}{'omd_g':>8}{'d_mass':>8}   "
          f"{'off_com(mm)':>26}   {'omd_com(mm)':>26}   {'d_com(mm)':>12}")
    tot_off = tot_omd = 0.0
    for n in sorted(names):
        o = off.get(n)
        m = omd.get(n)
        if o is None:
            continue
        dm = (m["mass"] - o["mass"]) * 1000
        dpos = [(m["pos"][i] - o["pos"][i]) * 1000 for i in range(3)]
        tot_off += o["mass"]
        tot_omd += m["mass"]
        print(f"{n:<22}{fmt(o['mass'])}{fmt(m['mass'])}{dm:+8.1f}   "
              f"({o['pos'][0]*1000:7.1f},{o['pos'][1]*1000:7.1f},{o['pos'][2]*1000:7.1f})   "
              f"({m['pos'][0]*1000:7.1f},{m['pos'][1]*1000:7.1f},{m['pos'][2]*1000:7.1f})   "
              f"({dpos[0]:+6.1f},{dpos[1]:+6.1f},{dpos[2]:+6.1f})")

    print()
    print(f"total mass: official={tot_off*1000:.1f} g  OpenMicroDuck={tot_omd*1000:.1f} g  "
          f"delta={ (tot_omd-tot_off)*1000:+.1f} g ({(tot_omd/tot_off-1)*100:+.2f}%)")

    def com_sum(g):
        mt = 0.0
        c = [0.0, 0.0, 0.0]
        for n, (mass, com) in g.items():
            mt += mass
            for i in range(3):
                c[i] += mass * com[i]
        return [x / mt for x in c], mt

    c_off, mt_off = com_sum(off_g)
    c_omd, mt_omd = com_sum(omd_g)
    print(f"global COM: official=({c_off[0]*1000:.1f},{c_off[1]*1000:.1f},{c_off[2]*1000:.1f}) mm")
    print(f"global COM: OpenMicroDuck=({c_omd[0]*1000:.1f},{c_omd[1]*1000:.1f},{c_omd[2]*1000:.1f}) mm")
    print(f"global COM delta: ({(c_omd[0]-c_off[0])*1000:+.1f},{(c_omd[1]-c_off[1])*1000:+.1f},"
          f"{(c_omd[2]-c_off[2])*1000:+.1f}) mm")


if __name__ == "__main__":
    main()
