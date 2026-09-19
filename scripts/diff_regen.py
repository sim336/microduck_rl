# -*- coding: utf-8 -*-
"""Diff old committed OpenMicroDuck XML vs regenerated XML."""
import xml.etree.ElementTree as ET

B = r"e:\optiDuck\microduck_rl\src\mjlab_microduck\robot\microduck"


def get(p):
    t = ET.parse(p)
    d = {}
    for b in t.getroot().iter("body"):
        i = b.find("inertial")
        if i is not None:
            d[b.get("name")] = (float(i.get("mass")), [float(x) for x in i.get("pos", "0 0 0").split()])
    return d


old = get(B + r"\robot_openmicroduck_groundcontact.xml")
new = get(r"e:\optiDuck\_regen_check.xml")
hdr = "{:<18}{:>8}{:>8}{:>7}   d_com_mm".format("body", "old_g", "new_g", "d_g")
print(hdr)
to = tn = 0.0
for n in sorted(old):
    om, op = old[n]
    nm, np_ = new[n]
    to += om
    tn += nm
    dp = [(np_[i] - op[i]) * 1000 for i in range(3)]
    print("{:<18}{:8.1f}{:8.1f}{:+7.1f}   ({:+.1f},{:+.1f},{:+.1f})".format(
        n, om * 1000, nm * 1000, (nm - om) * 1000, dp[0], dp[1], dp[2]))
print("total: old={:.1f}g new={:.1f}g delta={:+.1f}g ({:+.2f}%)".format(
    to * 1000, tn * 1000, (tn - to) * 1000, (tn / to - 1) * 100))
