# -*- coding: utf-8 -*-
"""Interactive 3D viewer for the OpenMicroDuck model (viser, browser 360 view).

Uses mujoco's COMPILED mesh vertices (m.mesh_vert, which mujoco re-centers),
NOT the raw STL files — the two differ and using the raw STL scatters parts.
"""
import time

import mujoco
import numpy as np
import trimesh
import viser

XML = r"E:\optiDuck\microduck_rl\src\mjlab_microduck\robot\microduck\robot_openmicroduck.xml"

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

server = viser.ViserServer(host="127.0.0.1", port=8080)
print("viser: http://127.0.0.1:8080 (open in browser)", flush=True)


def to_viser(p):
    """mujoco Z-up -> viser Y-up."""
    return np.column_stack([p[:, 0], p[:, 2], -p[:, 1]])


n_geoms = 0
for gi in range(m.ngeom):
    try:
        mid = m.geom_dataid[gi]
        pos = d.geom_xpos[gi]
        rot = d.geom_xmat[gi].reshape(3, 3)
        name = f"g{gi}"

        if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH and mid >= 0:
            va = m.mesh_vertadr[mid]; vn = m.mesh_vertnum[mid]
            fa = m.mesh_faceadr[mid]; fn = m.mesh_facenum[mid]
            verts_local = m.mesh_vert[va:va + vn]          # mujoco re-centered
            faces = m.mesh_face[fa:fa + fn * 3].reshape(-1, 3)  # relative idx
            # filter out-of-range face indices (mujoco can leave sentinel idx)
            valid = (faces >= 0) & (faces < vn)
            faces = faces[valid.all(axis=1)]
            world = pos + (rot @ verts_local.T).T          # mujoco world
            world = to_viser(world)                        # Z-up -> Y-up
            mesh = trimesh.Trimesh(vertices=world, faces=faces)
            server.scene.add_mesh_trimesh(name, mesh)
            n_geoms += 1
        elif m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_BOX:
            s = m.geom_size[gi]
            mesh = trimesh.creation.box(extents=2 * s)
            world = pos + (rot @ mesh.vertices.T).T
            world = to_viser(world)
            mesh.vertices = world
            server.scene.add_mesh_trimesh(name, mesh)
            n_geoms += 1
    except Exception as e:
        print(f"[skip geom{gi}] {e}", flush=True)
    if n_geoms and n_geoms % 10 == 0:
        print(f"progress: {n_geoms} geoms", flush=True)

print(f"added {n_geoms} geoms. 360 drag to rotate.", flush=True)
while True:
    time.sleep(1)