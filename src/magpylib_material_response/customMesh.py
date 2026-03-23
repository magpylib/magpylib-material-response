from __future__ import annotations

import os

import magpylib as magpy
import meshio
import numpy as np
from scipy.spatial.transform import Rotation as R


def make_cuboid_global_pol(center, dims, A, J_global):
    """
    center   : (3,) global position
    dims     : (3,) cuboid dimensions
    A        : (3,3) rotation matrix, local axes as columns, local->global
    J_global : (3,) desired polarization in global coordinates
    """
    rot = R.from_matrix(A)

    # convert desired GLOBAL polarization into LOCAL coordinates
    J_local = rot.inv().apply(J_global)

    return magpy.magnet.Cuboid(
        dimension=tuple(dims),
        position=tuple(center),
        orientation=rot,
        polarization=tuple(J_local),
    )


def frame_and_dims_from_hex_edges(coords, scaling=1, eps=1e-12):
    """
    coords: (8,3) hexahedron node coordinates
    returns:
        center : (3,)
        dims   : (3,)  -> edge lengths
        A      : (3,3) rotation matrix, columns are local axes
    """
    coords = coords * scaling
    coords = np.asarray(coords, dtype=float)
    center = coords.mean(axis=0)

    # pick one corner
    p0 = coords[0]

    # distances from p0 to all other nodes
    d = np.linalg.norm(coords - p0, axis=1)

    # exclude p0 itself
    idx = np.argsort(d)

    # the 3 nearest nonzero-distance nodes are the 3 edge neighbors
    edge_ids = [i for i in idx if d[i] > eps][:3]

    v1 = coords[edge_ids[0]] - p0
    v2 = coords[edge_ids[1]] - p0
    v3 = coords[edge_ids[2]] - p0

    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)
    l3 = np.linalg.norm(v3)

    if min(l1, l2, l3) < eps:
        msg = "Degenerate hex element with zero edge length."
        raise ValueError(msg)

    e1 = v1 / l1
    e2 = v2 / l2

    # make orthonormal frame
    # start with e1
    a1 = e1

    # remove component of e2 along a1
    e2p = e2 - np.dot(e2, a1) * a1
    n2 = np.linalg.norm(e2p)
    if n2 < eps:
        msg = "Hex edges are not linearly independent."
        raise ValueError(msg)
    a2 = e2p / n2

    # third axis from cross product to enforce orthogonality/right-handedness
    a3 = np.cross(a1, a2)
    n3 = np.linalg.norm(a3)
    if n3 < eps:
        msg = "Failed to build right-handed frame."
        raise ValueError(msg)
    a3 = a3 / n3

    A = np.column_stack([a1, a2, a3])

    # project all vertices into local frame and get exact box dimensions
    local = (coords - center) @ A
    dims = local.max(axis=0) - local.min(axis=0)

    return center, dims, A


def make_oriented_cuboids_from_hex(
    mesh, cell_key="hexahedron", polarization=(0, 0, 0), scaling=1, eps=1e-9
):
    pts = mesh.points

    # Check if requested cell type exists
    if cell_key not in mesh.cells_dict:
        available = list(mesh.cells_dict.keys())
        msg = (
            f"Cell type '{cell_key}' not found in mesh. "
            f"Available cell types: {available}"
        )
        raise ValueError(msg)

    cells = mesh.cells_dict[cell_key]
    J_global = np.asarray(polarization, dtype=float)
    mags = []
    for elem in cells:
        coords = pts[elem]
        center, dims, A = frame_and_dims_from_hex_edges(coords, scaling=scaling)

        rot = R.from_matrix(A)

        # avoid zeros
        dims = np.maximum(dims, eps)
        J_local = rot.inv().apply(J_global)

        m = magpy.magnet.Cuboid(
            dimension=tuple(dims.tolist()),
            position=tuple(center.tolist()),
            polarization=tuple(J_local),
            orientation=R.from_matrix(A),  # Magpylib accepts scipy Rotation
        )
        mags.append(m)
    return mags


def import_mesh(mesh_file, scaling=1, polarization=(0, 0, 0), succeptibility=None):
    valid_extensions = {".inp", ".msh"}
    ext = Path(mesh_file).suffix.lower()

    if ext not in valid_extensions:
        msg = f"Unsupported file format '{ext}'. Only .inp and .msh are allowed."
        raise ValueError(msg)

    ext = os.path.splitext(mesh_file)[1].lower()

    if ext not in valid_extensions:
        raise ValueError(
            f"Unsupported file format '{ext}'. Only .inp and .msh are allowed."
        )

    mesh = meshio.read(mesh_file)

    magnets = make_oriented_cuboids_from_hex(
        mesh, cell_key="hexahedron", polarization=polarization, scaling=scaling
    )
    for magnet in magnets:
        magnet.susceptibility = succeptibility
    # magnet.polarization   = polarization
    return magpy.Collection(magnets)
