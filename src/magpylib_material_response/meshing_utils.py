from __future__ import annotations

from itertools import product

import magpylib as magpy
import magpylib.graphics.model3d as _m3d
import numpy as np


def apportion_triple(triple, min_val=1, max_iter=30):
    """Apportion values of a triple, so that the minimum value `min_val` is respected
    and the product of all values remains the same.
    Example: apportion_triple([1,2,50], min_val=3)
    -> [ 2.99999999  3.         11.11111113]
    """
    triple = np.abs(np.array(triple, dtype=float))
    count = 0
    while any(n < min_val for n in triple) and count < max_iter:
        count += 1
        amin, amax = triple.argmin(), triple.argmax()
        factor = min_val / triple[amin]
        if triple[amax] >= factor * min_val:
            triple /= factor**0.5
            triple[amin] *= factor**1.5
    return triple


def cells_from_dimension(
    dim,
    target_elems,
    min_val=1,
    strict_max=False,
    parity=None,
):
    """Divide a dimension triple with a target scalar of elements, while apportioning
    the number of elements based on the dimension proportions. The resulting divisions
    are the closest to cubes.

    Parameters
    ----------
    dim: array_like of length 3
        Dimensions of the object to be divided.
    target_elems: int,
        Total number of elements as target for the procedure. Actual final number is
        likely to differ.
    min_val: int
        Minimum value of the number of divisions per dimension.
    strict_max: bool
        If `True`, the `target_elem` value becomes a strict maximum and the product of
        the resulting triple will be strictly smaller than the target.
    parity: {None, 'odd', 'even'}
        All elements of the resulting triple will match the given parity. If `None`, no
        parity check is performed.

    Returns
    -------
    numpy.ndarray of length 3
        array corresponding of the number of divisions for each dimension

    Examples
    --------
    >>> cells_from_dimension([1, 2, 6], 926, parity=None, strict_max=True)
    [ 4  9 25]  # Actual total: 900
    >>> cells_from_dimension([1, 2, 6], 926, parity=None, strict_max=False)
    [ 4  9 26]  # Actual total: 936
    >>> cells_from_dimension([1, 2, 6], 926, parity='odd', strict_max=True)
    [ 3 11 27]  # Actual total: 891
    >>> cells_from_dimension([1, 2, 6], 926, parity='odd', strict_max=False)
    [ 5  7 27]  # Actual total: 945
    >>> cells_from_dimension([1, 2, 6], 926, parity='even', strict_max=True)
    [ 4  8 26]  # Actual total: 832
    >>> cells_from_dimension([1, 2, 6], 926, parity='even', strict_max=False)
    [ 4 10 24]  # Actual total: 960
    """
    elems = np.prod(target_elems)  # in case target_elems is an iterable

    # define parity functions
    if parity == "odd":
        funcs = [
            lambda x, add=add, fn=fn: int(2 * fn(x / 2) + add)
            for add in (-1, 1)
            for fn in (np.ceil, np.floor)
        ]
    elif parity == "even":
        funcs = [lambda x, fn=fn: int(2 * fn(x / 2)) for fn in (np.ceil, np.floor)]
    else:
        funcs = [np.ceil, np.floor]

    # make sure the number of elements is sufficient
    elems = max(min_val**3, elems)

    # float estimate of the elements while product=target_elems and proportions are kept
    x, y, z = np.abs(dim)
    a = x ** (2 / 3) * (elems / y / z) ** (1 / 3)
    b = y ** (2 / 3) * (elems / x / z) ** (1 / 3)
    c = z ** (2 / 3) * (elems / x / y) ** (1 / 3)
    a, b, c = apportion_triple((a, b, c), min_val=min_val)
    epsilon = elems
    # run all combinations of rounding methods, including parity matching to find the
    # closest triple with the target_elems constrain
    result = [funcs[0](k) for k in (a, b, c)]  # first guess
    for fns in product(*[funcs] * 3):
        res = [f(k) for f, k in zip(fns, (a, b, c), strict=False)]
        epsilon_new = elems - np.prod(res)
        if (
            np.abs(epsilon_new) <= epsilon
            and all(r >= min_val for r in res)
            and (not strict_max or epsilon_new >= 0)
        ):
            epsilon = np.abs(epsilon_new)
            result = res
    return np.array(result).astype(int)


def get_volume(obj, return_containing_cube_edge=False):
    """Return object volume in m³ (SI units). The `containing_cube_edge` is the minimum
    side length of an unrotated cube centered at the origin containing the object.
    """
    if obj.__class__.__name__ == "Cuboid":
        dim = obj.dimension
        vol = dim[0] * dim[1] * dim[2]
        containing_cube_edge = max(obj.dimension)
    elif obj.__class__.__name__ == "Cylinder":
        d, h = obj.dimension
        vol = h * np.pi * (d / 2) ** 2
        containing_cube_edge = max(d, h)
    elif obj.__class__.__name__ == "CylinderSegment":
        r1, r2, h, phi1, phi2 = obj.dimension
        vol = h * np.pi * (r2**2 - r1**2) * (phi2 - phi1) / 360
        containing_cube_edge = max(h, 2 * r2)
    elif obj.__class__.__name__ == "Sphere":
        vol = 4 / 3 * np.pi * (obj.diameter / 2) ** 3
        containing_cube_edge = obj.diameter
    elif obj.__class__.__name__ == "TriangularMesh":
        vol = obj.volume
        containing_cube_edge = float(
            np.max(obj.vertices.max(axis=0) - obj.vertices.min(axis=0))
        )
    else:
        msg = "Unsupported object type for volume calculation"
        raise TypeError(msg)
    if return_containing_cube_edge:
        return vol, containing_cube_edge
    return vol


def mask_inside_Cuboid(obj, positions, tolerance=1e-14):
    """Return mask of provided positions inside a Cuboid"""
    a, b, c = obj.dimension / 2
    x, y, z = positions.T
    mx = (abs(x) - a) < tolerance * a
    my = (abs(y) - b) < tolerance * b
    mz = (abs(z) - c) < tolerance * c
    return mx & my & mz


def mask_inside_Cylinder(obj, positions, tolerance=1e-14):
    """Return mask of provided positions inside a Cylinder"""
    # transform to Cy CS
    x, y, z = positions.T
    r, _phi = np.sqrt(x**2 + y**2), np.arctan2(y, x)
    r0, z0 = obj.dimension.T / 2

    # scale invariance (make dimensionless)
    r = np.copy(r / r0)
    z = np.copy(z / r0)
    z0 = np.copy(z0 / r0)

    m2 = np.abs(z) <= z0 + tolerance  # in-between top and bottom plane
    m3 = r <= 1 + tolerance  # inside Cylinder hull plane

    return m2 & m3


def mask_inside_Sphere(obj, positions, tolerance=1e-14):
    """Return mask of provided positions inside a Sphere"""
    x, y, z = np.copy(positions.T)
    r = np.sqrt(x**2 + y**2 + z**2)  # faster than np.linalg.norm
    r0 = abs(obj.diameter) / 2
    return r - r0 < tolerance


def mask_inside_CylinderSegment(obj, positions, tolerance=1e-14):
    """Return mask of provided positions inside a CylinderSegment"""

    r1, r2, h, phi1, phi2 = obj.dimension.T
    r1 = abs(r1)
    r2 = abs(r2)
    h = abs(h)
    z1, z2 = -h / 2, h / 2

    # transform dim deg->rad
    phi1 = phi1 / 180 * np.pi
    phi2 = phi2 / 180 * np.pi

    # transform obs_pos to Cy CS --------------------------------------------
    x, y, z = positions.T
    r, phi = np.sqrt(x**2 + y**2), np.arctan2(y, x)

    # determine when points lie inside and on surface of magnet -------------

    # phip1 in [-2pi,0], phio2 in [0,2pi]
    phio1 = phi
    phio2 = phi - np.sign(phi) * 2 * np.pi

    # r, phi ,z lies in-between, avoid numerical fluctuations
    # (e.g. due to rotations) by including tolerance
    mask_r_in = (r1 - tolerance < r) & (r < r2 + tolerance)
    mask_phi_in = (np.sign(phio1 - phi1) != np.sign(phio1 - phi2)) | (
        np.sign(phio2 - phi1) != np.sign(phio2 - phi2)
    )
    mask_z_in = (z1 - tolerance < z) & (z < z2 + tolerance)

    # inside
    return mask_r_in & mask_phi_in & mask_z_in


def mask_inside_TriangularMesh(obj, positions, tolerance=1e-14):  # noqa: ARG001  # pylint: disable=unused-argument
    """Return mask of provided positions inside a TriangularMesh (local coordinates).

    Note: uses magpylib._src.fields.field_BH_triangularmesh._mask_inside_trimesh,
    a private magpylib internal API (requires magpylib>=5.0).
    """
    from magpylib._src.fields.field_BH_triangularmesh import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        _mask_inside_trimesh,
    )

    faces_as_vertices = obj.vertices[obj.faces]  # shape (m, 3, 3)
    return _mask_inside_trimesh(np.asarray(positions, dtype=float), faces_as_vertices)


def mask_inside(obj, positions, tolerance=1e-14):
    """Return mask of provided positions inside a Magpylib object"""
    mask_inside_funcs = {
        "Cuboid": mask_inside_Cuboid,
        "Cylinder": mask_inside_Cylinder,
        "Sphere": mask_inside_Sphere,
        "CylinderSegment": mask_inside_CylinderSegment,
        "TriangularMesh": mask_inside_TriangularMesh,
    }
    func = mask_inside_funcs.get(obj.__class__.__name__)
    if func is None:
        msg = "Unsupported object type for inside masking"
        raise TypeError(msg)
    return func(obj, positions, tolerance)


def trimesh_from_model3d(shape, polarization, **make_kwargs):
    """Build a closed ``magpy.magnet.TriangularMesh`` from a magpylib model3d
    surface trace identified by a short *shape* name.

    The corresponding ``magpylib.graphics.model3d.make_*`` function is called
    with *make_kwargs*, then shared-edge vertices in the Plotly Mesh3d output
    are deduplicated and a closed ``TriangularMesh`` is returned.

    Parameters
    ----------
    shape : str
        Short shape name (case-insensitive).  Supported values:

        * ``"cuboid"``
        * ``"cylinder_segment"``
        * ``"ellipsoid"``
        * ``"prism"``
        * ``"pyramid"``
        * ``"tetrahedron"``

    polarization : array-like, shape (3,)
        Polarization vector for the resulting ``TriangularMesh``.
    **make_kwargs
        Keyword arguments forwarded to the underlying ``make_*`` function
        (e.g. ``dimension``).

    Returns
    -------
    magpy.magnet.TriangularMesh
        Closed, consistently-oriented triangular surface mesh.
    """
    func_map = {
        "cuboid": _m3d.make_Cuboid,
        "cylinder_segment": _m3d.make_CylinderSegment,
        "ellipsoid": _m3d.make_Ellipsoid,
        "prism": _m3d.make_Prism,
        "pyramid": _m3d.make_Pyramid,
        "tetrahedron": _m3d.make_Tetrahedron,
    }
    key = shape.lower()
    if key not in func_map:
        msg = f"Unsupported shape {shape!r}. Choose from: {list(func_map)}"
        raise ValueError(msg)
    trace = func_map[key](**make_kwargs)
    kw = trace["kwargs"]
    verts_raw = np.column_stack([kw["x"], kw["y"], kw["z"]])
    faces_raw = np.column_stack([kw["i"], kw["j"], kw["k"]])
    # Deduplicate numerically identical vertices (shared edges produce duplicates)
    _, inv = np.unique(np.round(verts_raw, 10), axis=0, return_inverse=True)
    vertices = np.unique(np.round(verts_raw, 10), axis=0)
    faces = inv[faces_raw]
    return magpy.magnet.TriangularMesh(
        polarization=polarization,
        vertices=vertices,
        faces=faces,
    )
