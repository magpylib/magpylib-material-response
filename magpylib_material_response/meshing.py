from collections import Counter
from itertools import product

import magpylib as magpy
import numpy as np
from loguru import logger
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent
from scipy.spatial.transform import Rotation as R

from magpylib_material_response.meshing_utils import cells_from_dimension
from magpylib_material_response.meshing_utils import get_volume
from magpylib_material_response.meshing_utils import mask_inside


def _collection_from_obj_and_cells(obj, cells, **style_kwargs):
    xi = getattr(obj, "xi", None)
    if xi is not None:
        for cell in cells:
            cell.xi = xi
    coll = magpy.Collection(cells)
    coll.style.update(obj.style.as_dict(), _match_properties=False)
    coll.style.update(coll._process_style_kwargs(**style_kwargs))
    coll.position = obj.position
    coll.orientation = obj.orientation
    return coll


def mesh_Cuboid(cuboid, target_elems, verbose=False, **kwargs):
    """
    Split Magpylib cuboid up into small cuboid cells

    Parameters
    ----------
    cuboid: magpylib.magnet.Cuboid object
        Input object to be discretized
    target_elems: triple of int or int
        Target number of cells. If `target_elems` is a triple of integers, the number of
        divisions corresponds respectively to the x,y,z components of the unrotated
        cuboid in the global CS. If an integer, the number of divisions is apportioned
        proportionally to the cuboid dimensions. The resulting meshing cuboid cells are
        then the closest to cubes as possible.
    verbose: bool
        If True, prints out meshing information

    Returns
    -------
    discretization: magpylib.Collection
        Collection of Cuboid cells
    """
    if not isinstance(cuboid, magpy.magnet.Cuboid):
        raise TypeError(
            "Object to be meshed must be a Cuboid, "
            f"received instead {cuboid.__class__.__name__!r}"
        )
    dim0 = cuboid.dimension
    mag0 = cuboid.magnetization

    if np.isscalar(target_elems):
        nnn = cells_from_dimension(dim0, target_elems)
    else:
        nnn = target_elems
    elems = np.prod(nnn)
    if verbose:
        logger.opt(colors=True).info(
            f"Meshing Cuboid with <blue>{nnn[0]}x{nnn[1]}x{nnn[2]}={elems}</blue>"
            f" elements (target={target_elems})"
        )

    # secure input type
    nnn = np.array(nnn, dtype=int)

    # new dimension
    new_dim = dim0 / nnn

    # inside position grid
    xs, ys, zs = (
        np.linspace(d / 2 * (1 / n - 1), d / 2 * (1 - 1 / n), n)
        for d, n in zip(dim0, nnn)
    )
    grid = np.array([(x, y, z) for x in xs for y in ys for z in zs])

    # create cells as magpylib objects and return Collection
    cells = []
    for pp in grid:
        cell = magpy.magnet.Cuboid(magnetization=mag0, dimension=new_dim, position=pp)
        cells.append(cell)

    return _collection_from_obj_and_cells(cuboid, cells, **kwargs)


def mesh_Cylinder(cylinder, target_elems, verbose=False, **kwargs):
    """
    Split `Cylinder` or `CylinderSegment` up into small cylindrical or cylinder segment
    cells. In case of the cylinder, the middle cells are cylinders, all other being
    cylinder segments.

    Parameters
    ----------
    cylinder: `magpylib.magnet.Cylinder` or  `magpylib.magnet.CylinderSegment` object
        Input object to be discretized
    target_elems: int
        Target number of cells. If `target_elems` is a triple of integers, the number of
        divisions corresponds respectively to the divisions along the circumference,
        over the radius and over the height. If an integer, the number of divisions is
        apportioned proportionally to the cylinder (or cylinder segment) dimensions. The
        resulting meshing cylinder segment cells are then the closest to cubes as
        possible.
    verbose: bool
        If True, prints out meshing information

    Returns
    -------
    discretization: magpylib.Collection
        Collection of Cylinder and CylinderSegment cells
    """
    if isinstance(cylinder, magpy.magnet.CylinderSegment):
        r1, r2, h, phi1, phi2 = cylinder.dimension
    elif isinstance(cylinder, magpy.magnet.Cylinder):
        r1, r2, h, phi1, phi2 = (
            0,
            cylinder.dimension[0] / 2,
            cylinder.dimension[1],
            0,
            360,
        )
    else:
        raise TypeError("Input must be a Cylinder or CylinderSegment")

    al = (r2 + r1) * 3.14 * (phi2 - phi1) / 360  # arclen = D*pi*arcratio
    dim = al, r2 - r1, h
    mag0 = cylinder.magnetization
    # "unroll" the cylinder and distribute the target number of elemens along the
    # circumference, radius and height.
    if np.isscalar(target_elems):
        nphi, nr, nh = cells_from_dimension(dim, target_elems)
    else:
        nphi, nr, nh = target_elems
    elems = np.prod([nphi, nr, nh])
    if verbose:
        logger.opt(colors=True).info(
            f"Meshing CylinderSegement with <blue>{nphi}x{nr}x{nh}={elems}</blue>"
            f" elements (target={target_elems})"
        )
    r = np.linspace(r1, r2, nr + 1)
    dh = h / nh
    cells = []
    for r_ind in range(nr):
        # redistribute number divisions proportionally to the radius
        nphi_r = max(1, int(r[r_ind + 1] / ((r1 + r2) / 2) * nphi))
        phi = np.linspace(phi1, phi2, nphi_r + 1)
        for h_ind in range(nh):
            pos_h = dh * h_ind - h / 2 + dh / 2
            # use a cylinder for the innermost cells if there are at least 3 layers and
            # if it is closed, use cylinder segments otherwise
            if nr >= 3 and r[r_ind] == 0 and phi2 - phi1 == 360:
                dimension = r[r_ind + 1] * 2, dh
                cell = magpy.magnet.Cylinder(
                    magnetization=mag0,
                    dimension=dimension,
                    position=(0, 0, pos_h),
                )
                cells.append(cell)
            else:
                for phi_ind in range(nphi_r):
                    dimension = (
                        r[r_ind],
                        r[r_ind + 1],
                        dh,
                        phi[phi_ind],
                        phi[phi_ind + 1],
                    )
                    cell = magpy.magnet.CylinderSegment(
                        magnetization=mag0,
                        dimension=dimension,
                        position=(0, 0, pos_h),
                    )
                    cells.append(cell)
    return _collection_from_obj_and_cells(cylinder, cells, **kwargs)


def mesh_thin_CylinderSegment_with_cuboids(
    cyl_seg, target_elems, ratio_limit=10, match_volume=True, **kwargs
):
    """
    Split-up a Magpylib thin-walled cylinder segment into cuboid cells. Over the
    thickness, only one layer of cells is used.

    Parameters
    ----------
    cyl_seg: `magpylib.magnet.CylinderSegment` object
        CylinderSegment object to be discretized
    target_elems: int or tuple of int,
        If `target_elems` is a  tuple of integers, the cylinder segment is respectively
        divided over circumference and height, if an integer, divisions are infered to
        build cuboids with close to squared faces.
    ratio_limit: positive number,
        Sets the r2/(r2-r1) limit to be considered as thin-walled, r1 being the inner
        radius.
    match_volume: bool,
        If True, it ensures the meshed volume equals the original CylinderSegment by
        allowing mesh elements overlapping. This improves the field calculation
        approximation accuracy.

    Returns
    -------
    discretization: magpylib.Collection
        Collection of Cuboid cells
    """

    r1, r2, h, phi1, phi2 = cyl_seg.dimension
    mag0 = cyl_seg.magnetization
    if ratio_limit > r2 / (r2 - r1):
        raise ValueError(
            "This meshing function is intended for thin-walled CylinderSegment objects"
            f" of radii-ratio r2/(r2-r1)>{ratio_limit}, r1 being the inner radius."
            f"\nInstead r2/(r2-r1)={r2 / (r2 - r1)}"
        )
    # distribute elements -> targeting thin close-to-square surface cells
    circumf = 2 * np.pi * r1
    if np.isscalar(target_elems):
        nphi = int(np.round((target_elems * circumf / h) ** 0.5))
        nh = int(np.round(target_elems / nphi))
    else:
        nphi, nh = target_elems
    dh = h / nh
    dphi = 2 * np.pi / nphi * (phi2 - phi1) / 360
    dim = np.array([r2 - r1, 2 * r1 * np.sin(dphi / 2), dh])  # cuboids edge sizes
    if match_volume:
        vol_ratio = get_volume(cyl_seg) / (np.prod(dim) * nh * nphi)
        dim *= vol_ratio ** (1 / 3)
    x0 = r1 * np.cos(dphi / 2) + dim[0] / 2
    phi_vec = np.linspace(
        np.deg2rad(phi1) + dphi / 2, np.deg2rad(phi2) - dphi / 2, nphi
    )
    poss = np.array([x0 * np.cos(phi_vec), x0 * np.sin(phi_vec), np.zeros(nphi)]).T
    rots = R.from_euler("z", phi_vec)
    cells = []
    for z in np.linspace(-h / 2 + dh / 2, h / 2 - dh / 2, nh):
        for pos, orient in zip(poss, rots):
            child = magpy.magnet.Cuboid(
                magnetization=orient.inv().apply(mag0),
                dimension=dim,
                position=pos + np.array([0, 0, z]),
                orientation=orient,
            )
            cells.append(child)
    return _collection_from_obj_and_cells(cyl_seg, cells, **kwargs)


def slice_Cuboid(cuboid, shift=0.5, axis="z", **kwargs):
    """
    Slice a cuboid magnet along a specified axis and return a collection of the
    resulting parts.

    Parameters
    ----------
    cuboid : magpy.magnet.Cuboid
        The cuboid to be sliced.
    shift : float, optional, default=0.5
        The relative position of the slice along the specified axis, ranging from
        0 to 1 (exclusive).
    axis : {'x', 'y', 'z'}, optional, default='z'
        The axis along which to slice the cuboid.
    **kwargs
        Additional keyword arguments to pass to the magpy.Collection constructor.

    Returns
    -------
    coll : magpy.Collection
        A collection of the resulting parts after slicing the cuboid.

    Raises
    ------
    ValueError
        If the shift value is not between 0 and 1 (exclusive).
    """
    if not isinstance(cuboid, magpy.magnet.Cuboid):
        raise TypeError(
            "Object to be sliced must be a Cuboid, "
            f"received instead {cuboid.__class__.__name__!r}"
        )
    if not 0 < shift < 1:
        raise ValueError("Shift must be between 0 and 1 (exclusive)")
    dim0 = cuboid.dimension
    mag0 = cuboid.magnetization
    ind = "xyz".index(axis)
    dim_k = cuboid.dimension[ind]
    dims_k = dim_k * (1 - shift), dim_k * (shift)
    shift_k = (dim_k - dims_k[0]) / 2, -(dim_k - dims_k[1]) / 2
    cells = []
    for d, s in zip(dims_k, shift_k):
        dimension = dim0.copy()
        dimension[ind] = d
        position = np.array([0, 0, 0], dtype=float)
        position[ind] = s
        cell = magpy.magnet.Cuboid(
            magnetization=mag0,
            dimension=dimension,
            position=position,
        )
        cells.append(cell)
    return _collection_from_obj_and_cells(cuboid, cells[::-1], **kwargs)


def voxelize(obj, target_elems, strict_inside=True, **kwargs):
    """
    Split-up a Magpylib magnet into a regular grid of identical cells. A grid of
    identical cube cells and containing the object is created. Only the cells with their
    barycenter inside the original object are kept.

    Parameters
    ----------
    obj: `magpylib.magnet` object
        Input object to be discretized
    target_elems: int
        Target number of cells
    strict inside: bool
        If True, also filters out the cells with vertices outside the object boundaries

    Returns
    -------
    discretization: magpylib.Collection
        Collection of Cylinder and CylinderSegment cells"""
    mag0 = obj.magnetization
    vol, containing_cube_edge = get_volume(obj, return_containing_cube_edge=True)
    vol_ratio = (containing_cube_edge**3) / vol

    grid_elems = [int((vol_ratio * target_elems) ** (1 / 3))] * 3
    grid_dim = [containing_cube_edge] * 3

    slices = [slice(-d / 2, d / 2, N * 1j) for d, N in zip(grid_dim, grid_elems)]
    grid = np.mgrid[slices].reshape(len(slices), -1).T
    grid = grid[mask_inside(obj, grid, tolerance=1e-14)]
    cube_cell_dim = np.array([containing_cube_edge / (grid_elems[0] - 1)] * 3)
    if strict_inside:
        elemgrid = np.array(
            list(product(*[[-cube_cell_dim[0] / 2, cube_cell_dim[0] / 2]] * 3))
        )
        cube_grid = np.array([elemgrid + pos for pos in grid])
        pos_inside_strict_mask = np.all(
            mask_inside(obj, cube_grid.reshape(-1, 3)).reshape(cube_grid.shape[:-1]),
            axis=1,
        )
        grid = grid[pos_inside_strict_mask]
        if grid.size == 0:
            raise ValueError("No cuboids left with strict-inside method")

    cells = []
    for pos in grid:
        cell = magpy.magnet.Cuboid(
            magnetization=mag0,
            dimension=cube_cell_dim,
            position=pos,
        )
        cells.append(cell)
    return _collection_from_obj_and_cells(obj, cells, **kwargs)


def mesh_all(
    obj,
    target_elems,
    min_elems=8,
    per_child_elems=False,
    inplace=False,
    **kwargs,
):
    """
    Mesh all the supported objects into a Collection of equivalent children, replacing
    them with their meshed version.

    Parameters
    ----------
    obj : magpy.Collection, magpy.magnet.Cuboid, magpy.magnet.Cylinder, BaseCurrent
        The object to be meshed. If a magpy.Collection, all its children will be
        meshed.
    target_elems : int
        Target number of elements for the meshing.
    min_elems : int, optional, default=8
        Minimum number of elements allowed in the mesh.
    per_child_elems : bool, optional, default=False
        If False, target_elems will be divided among children objects based on their
        volumes. If True, all children objects will have the same target_elems.
    inplace : bool, optional, default=False
        If True, meshing will be performed in-place, modifying the original object.
        If False, a new object will be created.

    Returns
    -------
    obj : magpy.Collection, original object
        The meshed object or a new object with meshed components.

    Raises
    ------
    TypeError
        If there are incompatible objects found.
    """
    supported = (magpy.magnet.Cuboid, magpy.magnet.Cylinder)
    allowed = supported + (BaseCurrent,)
    if not inplace:
        obj = obj.copy(**kwargs)
    children = [obj]
    if isinstance(obj, magpy.Collection):
        children = list(obj.sources_all)
    incompatible_objs = [c for c in children if not isinstance(c, allowed)]
    supported_objs = [c for c in children if isinstance(c, supported)]
    if not per_child_elems:
        volumes = np.array([get_volume(c) for c in supported_objs])
        volumes /= volumes.sum()
        target_elems_by_child = (volumes * target_elems).astype(int)
        target_elems_by_child[target_elems_by_child < min_elems] = min_elems
    else:
        target_elems_by_child = [max(min_elems, target_elems)] * len(supported_objs)
    if incompatible_objs:
        raise TypeError(
            "Incompatible objects found: "
            f"{Counter(s.__class__.__name__ for s in incompatible_objs)}"
            f"\nSupported: {[s.__name__ for s in supported]}."
        )
    for child, target_elems in zip(supported_objs, target_elems_by_child):
        parent = child.parent
        kw = kwargs if parent is None else {}
        if isinstance(child, magpy.magnet.Cuboid):
            child_meshed = mesh_Cuboid(child, target_elems, **kw)
        elif isinstance(child, magpy.magnet.Cylinder):
            child_meshed = mesh_Cylinder(child, target_elems, **kw)
        child.parent = None
        if parent is not None:
            parent.add(child_meshed)
        else:
            obj = child_meshed
    return obj
