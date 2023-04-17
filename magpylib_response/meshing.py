from itertools import product

import magpylib as magpy
import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R

from magpylib_response.meshing_utils import cells_from_dimension
from magpylib_response.meshing_utils import get_volume
from magpylib_response.meshing_utils import mask_inside


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
    # TODO: make function compatible with paths
    # load cuboid properties
    dim = cuboid.dimension
    mag = cuboid.magnetization
    xi = getattr(cuboid, "xi", None)

    if np.isscalar(target_elems):
        nnn = cells_from_dimension(dim, target_elems)
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
    new_dim = dim / nnn

    # inside position grid
    xs, ys, zs = (
        np.linspace(d / 2 * (1 / n - 1), d / 2 * (1 - 1 / n), n)
        for d, n in zip(dim, nnn)
    )
    grid = np.array([(x, y, z) for x in xs for y in ys for z in zs])

    # create cells as magpylib objects and return Collection
    cells = []
    for pp in grid:
        cell = magpy.magnet.Cuboid(
            magnetization=mag,
            dimension=new_dim,
            position=pp,
        )
        if xi is not None:
            cell.xi = xi
        cells.append(cell)

    coll = magpy.Collection(cells, **kwargs)

    coll.position = cuboid.position
    coll.orientation = cuboid.orientation
    return coll


def mesh_cuboids_with_cuboids(obj, target_elems, inplace=False):
    """
    Mesh a Cuboid or Collection of Cuboids with a target number of elements.

    Parameters
    ----------
    obj : magpy.magnet.Cuboid or magpy.Collection
        The Cuboid or Collection to mesh.
    target_elems : int
        The target number of elements for the mesh.
    inplace : bool, optional
        If True, perform the mesh operation in-place, modifying the original
        object. If False, create a new object with the mesh operation applied.
        Default is False.

    Returns
    -------
    obj : magpy.magnet.Cuboid or magpy.Collection
        The meshed Cuboid or Collection.

    Notes
    -----
    This function recursively processes any child objects within the input
    Collection, performing the mesh operation on each.
    """
    if isinstance(obj, magpy.magnet.Cuboid):
        label = obj.style.label
        cuboid_meshed = mesh_Cuboid(obj, target_elems, style_label=label)
        if inplace:
            parent = obj.parent
            obj.parent = None
            parent.add(cuboid_meshed)
        obj = cuboid_meshed
    elif isinstance(obj, magpy.Collection):
        if not inplace:
            obj = obj.copy()
        children = list(
            obj.children
        )  # otherwise children list is changed while looping!!
        for child in children:
            mesh_cuboids_with_cuboids(child, target_elems, inplace=True)
    return obj


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

    pos = cylinder._position
    rot = cylinder._orientation
    mag = cylinder.magnetization
    xi = getattr(cylinder, "xi", None)
    al = (r2 + r1) * 3.14 * (phi2 - phi1) / 360  # arclen = D*pi*arcratio
    dim = al, r2 - r1, h

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
    cyl_segs = []
    for r_ind in range(nr):
        # redistribute number divisions proportionally to the radius
        nphi_r = max(1, int(r[r_ind + 1] / ((r1 + r2) / 2) * nphi))
        phi = np.linspace(phi1, phi2, nphi_r + 1)
        for h_ind in range(nh):
            pos_h = dh * h_ind - h / 2 + dh / 2
            # use a cylinder for the innermost cells if there are at least 3 layers,
            # cylinder segment otherwise
            if nr >= 3 and r[r_ind] == 0 and phi2 - phi1 == 360:
                dimension = r[r_ind + 1] * 2, dh
                cell = magpy.magnet.Cylinder(
                    magnetization=mag, dimension=dimension, position=(0, 0, pos_h)
                )
                cyl_segs.append(cell)
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
                        magnetization=mag, dimension=dimension, position=(0, 0, pos_h)
                    )
                    cyl_segs.append(cell)
    for cell in cyl_segs:
        if xi is not None:
            cell.xi = xi
    return (
        magpy.Collection(cyl_segs, **kwargs)
        .rotate(rot, anchor=0, start=0)
        .move(pos, start=0)
    )


def mesh_thin_CylinderSegment_with_cuboids(
    cyl_seg, target_elems, thin_ratio_limit=10, **kwargs
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
    thin_ratio_limit: positive number,
        Sets the r2/(r2-r1) limit to be considered as thin-walled, r1 being the inner
        radius.

    Returns
    -------
    discretization: magpylib.Collection
        Collection of Cuboid cells
    """

    r1, r2, h, phi1, phi2 = cyl_seg.dimension
    xi = getattr(cyl_seg, "xi", None)
    if thin_ratio_limit > r2 / (r2 - r1):
        raise ValueError(
            "This meshing function is intended for thin-walled CylinderSegment objects"
            f" of radii-ratio r2/(r2-r1)>{thin_ratio_limit}, r1 being the inner radius."
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
    a, b, c = r2 - r1, 2 * r1 * np.sin(dphi / 2), dh  # cuboids edge sizes
    x0 = r1 * np.cos(dphi / 2) + a / 2
    phi_vec = np.linspace(
        np.deg2rad(phi1) + dphi / 2, np.deg2rad(phi2) - dphi / 2, nphi
    )
    poss = np.array([x0 * np.cos(phi_vec), x0 * np.sin(phi_vec), np.zeros(nphi)]).T
    rots = R.from_euler("z", phi_vec)
    cuboids = []
    for z in np.linspace(-h / 2 + dh / 2, h / 2 - dh / 2, nh):
        for pos, orient in zip(poss, rots):
            child = magpy.magnet.Cuboid(
                cyl_seg.magnetization, (a, b, c), pos + np.array([0, 0, z]), orient
            )
            if xi is not None:
                child.xi = xi
            cuboids.append(child)
    coll = magpy.Collection(cuboids, **kwargs)
    coll.orientation = cyl_seg.orientation
    coll.position = cyl_seg.position
    return coll


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
    if not 0 < shift < 1:
        raise ValueError("Shift must be between 0 and 1 (exclusive)")
    dim0 = cuboid.dimension
    mag0 = cuboid.magnetization
    xi = getattr(cuboid, "xi", None)
    ind = "xyz".index(axis)
    dim_k = cuboid.dimension[ind]
    dims_k = dim_k * (1 - shift), dim_k * (shift)
    shift_k = (dim_k - dims_k[0]) / 2, -(dim_k - dims_k[1]) / 2
    children = []
    for d, s in zip(dims_k, shift_k):
        dimension = dim0.copy()
        dimension[ind] = d
        position = np.array([0, 0, 0], dtype=float)
        position[ind] = s
        child = magpy.magnet.Cuboid(
            magnetization=mag0, dimension=dimension, position=position
        )
        if xi is not None:
            child.xi = xi
        children.append(child)
    coll = magpy.Collection(children, **kwargs)
    coll.orientation = cuboid.orientation
    coll.position = cuboid.position
    return coll


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
    xi = getattr(obj, "xi", None)
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
    cube_poss = obj.orientation.apply(grid) + obj.position

    obj_list = []
    for pos in cube_poss:
        child = magpy.magnet.Cuboid(
            magnetization=obj.magnetization,
            dimension=cube_cell_dim,
            position=pos,
            orientation=obj.orientation,
        )
        if xi is not None:
            child.xi = xi
        obj_list.append(child)
    return magpy.Collection(obj_list, **kwargs)
