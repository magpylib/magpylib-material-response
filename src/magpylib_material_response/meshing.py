from __future__ import annotations

import copy
import warnings
from collections import Counter
from itertools import product

import magpylib as magpy
import numpy as np
from loguru import logger
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent
from scipy.spatial.transform import Rotation as R

from magpylib_material_response.meshing_utils import (
    get_volume,
    mask_inside,
    near_cubic_divisions,
)


def _collection_from_obj_and_cells(obj, cells, **style_kwargs):
    susceptibility = getattr(obj, "susceptibility", None)
    if susceptibility is not None:
        for cell in cells:
            cell.susceptibility = copy.deepcopy(susceptibility)
    coll = magpy.Collection(cells)
    coll.style.update(obj.style.as_dict(), _match_properties=False)
    coll.style.update(coll._process_style_kwargs(**style_kwargs))
    coll.position = obj.position
    coll.orientation = obj.orientation
    # The applied field is uniform over the object, so it belongs to the meshed
    # collection: the cells find it there, and a later change still reaches them.
    h_ext = getattr(obj, "H_ext", None)
    if h_ext is not None:
        coll.H_ext = copy.deepcopy(h_ext)
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
        cuboid in the global CS. If an integer, the cell edge is taken from the volume
        and the count and each side is cut into a whole number of cells, so the cells
        are as close to cubes as possible and the count can differ from the target,
        as in magpylib's force mesher.
    verbose: bool
        If True, prints out meshing information

    Returns
    -------
    discretization: magpylib.Collection
        Collection of Cuboid cells
    """
    if not isinstance(cuboid, magpy.magnet.Cuboid):
        msg = (
            "Object to be meshed must be a Cuboid, "
            f"received instead {cuboid.__class__.__name__!r}"
        )
        raise TypeError(msg)
    dim0 = cuboid.dimension
    pol0 = cuboid.polarization

    if np.isscalar(target_elems):
        nnn = near_cubic_divisions(dim0, target_elems)
    else:
        nnn = target_elems
    elems = np.prod(nnn)
    if verbose:
        logger.info(
            "Meshing Cuboid",
            dimensions=f"{nnn[0]}x{nnn[1]}x{nnn[2]}",
            elements=elems,
            target=target_elems,
        )

    # secure input type
    nnn = np.array(nnn, dtype=int)

    # new dimension
    new_dim = dim0 / nnn

    # inside position grid
    xs, ys, zs = (
        np.linspace(d / 2 * (1 / n - 1), d / 2 * (1 - 1 / n), n)
        for d, n in zip(dim0, nnn, strict=False)
    )
    grid = np.array([(x, y, z) for x in xs for y in ys for z in zs])

    # create cells as magpylib objects and return Collection
    cells = []
    for pp in grid:
        cell = magpy.magnet.Cuboid(polarization=pol0, dimension=new_dim, position=pp)
        cells.append(cell)

    return _collection_from_obj_and_cells(cuboid, cells, **kwargs)


def _sectors_with_centroid_inside(r_in, r_out, span, n):
    """Smallest number of sectors, at least `n`, that splits the ring
    `r_in..r_out` over `span` degrees so that each sector's centroid lies within
    a quarter of the ring's thickness of its mid-radius.

    Demagnetization matches each cell's field at its centroid. A wide sector of a
    thin ring has its centroid outside the material (a 120 deg sector of a
    4-5 mm ring has it at 3.7 mm), which makes the result wrong.
    """
    if r_in == 0:
        return n  # a disc sector's centroid is always inside it
    r_mid = (r_in + r_out) / 2
    tolerance = (r_out - r_in) / 4
    # centroid radius of an annular sector of half-angle a: r_bar * sin(a) / a
    r_bar = 2 / 3 * (r_out**3 - r_in**3) / (r_out**2 - r_in**2)
    while True:
        half_angle = np.deg2rad(span / n) / 2
        if r_mid - r_bar * np.sin(half_angle) / half_angle <= tolerance:
            return n
        n += 1


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
        over the radius and over the height. If an integer, the unrolled cylinder
        (arc at the mean radius, radial width, height) is cut into near-cubic
        divisions, as in magpylib's force mesher. Either way, a ring gets more sectors
        where needed so that every
        cell's centroid lies inside it, so thin rings can get more cells than asked
        for.
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
        msg = "Input must be a Cylinder or CylinderSegment"
        raise TypeError(msg)

    al = (r2 + r1) * np.pi * (phi2 - phi1) / 360  # arclen = D*pi*arcratio
    dim = al, r2 - r1, h
    pol0 = cylinder.polarization
    # "unroll" the cylinder and distribute the target number of elements along the
    # circumference, radius and height.
    if np.isscalar(target_elems):
        nphi, nr, nh = near_cubic_divisions(dim, target_elems)
    else:
        nphi, nr, nh = target_elems
    elems = np.prod([nphi, nr, nh])
    if verbose:
        logger.info(
            "Meshing CylinderSegment",
            dimensions=f"{nphi}x{nr}x{nh}",
            elements=elems,
            target=target_elems,
        )
    r = np.linspace(r1, r2, nr + 1)
    dh = h / nh
    cells = []
    for r_ind in range(nr):
        # angular divisions in proportion to the ring's middle radius, so that all
        # cells have about the same arc length and the rings add up to nphi * nr
        nphi_r = max(1, round((r[r_ind] + r[r_ind + 1]) / (r1 + r2) * nphi))
        nphi_r = _sectors_with_centroid_inside(
            r[r_ind], r[r_ind + 1], phi2 - phi1, nphi_r
        )
        phi = np.linspace(phi1, phi2, nphi_r + 1)
        for h_ind in range(nh):
            pos_h = dh * h_ind - h / 2 + dh / 2
            # use a cylinder for the innermost cells if there are at least 3 layers and
            # if it is closed, use cylinder segments otherwise
            if nr >= 3 and r[r_ind] == 0 and phi2 - phi1 == 360:
                dimension = r[r_ind + 1] * 2, dh
                cell = magpy.magnet.Cylinder(
                    polarization=pol0,
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
                        polarization=pol0,
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
        divided over circumference and height, if an integer, divisions are inferred to
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

    Notes
    -----
    Each cell is turned about the z axis to follow the ring, so an anisotropic
    susceptibility must be the same along x and y: the cells cannot keep
    different values along the segment's own x and y axes. Use `mesh_Cylinder`
    for those.
    """

    r1, r2, h, phi1, phi2 = cyl_seg.dimension
    pol0 = cyl_seg.polarization
    chi = getattr(cyl_seg, "susceptibility", None)
    chi = np.asarray(0 if chi is None else chi, dtype=float)
    if chi.shape == (3,) and not np.isclose(chi[0], chi[1]):
        msg = (
            "mesh_thin_CylinderSegment_with_cuboids turns each cell about the z axis, "
            "so an anisotropic susceptibility must be the same along x and y; "
            f"instead received {tuple(chi)}. Use mesh_Cylinder, whose cells keep the "
            "segment's axes."
        )
        raise ValueError(msg)
    if ratio_limit > r2 / (r2 - r1):
        msg = (
            "This meshing function is intended for thin-walled CylinderSegment objects"
            f" of radii-ratio r2/(r2-r1)>{ratio_limit}, r1 being the inner radius."
            f"\nInstead r2/(r2-r1)={r2 / (r2 - r1)}"
        )
        raise ValueError(msg)
    # distribute elements -> targeting thin close-to-square surface cells
    circumf = 2 * np.pi * r1
    if np.isscalar(target_elems):
        # clamp to >=1: extreme aspect ratios can round either count to zero
        nphi = max(1, int(np.round((target_elems * circumf / h) ** 0.5)))
        nh = max(1, int(np.round(target_elems / nphi)))
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
    rots = R.from_rotvec(np.column_stack([np.zeros((nphi, 2)), phi_vec]))
    cells = []
    for z in np.linspace(-h / 2 + dh / 2, h / 2 - dh / 2, nh):
        for pos, orient in zip(poss, rots, strict=False):
            child = magpy.magnet.Cuboid(
                polarization=orient.inv().apply(pol0),
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
        msg = (
            "Object to be sliced must be a Cuboid, "
            f"received instead {cuboid.__class__.__name__!r}"
        )
        raise TypeError(msg)
    if not 0 < shift < 1:
        msg = "Shift must be between 0 and 1 (exclusive)"
        raise ValueError(msg)
    dim0 = cuboid.dimension
    pol0 = cuboid.polarization
    ind = "xyz".index(axis)
    dim_k = cuboid.dimension[ind]
    dims_k = dim_k * (1 - shift), dim_k * (shift)
    shift_k = (dim_k - dims_k[0]) / 2, -(dim_k - dims_k[1]) / 2
    cells = []
    for d, s in zip(dims_k, shift_k, strict=False):
        dimension = dim0.copy()
        dimension[ind] = d
        position = np.array([0, 0, 0], dtype=float)
        position[ind] = s
        cell = magpy.magnet.Cuboid(
            polarization=pol0,
            dimension=dimension,
            position=position,
        )
        cells.append(cell)
    return _collection_from_obj_and_cells(cuboid, cells[::-1], **kwargs)


#: Spacing updates `voxelize` tries for touching cells that hold the volume.
VOXELIZE_MAX_TUNING = 10
#: Cell counts `voxelize` takes around `target_elems`, as a factor either way.
VOXELIZE_COUNT_RANGE = 1.5
#: Spacings `voxelize` scans when the updates find none, 1 % apart from 0.8 to
#: 1.25 times the target's.
VOXELIZE_SCAN = np.geomspace(0.8, 1.25, 46)
#: Share of the volume `voxelize` cells may miss without a warning.
VOXELIZE_WARN_MISS = 0.01
#: Share of the volume `voxelize` cells may miss to keep a count near the
#: target; beyond it, an exact volume at any count met is taken instead.
VOXELIZE_NEAR_MISS = 0.05
#: Share of the volume beyond which `voxelize` refuses: the body is thinner
#: than the cells.
VOXELIZE_MAX_VOLUME_MISS = 0.5


def _lattice_inside(obj, spacing, extent):
    """The points inside `obj` of a cubic lattice of `spacing` with a point at
    the origin, over a cube of edge `extent` centred there."""
    m = int(np.ceil(extent / 2 / spacing))
    axis = np.arange(-m, m + 1) * spacing
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    return grid[mask_inside(obj, grid, tolerance=1e-14)]


def _voxel_edge(obj, vol, extent, target):
    """The edge of touching cubic cells, kept where their centre lies inside
    `obj`, that hold its volume `vol` at a count near `target`, and the share
    of the volume they miss; ``None`` when no spacing keeps any cell.

    N touching cells of edge h hold the volume when h = (V / N)^(1/3). From the
    target's spacing, moving h there reaches that point within a few lattices
    on compact bodies. Rings and tubes gain or lose cells by whole rings, and
    the moves can jump past it. A scan of spacings then takes, in this order:
    the exact spacing of a count near the target, nearest it; the spacing near
    the target whose cells come closest to the volume, within
    ``VOXELIZE_NEAR_MISS``; the exact spacing of any count met, as a flat body
    may hold its volume only with whole layers; the closest volume met.
    """
    start = (vol / target) ** (1 / 3)

    def count(h):
        return len(_lattice_inside(obj, h, extent))

    def near(n):
        return target / VOXELIZE_COUNT_RANGE <= n <= target * VOXELIZE_COUNT_RANGE

    def nearest(counts):
        n = min(counts, key=lambda n: abs(n - target))
        return (vol / n) ** (1 / 3), 0.0

    exact, edge = set(), start
    for _ in range(VOXELIZE_MAX_TUNING):
        n = count(edge)
        if n == 0:
            break
        if abs(n * edge**3 / vol - 1) < 1e-12:
            if near(n):
                return edge, 0.0
            exact.add(n)
            break
        edge = (vol / n) ** (1 / 3)
        if not start / 2 <= edge <= 2 * start:
            break
    closest = {}  # count -> (volume miss, spacing)
    for h in start * VOXELIZE_SCAN:
        n = count(h)
        miss = abs(n * h**3 / vol - 1)
        if n and miss < closest.get(n, (np.inf,))[0]:
            closest[n] = (miss, h)
    # within the updates' bounds: on a body thinner than the cells, the exact
    # spacing of a count met is tiny, and its lattice would hold billions
    exact |= {
        n
        for n in closest
        if start / 2 <= (vol / n) ** (1 / 3) <= 2 * start
        and count((vol / n) ** (1 / 3)) == n
    }
    if any(near(n) for n in exact):
        return nearest([n for n in exact if near(n)])
    near_miss = min((v for n, v in closest.items() if near(n)), default=None)
    if near_miss is not None and near_miss[0] <= VOXELIZE_NEAR_MISS:
        return near_miss[1], near_miss[0]
    if exact:
        return nearest(exact)
    if not closest:
        return None
    miss, edge = min(closest.values())
    return edge, miss


def voxelize(obj, target_elems, strict_inside=False, **kwargs):
    """
    Split-up a Magpylib magnet into a regular grid of identical, touching cube
    cells. The cells whose centre lies inside the object are kept, and the grid
    spacing is tuned so that together they hold the object's volume, and a
    uniformly polarized magnet its moment. Untuned, they hold 96 to 109 % of a
    sphere's volume, depending on the cell count. The count lands within a
    factor of 1.5 of `target_elems` where a spacing there holds the volume.
    Where none does, as on some rings at some targets, the closest within 5 %
    is taken, with a warning above 1 %; a flat body may hold its volume only
    with whole layers of cells, and then gets the count that does.

    From a susceptibility of about 10 on, the staircase outline sets the moment
    instead: a sphere at χ = 1000 comes out 5 % high at 3,000 cells, and the
    error shrinks slowly. Meshes that follow the surface, such as `mesh_Cylinder`
    or `mesh_TriangularMesh`, converge faster.

    Parameters
    ----------
    obj: `magpylib.magnet` object
        Input object to be discretized
    target_elems: int
        Target number of cells
    strict_inside: bool, default=False
        If True, keep only the cells lying wholly inside the object, untuned.
        They hold less than its volume: a sphere at 1,000 cells loses a quarter.

    Returns
    -------
    discretization: magpylib.Collection
        Collection of Cuboid cells

    Raises
    ------
    ValueError
        If no cell centre lies inside the object, or the cells would miss more
        than half its volume, as on a body thinner than the cells.
    """
    pol0 = obj.polarization
    vol, containing_cube_edge = get_volume(obj, return_containing_cube_edge=True)
    no_centres = (
        "voxelize produced no cell centres inside the object; increase target_elems"
    )
    if strict_inside:
        vol_ratio = (containing_cube_edge**3) / vol
        # clamp to >=2: the cell size below divides by (grid_elems - 1)
        grid_elems = [max(2, int((vol_ratio * target_elems) ** (1 / 3)))] * 3
        grid_dim = [containing_cube_edge] * 3
        slices = [
            slice(-d / 2, d / 2, N * 1j)
            for d, N in zip(grid_dim, grid_elems, strict=False)
        ]
        grid = np.mgrid[slices].reshape(len(slices), -1).T
        grid = grid[mask_inside(obj, grid, tolerance=1e-14)]
        if grid.size == 0:
            raise ValueError(no_centres)
        cube_cell_dim = np.array([containing_cube_edge / (grid_elems[0] - 1)] * 3)
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
            msg = "No cuboids left with strict-inside method"
            raise ValueError(msg)
    else:
        tuned = _voxel_edge(obj, vol, containing_cube_edge, target_elems)
        if tuned is None:
            raise ValueError(no_centres)
        edge, miss = tuned
        grid = _lattice_inside(obj, edge, containing_cube_edge)
        held = len(grid) * edge**3 / vol
        if miss > VOXELIZE_MAX_VOLUME_MISS:
            msg = (
                f"voxelize finds no spacing near target_elems={target_elems} at "
                f"which touching cells hold the object's volume (the closest holds "
                f"{held:.0%}): the object is thinner than the cells; increase "
                "target_elems"
            )
            raise ValueError(msg)
        if miss > VOXELIZE_WARN_MISS:
            warnings.warn(
                f"voxelize finds no spacing near target_elems={target_elems} at "
                f"which touching cells hold the object's volume exactly; the "
                f"closest holds {held:.1%}. Another target_elems may hit it.",
                stacklevel=2,
            )
        cube_cell_dim = np.full(3, edge)

    cells = []
    for pos in grid:
        cell = magpy.magnet.Cuboid(
            polarization=pol0,
            dimension=cube_cell_dim,
            position=pos,
        )
        cells.append(cell)
    return _collection_from_obj_and_cells(obj, cells, **kwargs)


#: TetGen makes about twice the target on a simple surface: the target caps the
#: cell volume, and the cells come out smaller. Past this ratio the surface, not
#: the target, sets the count.
TETGEN_MAX_OVERSHOOT = 4


def mesh_TriangularMesh(
    obj,
    target_elems,
    minratio=1.5,
    mindihedral=10.0,
    quality=True,
    verbose=False,
    **kwargs,
):
    """
    Split a TriangularMesh magnet into conforming Tetrahedron cells using TetGen.

    TetGen generates a surface-conforming tetrahedral mesh: all tetrahedra are strictly
    inside the closed surface and no vertex protrudes outside. Cell sizes are controlled
    via ``target_elems`` (sets ``maxvolume`` from the object volume). Quality is
    controlled by ``minratio`` and ``mindihedral``.

    Parameters
    ----------
    obj : magpy.magnet.TriangularMesh
        Input object to be discretized.
    target_elems : int
        Sets the largest cell volume, ``obj.volume / target_elems``. TetGen makes
        cells smaller than that where its quality limits ask for it, so a simple
        surface gives about twice the target. A detailed surface sets a floor:
        TetGen keeps every input triangle, so a 64-sided cylinder gives about
        6,900 cells for any target up to 1,000. A warning says when the count
        exceeds ``TETGEN_MAX_OVERSHOOT`` times the target.
    minratio : float, default 1.5
        TetGen quality constraint — maximum radius-edge ratio. Lower values give
        better-shaped tetrahedra but more cells. 1.0 is near-equilateral; 2.0 is
        permissive.
    mindihedral : float, default 10.0
        TetGen quality constraint — minimum dihedral angle in degrees. Prevents
        flat/sliver tetrahedra.
    quality : bool, default True
        If True, TetGen inserts Steiner points to meet the volume constraint set by
        ``target_elems`` and improve element quality. If False, only the input surface
        vertices are tetrahedralized and ``target_elems`` has no effect.
    verbose : bool
        If True, prints out meshing information.

    Returns
    -------
    discretization : magpy.Collection
        Collection of Tetrahedron cells.
    """
    try:
        import tetgen  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        msg = (
            "mesh_TriangularMesh requires the 'tetgen' package, which is an optional "
            "dependency. Install it with: pip install magpylib-material-response[tetgen]"
        )
        raise ImportError(msg) from exc

    if not isinstance(obj, magpy.magnet.TriangularMesh):
        msg = (
            "Object to be meshed must be a TriangularMesh, "
            f"received instead {obj.__class__.__name__!r}"
        )
        raise TypeError(msg)

    pol0 = obj.polarization
    maxvol = obj.volume / target_elems

    tet = tetgen.TetGen(
        obj.vertices.astype(float),
        obj.faces.astype(np.int32),
    )
    tet.tetrahedralize(
        order=1,
        quality=quality,
        minratio=minratio,
        mindihedral=mindihedral,
        fixedvolume=True,
        maxvolume=maxvol,
        quiet=not verbose,
    )

    nodes = tet.node  # (n_nodes, 3)
    elems = tet.elem  # (n_tets, 4) — node indices, 0-based
    tet_verts = nodes[elems]  # (n_tets, 4, 3)

    if tet_verts.shape[0] == 0:
        msg = "TetGen produced no tetrahedra — check surface normals are outward-facing"
        raise ValueError(msg)
    if len(tet_verts) > TETGEN_MAX_OVERSHOOT * target_elems:
        warnings.warn(
            f"TetGen made {len(tet_verts)} tetrahedra for target_elems={target_elems}: "
            "it keeps every triangle of the surface, and this surface needs more cells "
            "than the target. A surface with fewer facets, or cuboid cells, gives fewer.",
            stacklevel=2,
        )

    if verbose:
        logger.info(
            "Meshing TriangularMesh (TetGen)",
            elements=len(tet_verts),
            target=target_elems,
        )

    cells = [
        magpy.magnet.Tetrahedron(polarization=pol0, vertices=tv) for tv in tet_verts
    ]
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
    obj : Collection, magnet.Cuboid, magnet.Cylinder, magnet.CylinderSegment, magnet.TriangularMesh
        The object to be meshed. If a Collection, all its children will be
        meshed. BaseCurrent objects are allowed but not meshed.
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
    supported = (
        magpy.magnet.Cuboid,
        magpy.magnet.Cylinder,
        magpy.magnet.CylinderSegment,
        magpy.magnet.TriangularMesh,
    )
    allowed = (*supported, BaseCurrent)
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
        counts_incompatible = Counter(s.__class__.__name__ for s in incompatible_objs)
        counts_str = ", ".join(
            f"{count} {name}" for name, count in counts_incompatible.items()
        )
        msg = (
            "Incompatible objects found: "
            f"{counts_str}"
            f"\nSupported: {[s.__name__ for s in supported]}."
        )
        raise TypeError(msg)
    for child, targ_elems in zip(supported_objs, target_elems_by_child, strict=False):
        parent = child.parent
        kw = kwargs if parent is None else {}
        child_meshed = None
        if isinstance(child, magpy.magnet.Cuboid):
            child_meshed = mesh_Cuboid(child, targ_elems, **kw)
        elif isinstance(child, (magpy.magnet.Cylinder, magpy.magnet.CylinderSegment)):
            child_meshed = mesh_Cylinder(child, targ_elems, **kw)
        elif isinstance(child, magpy.magnet.TriangularMesh):
            child_meshed = mesh_TriangularMesh(child, targ_elems, **kw)
        if child_meshed is not None:
            child.parent = None
            if parent is not None:
                parent.add(child_meshed)
            if child is obj:
                obj = child_meshed
    return obj


def _responds(src):
    """Whether `src` responds to fields: a magnet whose susceptibility, its own
    or its nearest parent's, is set and not zero."""
    if isinstance(src, BaseCurrent):
        return False
    while src is not None:
        chi = getattr(src, "susceptibility", None)
        if chi is not None:
            return bool(np.any(np.asarray(chi, dtype=float) != 0))
        src = src.parent
    return False


def _thinnest(obj):
    """The smallest extent of a part that mesh_uniform meshes."""
    if isinstance(obj, magpy.magnet.CylinderSegment):
        r1, r2, h, *_ = obj.dimension
        return float(min(r2 - r1, h))
    if isinstance(obj, magpy.magnet.TriangularMesh):
        return float(np.ptp(obj.vertices, axis=0).min())
    return float(min(obj.dimension))  # Cuboid, Cylinder (diameter, height)


def mesh_uniform(obj, cells_across, inplace=False, max_cells=1_000_000, **kwargs):
    """
    Mesh every part that responds to fields into cells of one size,
    ``cells_across`` cells across the thinnest such part.

    A part responds when its susceptibility, set on itself or on a parent
    collection, is not zero. Each side of a cuboid is cut into
    ``round(side / size)`` cells, so cuboids whose sides are multiples of the size
    become grids of one spacing, which the solver multiplies by FFT, also between
    parts. Cylinders, cylinder segments and triangular meshes get their own mesher
    with as many cells as their volume holds at that size. Parts that do not
    respond and current sources stay as they are: meshing a uniformly polarized
    magnet does not change its field.

    Parameters
    ----------
    obj : magpylib.Collection or magnet
        The model to mesh.
    cells_across : int
        Cells across the thinnest part that responds.
    inplace : bool, optional, default=False
        If True, mesh `obj` itself; if False, a copy.
    max_cells : int, optional, default=1_000_000
        Refuse to make more cells than this. The cell size comes from the
        thinnest part, so a thin shim beside a large yoke can ask for millions.

    Returns
    -------
    magpylib.Collection or magnet
        The meshed model, with the collection hierarchy of `obj`.

    Raises
    ------
    ValueError
        If no part responds, `cells_across` is below 1, or the mesh would exceed
        `max_cells`.
    TypeError
        If a part that responds has a shape this function cannot mesh.
    """
    if cells_across < 1:
        msg = f"cells_across must be at least 1, got {cells_across}."
        raise ValueError(msg)
    supported = (
        magpy.magnet.Cuboid,
        magpy.magnet.Cylinder,
        magpy.magnet.CylinderSegment,
        magpy.magnet.TriangularMesh,
    )

    def parts(model):
        children = model.sources_all if isinstance(model, magpy.Collection) else [model]
        return [c for c in children if _responds(c)]

    def cells_of(part, size):
        """Divisions of a cuboid, or the cell count of any other part."""
        if isinstance(part, magpy.magnet.Cuboid):
            return tuple(max(1, round(d / size)) for d in part.dimension)
        return max(1, round(get_volume(part) / size**3))

    responding = parts(obj)
    if not responding:
        msg = "No part has a susceptibility other than zero: nothing to mesh."
        raise ValueError(msg)
    unsupported = Counter(
        c.__class__.__name__ for c in responding if not isinstance(c, supported)
    )
    if unsupported:
        msg = (
            f"mesh_uniform cannot mesh {dict(unsupported)}; "
            f"it meshes {[s.__name__ for s in supported]}."
        )
        raise TypeError(msg)
    thinnest = min(_thinnest(c) for c in responding)
    size = thinnest / cells_across
    total = sum(int(np.prod(cells_of(c, size))) for c in responding)
    if total > max_cells:
        msg = (
            f"cells_across={cells_across} would make about {total:,} cells, more "
            f"than max_cells={max_cells:,}: the cell size follows the thinnest part, "
            f"{thinnest:.3g} m across. Lower cells_across, raise max_cells, or mesh "
            "that part with its own mesher."
        )
        raise ValueError(msg)
    if not inplace:
        obj = obj.copy(**kwargs)
    for child in parts(obj):
        parent = child.parent
        kw = kwargs if parent is None else {}
        if isinstance(child, magpy.magnet.Cuboid):
            meshed = mesh_Cuboid(child, cells_of(child, size), **kw)
        else:
            mesher = (
                mesh_TriangularMesh
                if isinstance(child, magpy.magnet.TriangularMesh)
                else mesh_Cylinder
            )
            meshed = mesher(child, cells_of(child, size), **kw)
        child.parent = None
        if parent is not None:
            parent.add(meshed)
        if child is obj:
            obj = meshed
    return obj
