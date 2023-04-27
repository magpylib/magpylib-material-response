"""demag_functions"""
# +
# pylint: disable=invalid-name, redefined-outer-name, protected-access
import sys
from collections import Counter

import magpylib as magpy
import numpy as np
from loguru import logger
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib.magnet import Cuboid
from scipy.spatial.transform import Rotation as R

from magpylib_material_response.utils import timelog

config = {
    "handlers": [
        dict(
            sink=sys.stdout,
            colorize=True,
            format=(
                "<magenta>{time:YYYY-MM-DD at HH:mm:ss}</magenta>"
                " | <level>{level:^8}</level>"
                " | <cyan>{function}</cyan>"
                " | <yellow>{extra}</yellow> {level.icon:<2} {message}"
            ),
        ),
    ],
}
logger.configure(**config)


def get_xis(*sources, xi=None):
    """Return a list of length (len(sources)) with xi values
    Priority is given at the source level, hovever if value is not found, it is searched
    up the parent tree, if available. Raises an error if no value is found when reached
    the top level of the tree."""
    xis = []
    for src in sources:
        xi = getattr(src, "xi", None)
        if xi is None:
            if src.parent is None:
                raise ValueError(
                    "No susceptibility `xi` defined in any parent collection"
                )
            xis.extend(get_xis(src.parent))
        else:
            xis.append(xi)
    return xis


def demag_tensor(
    src_list,
    pairs_matching=False,
    split=False,
    max_dist=0,
    min_log_time=1,
):
    """
    Compute the demagnetization tensor T based on point matching (see Chadbec 2006)
    for n sources in the input collection.

    Parameters
    ----------
    collection: magpylib.Collection object with n magnet sources
        Each magnet source in collection is treated as a magnetic cell.

    pairs_matching: bool
        If True, equivalent pair of interactions are identified and unique pairs are
        calculated only once and copied to duplicates.

    split: int
        Number of times the sources list is splitted before getH calculation ind demag
        tensor calculation

    min_log_time:
        Minimum logging time in seconds. If computation time is below this value, step
        will not be logged.

    Returns
    -------
    Demagnetization tensor: ndarray, shape (3,n,n,3)

    TODO: allow multi-point matching
    TODO: allow current sources
    TODO: allow external stray fields
    TODO: status bar when n>1000
    TODO: Speed up with direct interface for field computation
    TODO: Use newell formulas for cube-cube interactions
    """
    nof_src = len(src_list)

    if pairs_matching and split != 1:
        raise ValueError("Pairs matching does not support splitting")
    elif max_dist != 0:
        mask_inds, getH_params, pos0, rot0 = filter_distance(
            src_list, max_dist, return_params=False, return_base_geo=True
        )
    elif pairs_matching:
        getH_params, mask_inds, unique_inv_inds, pos0, rot0 = match_pairs(src_list)
    else:
        pos0 = np.array([getattr(src, "barycenter", src.position) for src in src_list])
        rotQ0 = [src.orientation.as_quat() for src in src_list]
        rot0 = R.from_quat(rotQ0)

    H_point = []
    for unit_mag in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        mag_all = rot0.inv().apply(unit_mag)
        # point matching field and demag tensor
        with timelog(f"getH with unit_mag={unit_mag}", min_log_time=min_log_time):
            if pairs_matching or max_dist != 0:
                magnetization = np.repeat(mag_all, len(src_list), axis=0)[mask_inds]
                H_unique = magpy.getH(
                    "Cuboid", magnetization=magnetization, **getH_params
                )
                if max_dist != 0:
                    H_temp = np.zeros((len(src_list) ** 2, 3))
                    H_temp[mask_inds] = H_unique
                    H_unit_mag = H_temp
                else:
                    H_unit_mag = H_unique[unique_inv_inds]
            else:
                for src, mag in zip(src_list, mag_all):
                    src.magnetization = mag
                if split > 1:
                    src_list_split = np.array_split(src_list, split)
                    with logger.contextualize(
                        task="Splitting field calculation", split=split
                    ):
                        H_unit_mag = []
                        for split_ind, src_list_subset in enumerate(src_list_split):
                            logger.info(
                                f"Sources subset {split_ind+1}/{len(src_list_split)}"
                            )
                            if src_list_subset.size > 0:
                                H_unit_mag.append(
                                    magpy.getH(src_list_subset.tolist(), pos0)
                                )
                        H_unit_mag = np.concatenate(H_unit_mag, axis=0)
                else:
                    H_unit_mag = magpy.getH(src_list, pos0)
            H_point.append(H_unit_mag)  # shape (n_cells, n_pos, 3_xyz)

    # shape (3_unit_mag, n_cells, n_pos, 3_xyz)
    T = np.array(H_point).reshape((3, nof_src, nof_src, 3))

    return T


def filter_distance(
    src_list,
    max_dist,
    min_log_time=1,
    return_params=False,
    return_base_geo=False,
):
    """filter indices by distance parameter"""
    with timelog("Distance filter", min_log_time=min_log_time):
        all_cuboids = all(isinstance(src, Cuboid) for src in src_list)
        if not all_cuboids:
            raise ValueError(
                "filter_distance only implemented if all sources are Cuboids"
            )
        pos0 = np.array([getattr(src, "barycenter", src.position) for src in src_list])
        rotQ0 = [src.orientation.as_quat() for src in src_list]
        rot0 = R.from_quat(rotQ0)
        dim0 = [src.dimension for src in src_list]

        pos2 = np.tile(pos0, (len(pos0), 1)) - np.repeat(pos0, len(pos0), axis=0)
        dist2 = np.linalg.norm(pos2, axis=1)
        dim2 = np.tile(dim0, (len(dim0), 1)), np.repeat(dim0, len(dim0), axis=0)
        maxdim2 = np.concatenate(dim2, axis=1).max(axis=1)
        mask = (dist2 / maxdim2) < max_dist
        if return_params:
            params = dict(
                observers=np.tile(pos0, (len(src_list), 1))[mask],
                position=np.repeat(pos0, len(src_list), axis=0)[mask],
                orientation=R.from_quat(np.repeat(rotQ0, len(src_list), axis=0))[mask],
                dimension=np.repeat(dim0, len(src_list), axis=0)[mask],
            )
        dsf = sum(mask) / len(mask) * 100
    log_msg = (
        "Interaction pairs left after distance factor filtering: "
        f"<blue>{dsf:.2f}%</blue>"
    )
    if dsf == 0:
        logger.opt(colors=True).warning(log_msg)
    else:
        logger.opt(colors=True).success(log_msg)
    out = [mask]
    if return_params:
        out.append(params)
    if return_base_geo:
        out.extend([pos0, rot0])
    if len(out) == 1:
        return out[0]
    return tuple(out)


def match_pairs(src_list, min_log_time=1):
    """match all pairs of sources from `src_list`"""
    with timelog("Pairs matching", min_log_time=min_log_time):
        all_cuboids = all(isinstance(src, Cuboid) for src in src_list)
        if not all_cuboids:
            raise ValueError(
                "Pairs matching only implemented if all sources are Cuboids"
            )
        pos0 = np.array([getattr(src, "barycenter", src.position) for src in src_list])
        rotQ0 = [src.orientation.as_quat() for src in src_list]
        rot0 = R.from_quat(rotQ0)
        dim0 = [src.dimension for src in src_list]
        len_src = len(src_list)
        num_of_pairs = len_src**2
        with logger.contextualize(task="Match interactions pairs"):
            logger.info("position")
            pos2 = np.tile(pos0, (len_src, 1)) - np.repeat(pos0, len_src, axis=0)
            logger.info("orientation")
            rotQ2a = np.tile(rotQ0, (len_src, 1)).reshape((num_of_pairs, -1))
            rotQ2b = np.repeat(rotQ0, len_src, axis=0).reshape((num_of_pairs, -1))
            logger.info("dimension")
            dim2 = np.tile(dim0, (len_src, 1)) - np.repeat(dim0, len_src, axis=0)
            logger.info("concatenate properties")
            prop = (np.concatenate([pos2, rotQ2a, rotQ2b, dim2], axis=1) + 1e-9).round(
                8
            )
            logger.info("find unique indices")
            _, unique_inds, unique_inv_inds = np.unique(
                prop, return_index=True, return_inverse=True, axis=0
            )
            perc = len(unique_inds) / len(unique_inv_inds) * 100
            logger.opt(colors=True).info(
                "Interaction pairs left after pair matching filtering: "
                f"<blue>{perc:.2f}%</blue>"
            )

        params = dict(
            observers=np.tile(pos0, (len(src_list), 1))[unique_inds],
            position=np.repeat(pos0, len(src_list), axis=0)[unique_inds],
            orientation=R.from_quat(rotQ2b)[unique_inds],
            dimension=np.repeat(dim0, len(src_list), axis=0)[unique_inds],
        )
    return params, unique_inds, unique_inv_inds, pos0, rot0


def apply_demag(
    collection,
    xi=None,
    inplace=False,
    pairs_matching=False,
    max_dist=0,
    split=1,
    min_log_time=1,
    style=None,
):
    """
    Computes the interaction between all collection magnets and fixes their
    magnetization.

    Parameters
    ----------
    collection: magpylib.Collection object with n magnet sources
        Each magnet source in collection is treated as a magnetic cell.

    xi: array_like, shape (n,)
        Vector of n magnetic susceptibilities of the cells. If not defined, values are
        searched at object level or parent level if needed.

    inplace: bool
        If True, applies demagnetization on a copy of the input collection and returns
        the demagnetized collection

    pairs_matching: bool
        If True, equivalent pair of interactions are identified and unique pairs are
        calculated only once and copied to duplicates. This parameter is not compatible
        with `max_dist` or `split` and applies only cuboid cells.

    max_dist: float
        Posivive number representing the max_dimension to distance ratio for each pair
        of interacting cells. This filters out far interactions. If `max_dist=0`, all
        interactions are calculated. This parameter is not compatible with
        `pairs_matching` or `split` and applies only cuboid cells.

    split: int
        Number of times the sources list is splitted before getH calculation ind demag
        tensor calculation. This parameter is not compatible with `pairs_matching` or
        `max_dist`.

    min_log_time:
        Minimum logging time in seconds. If computation time is below this value, step
        will not be logged.

    style: dict
        Set collection style. If `inplace=False` only affects the copied collection

    Returns
    -------
    None
    """
    if not inplace:
        collection = collection.copy()
    if style is not None:
        collection.style = style
    srcs = collection.sources_all
    src_with_paths = [src for src in srcs if src.position.ndim != 1]
    if src_with_paths:
        raise ValueError(
            f"{len(src_with_paths)} objects with paths, found. Demagnetization of "
            "objects with paths is not yet supported"
        )
    magnets_list = [src for src in srcs if isinstance(src, BaseMagnet)]
    currents_list = [src for src in srcs if isinstance(src, BaseCurrent)]
    others_list = [
        src
        for src in srcs
        if not isinstance(src, (BaseMagnet, BaseCurrent, magpy.Sensor))
    ]
    if others_list:
        raise TypeError(
            "Only Magnet and Current sources supported. "
            "Incompatible objects found: "
            f"{Counter(s.__class__.__name__ for s in others_list)}"
        )
    n = len(magnets_list)
    counts = Counter(s.__class__.__name__ for s in magnets_list)
    inplace_str = f"""{" (inplace)" if inplace else ""}"""
    lbl = collection.style.label
    coll_str = str(collection) if not lbl else lbl
    demag_msg = (
        f"Demagnetization{inplace_str} of <blue>{coll_str}</blue>"
        f" with {n} cells - {counts}"
    )
    with timelog(demag_msg, min_log_time=min_log_time):
        # set up mr
        mag_magnets = [
            src.orientation.apply(src.magnetization) for src in magnets_list
        ]  # ROTATION CHECK
        mag_magnets = np.reshape(
            mag_magnets, (3 * n, 1), order="F"
        )  # shape ii = x1, ... xn, y1, ... yn, z1, ... zn

        # set up S
        if xi is None:
            xi = get_xis(*magnets_list)
        xi = np.array(xi)
        if len(xi) != n:
            raise ValueError(
                "Apply_demag input collection and xi must have same length."
            )
        S = np.diag(np.tile(xi, 3))  # shape ii, jj

        # set up T (3 mag unit, n cells, n positions, 3 Bxyz)
        with timelog("Demagnetization tensor calculation", min_log_time=min_log_time):
            T = demag_tensor(
                magnets_list,
                split=split,
                pairs_matching=pairs_matching,
                max_dist=max_dist,
            )

            T *= 4 * np.pi / 10
            T = T.swapaxes(2, 3).reshape((3 * n, 3 * n)).T  # shape ii, jj

        mag_tolal = mag_magnets

        if currents_list:
            with timelog(
                "Add current sources contributions", min_log_time=min_log_time
            ):
                pos = np.array([src.position for src in magnets_list])
                mag_currents = magpy.getB(currents_list, pos, sumup=True)
                mag_currents = np.reshape(mag_currents, (3 * n, 1), order="F")
                mag_tolal += np.matmul(S, mag_currents)

        # set up Q
        Q = np.eye(3 * n) - np.matmul(S, T)

        # determine new magnetization vectors
        with timelog("Solving of linear system", min_log_time=1):
            mag_new = np.linalg.solve(Q, mag_tolal)

        mag_new = np.reshape(mag_new, (n, 3), order="F")
        # mag_new *= .4*np.pi

        for s, mag in zip(collection.sources_all, mag_new):
            s.magnetization = s.orientation.inv().apply(mag)  # ROTATION CHECK

    if not inplace:
        return collection
