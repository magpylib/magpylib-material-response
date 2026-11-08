from __future__ import annotations

import numpy as np
from loguru import logger


def _find_circle_center_and_tangent_points(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, r: float, max_ratio: float = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Find the center of a circle and its tangent points with given vertices and radius.

    Parameters
    ----------
    a, b, c : array-like
        Vertices of a triangle (b is the middle vertex) with shape (2,) or (3,).
    r : float
        Radius of the circle.
    max_ratio : float, optional, default: 1
        Maximum allowed ratio of the distance to the tangent point relative to the length of the
        triangle sides.

    Returns
    -------
    tuple
        Circle center, tangent point a, tangent point b as NumPy arrays.
    """
    # Calculate the unit vectors along AB and BC
    norm_ab = np.linalg.norm(a - b)
    norm_bc = np.linalg.norm(c - b)
    ba_unit = (a - b) / norm_ab
    bc_unit = (c - b) / norm_bc

    dot_babc = np.dot(ba_unit, bc_unit)
    if dot_babc == -1:  # angle is 180°
        msg = "Triangle is flat"
        raise ValueError(msg)
    theta = np.arccos(dot_babc) / 2
    tan_theta = np.tan(theta)
    d = r / tan_theta
    if d > norm_bc * max_ratio or d > norm_ab * max_ratio:
        # rold, dold = r, d
        logger.debug(
            "Fillet parameters adjusted", r=r, d=d, norm_ab=norm_ab, norm_bc=norm_bc
        )
        d = min(norm_bc * max_ratio, norm_ab * max_ratio)
        r = d * tan_theta if theta > 0 else 0
        # warnings.warn(f"Radius {rold:.4g} is too big and has been reduced to {r:.4g}")
    ta = b + ba_unit * d
    tb = b + bc_unit * d

    rl = (d**2 + r**2) ** 0.5
    bisector = ba_unit + bc_unit
    unit_bisector = bisector / np.linalg.norm(bisector)
    circle_center = b + unit_bisector * rl

    return circle_center, ta, tb


def _interpolate_circle(
    center: np.ndarray, start: np.ndarray, end: np.ndarray, n_points: int
) -> list[np.ndarray]:
    """
    Interpolate points along a circle arc between two points.

    Parameters
    ----------
    center : array-like
        Center of the circle with shape (2,) or (3,).
    start, end : array-like
        Start and end points of the arc with shape (2,) or (3,).
    n_points : int
        Number of points to interpolate.

    Returns
    -------
    list
        List of NumPy arrays representing the interpolated points.
    """
    angle_diff = np.arccos(
        np.clip(
            np.dot(start - center, end - center)
            / (np.linalg.norm(start - center) * np.linalg.norm(end - center)),
            -1.0,
            1.0,
        )
    )
    v = start - center
    w = np.cross(v, end - start)
    w_norm = np.linalg.norm(w)
    if angle_diff == 0 or w_norm == 0:
        # degenerate arc (collinear points): nothing to sweep
        return [start, end]
    w /= w_norm
    angles = np.linspace(0, angle_diff, n_points)
    return [
        center + np.cos(angle) * v + np.sin(angle) * np.cross(w, v) for angle in angles
    ]


def _create_fillet_segment(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, r: float, N: int
) -> list[np.ndarray]:
    """
    Create a fillet segment with a given radius between three vertices.

    Parameters
    ----------
    a, b, c : array-like
        Vertices of a triangle (b is the middle vertex) with shape (2,) or (3,).
    r : float
        Radius of the fillet.
    N : int
        Number of points to interpolate along the fillet.

    Returns
    -------
    list
        List of NumPy arrays representing the fillet points.
    """
    try:
        res = _find_circle_center_and_tangent_points(a, b, c, r)
        circle_center, ta, tb = res
    except ValueError:
        return [b]
    return _interpolate_circle(circle_center, ta, tb, N)


def create_polyline_fillet(
    polyline: list | np.ndarray, max_radius: float, N: int
) -> np.ndarray:
    """
    Create a filleted polyline with specified maximum radius and number of points.

    Parameters
    ----------
    polyline : list or array-like
        List or array of vertices forming the polyline with shape (N, 2) or (N, 3).
    max_radius : float
        Maximum radius of the fillet.
    N : int
        Number of points to interpolate along the fillet.

    Returns
    -------
    numpy.ndarray
        Array of filleted points with shape (M, 2) or (M, 3), where M depends on the number of
        filleted segments.
    """
    points = np.array(polyline, dtype=float)
    radius = max_radius
    if radius == 0 or N == 0:
        return points

    # 2D polylines are lifted to z=0 so the single 3D code path (cross
    # products) serves both; the result is sliced back at the end.
    is_2d = points.shape[1] == 2
    if is_2d:
        points = np.column_stack([points, np.zeros(len(points))])

    closed = np.allclose(points[0], points[-1])
    if closed:
        points = np.append(points, points[1:2], axis=0)
    filleted_points = [points[0]]
    n = len(points)
    for i in range(1, n - 1):
        a, b, c = (
            filleted_points[-1],
            points[i],
            points[i + 1],
        )
        if closed and i == n - 2:
            c = filleted_points[1]
        try:
            filleted_points.extend(_create_fillet_segment(a, b, c, radius, N))
        except ValueError as exc:
            msg = f"The radius {radius} on position vertex {i} is too large"
            raise ValueError(msg) from exc
    if closed:
        filleted_points[0] = filleted_points[-1]
    else:
        filleted_points = np.append(filleted_points, points[-1:], axis=0)
    out = np.array(filleted_points)
    return out[:, :2] if is_2d else out


def _bisectors(polyline: np.ndarray) -> np.ndarray:
    """
    Calculate and normalize bisectors of the segments in a polyline.

    Parameters
    ----------
    polyline : numpy.ndarray
        A 2D array of shape (N, 3) representing N vertices of a polyline in 3D space.

    Returns
    -------
    bisectors_normalized : numpy.ndarray
        A 2D array of shape (N-2, 3) representing normalized bisectors for each pair of consecutive
        segments in the polyline.
    """
    # Calculate the segment vectors
    segment_vectors = np.diff(polyline, axis=0)

    # Normalize the segment vectors
    normalized_vectors = (
        segment_vectors / np.linalg.norm(segment_vectors, axis=1)[:, np.newaxis]
    )

    # Calculate the bisectors by adding normalized adjacent vectors
    bisectors = normalized_vectors[:-1] + normalized_vectors[1:]

    # Normalize the bisectors
    return bisectors / np.linalg.norm(bisectors, axis=1)[:, np.newaxis]


def _line_plane_intersection(
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    line_points: np.ndarray,
    line_directions: np.ndarray,
) -> np.ndarray:
    """
    Find the intersection points of multiple lines and a plane.

    Parameters
    ----------
    plane_point : numpy.ndarray
        A 1D array of shape (3,) representing a point on the plane.
    plane_normal : numpy.ndarray
        A 1D array of shape (3,) representing the normal vector of the plane.
    line_points : numpy.ndarray
        A 2D array of shape (N, 3) representing N points on the lines.
    line_directions : numpy.ndarray
        A 2D array of shape (N, 3) representing the direction vectors of the lines.

    Returns
    -------
    intersection_points : numpy.ndarray
        A 2D array of shape (N, 3) representing the intersection points of the lines and the plane.
    """
    # Calculate the plane equation coefficients A, B, C, and D
    A, B, C = plane_normal
    # x0, y0, z0 = plane_point
    D = -np.dot(plane_normal, plane_point)

    # Calculate the parameter t
    t = -(A * line_points[:, 0] + B * line_points[:, 1] + C * line_points[:, 2] + D) / (
        A * line_directions[:, 0]
        + B * line_directions[:, 1]
        + C * line_directions[:, 2]
    )

    # Find the intersection points by plugging t back into the parametric line equation
    return line_points + np.expand_dims(t, axis=-1) * line_directions


def cross_section_frame(
    verts: np.ndarray,
    axis: np.ndarray | None = None,
    angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The axes ``(u, v)`` of a cross-section at ``verts[0]``, across the first
    segment: the frame `move_grid_along_polyline` places its grid in.

    ``t`` is the direction of the first segment. ``u`` is the polyline's
    vector area ``A = 1/2 sum (verts[i] - verts[0]) x (verts[i+1] - verts[0])``,
    the polyline closed by its chord back to ``verts[0]``, made perpendicular
    to ``t``; ``v = t x u``. Then both are turned by ``angle`` about ``t``,
    from ``u`` toward ``v``.

    ``A`` points to where the polyline, closed by its chord, looks largest
    from, and its length is that largest shadow's area: on a flat loop, the
    normal on the side from which the loop runs counterclockwise. For a coil
    turn, ``u`` is the coil's axis, the direction of its moment for a current
    along the polyline, and ``v`` points outward, as in the centroid frame of
    finite-build coil codes. ``A`` depends on the whole polyline, and not on
    where a closed one starts: a small dent moves the frame only a little, and
    turning the polyline turns the frame with it.

    Parameters
    ----------
    verts : np.ndarray, shape (n, 3) or (n, 2)
        Vertices of the polyline.
    axis : array-like, shape (3,), optional
        Used in place of the vector area, for a 3D polyline. Only its part
        across the first segment counts. Required where the vector area has
        no part across it: a straight polyline, or one that encloses as much
        area turning one way as the other, like a figure of eight. Less than
        1e-4 of the square of the polyline's size (its bounding box diagonal)
        counts as none: the frame of a nearly straight polyline would follow
        its noise.
    angle : float, default 0
        Turn of the frame about ``t``, in degrees, for a 3D polyline.

    Returns
    -------
    u, v : np.ndarray, shape (3,)
        Unit vectors, at right angles to each other and to ``t``. For a 2D
        polyline, ``u`` is ``(0, 0, 1)`` or ``(0, 0, -1)``, ``v`` lies in the
        plane, and a straight 2D polyline takes ``u = (0, 0, 1)``: ``v`` to
        the right of ``t``.

    Raises
    ------
    ValueError
        If the vector area, or ``axis``, has no part across the first segment,
        ``axis`` or ``angle`` is given for a 2D polyline, the polyline has fewer
        than two vertices, two consecutive vertices are the same point, or it
        turns back on itself (within 0.1° of 180°).

    Examples
    --------
    A square turn in the yz-plane, run counterclockwise seen from +x:

    >>> import numpy as np
    >>> from magpylib_material_response.polyline import cross_section_frame
    >>> turn = [(0, 1, 0), (0, 1, 1), (0, -1, 1), (0, -1, -1), (0, 1, -1), (0, 1, 0)]
    >>> u, v = cross_section_frame(np.array(turn, float))
    >>> u, v
    (array([1., 0., 0.]), array([0., 1., 0.]))
    """
    verts = np.asarray(verts, dtype=float)
    angle = float(angle)
    _check_polyline(verts)
    if verts.shape[1] == 2:
        if axis is not None or angle:
            msg = (
                "axis and angle are for 3D polylines; a 2D one has its "
                "cross-section in its plane."
            )
            raise ValueError(msg)
        verts = np.column_stack([verts, np.zeros(len(verts))])
        if abs(_vector_area(verts)[2]) <= _NO_AREA * _span(verts) ** 2:
            axis = np.array([0.0, 0.0, 1.0])
    d = verts[1] - verts[0]
    t = d / np.linalg.norm(d)
    if axis is None:
        ref = _vector_area(verts)
        tol = _NO_AREA * _span(verts) ** 2
        what = (
            "The polyline's vector area has no part across its first segment "
            "(a straight polyline, or a figure of eight), so it does not orient "
            "the cross-section: pass axis, the direction of u."
        )
    else:
        ref = np.asarray(axis, dtype=float)
        if ref.shape != (3,):
            msg = f"axis must be a 3-vector, got shape {ref.shape}."
            raise ValueError(msg)
        tol = 1e-9 * np.linalg.norm(ref)
        if tol == 0:
            msg = "axis is the zero vector: it has no direction."
            raise ValueError(msg)
        what = "axis is parallel to the polyline's first segment."
    u = ref - (ref @ t) * t
    norm = np.linalg.norm(u)
    if norm <= tol:
        raise ValueError(what)
    u /= norm
    v = np.cross(t, u)
    if angle:
        c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        u, v = c * u + s * v, c * v - s * u
    return u, v


def _vector_area(verts: np.ndarray) -> np.ndarray:
    """Half the sum of (v_i - v_0) x (v_i+1 - v_0), closed by the chord."""
    r = np.concatenate([verts, verts[:1]]) - verts[0]
    return 0.5 * np.cross(r[:-1], r[1:]).sum(axis=0)


def _span(verts: np.ndarray) -> float:
    """Length of the polyline's bounding box diagonal: its scale."""
    return float(np.linalg.norm(np.ptp(verts, axis=0)))


#: A vector area under this share of the polyline's span squared orients
#: nothing: below it, the frame of a nearly straight polyline follows noise.
_NO_AREA = 1e-4


def _is_closed(verts: np.ndarray) -> bool:
    """Whether the last vertex is on the first, relative to the polyline's size."""
    return bool(np.linalg.norm(verts[-1] - verts[0]) <= 1e-9 * _span(verts))


def _check_polyline(verts: np.ndarray) -> None:
    """Refuse what has no frame or no mitre."""
    if verts.ndim != 2 or verts.shape[1] not in (2, 3) or len(verts) < 2:
        msg = (
            "verts must be at least two vertices, of shape (n, 3) or (n, 2); "
            f"got shape {verts.shape}."
        )
        raise ValueError(msg)
    _check_segments(verts, _is_closed(verts))


def move_grid_along_polyline(
    verts: np.ndarray,
    grid: np.ndarray,
    axis: np.ndarray | None = None,
    angle: float = 0.0,
) -> np.ndarray:
    """
    Sweep a cross-section along a polyline, one copy of the polyline per point.

    The cross-section is placed at the first vertex, across the first segment,
    and keeps its shape through every bend: one polyline per point, such as
    the windings of a coil from its winding pack.

    Parameters
    ----------
    verts : np.ndarray, shape (n, 3) or (n, 2)
        Vertices of the polyline.
    grid : np.ndarray, shape (m, 2) for a 3D polyline, (m,) for a 2D one
        Points of the cross-section, relative to ``verts[0]``: a point
        ``(a, b)`` sits at ``verts[0] + a*u + b*v``, in the frame
        `cross_section_frame` returns. For a 2D polyline a point is the offset
        ``b`` along ``v``. On a closed polyline the windings start and end on
        the mitre at ``verts[0]``, the plane halfway between the last segment
        and the first: each point is moved along ``t`` onto it.
    axis : array-like, shape (3,), optional
        Passed to `cross_section_frame`: used in place of the polyline's
        vector area, required where that has no part across the first segment.
    angle : float, default 0
        Passed to `cross_section_frame`: turn of the cross-section about the
        first segment, in degrees.

    Returns
    -------
    np.ndarray, shape (m, n, 3) or (m, n, 2)
        One polyline per cross-section point. When the polyline is closed (its
        last vertex on its first, within 1e-9 of its size), so is each of
        them.

    Raises
    ------
    ValueError
        If ``grid`` does not have one column fewer than ``verts``, or
        `cross_section_frame` raises: the frame is undefined, or the polyline
        has a repeated vertex or turns back on itself.

    Notes
    -----
    The frame is `cross_section_frame`'s: ``u`` along the polyline's vector
    area, made perpendicular to the first segment ``t``, and ``v = t x u``;
    for a coil turn, its axis and outward.

    At each corner the cross-section is cut by the mitre plane, which turns it
    about the corner's bend axis, never about the wire: the rotation-minimizing
    sweep of computer graphics. A closed polyline that is not flat therefore
    brings it back turned about the wire, by an angle set by the loop's shape
    (86° for the loop through (0, 0, 0), (2, 0, 0), (1, 2, 0.5) and
    (1, 2, -0.5)). That turn is undone in steps, in proportion to arc length,
    so each winding closes onto itself; its segments then run up to a few
    degrees off the polyline's, more for a wider cross-section.
    """
    verts = np.asarray(verts, dtype=float)
    grid = np.asarray(grid, dtype=float)
    dim = verts.shape[1]
    if dim == 2 and grid.ndim == 1:
        grid = grid[:, np.newaxis]
    if grid.ndim != 2 or grid.shape[1] != dim - 1:
        msg = (
            f"grid has shape {grid.shape}, but for a polyline in {dim}D it is "
            f"the cross-section with {dim - 1} coordinate(s) per point, relative "
            "to verts[0] and across the first segment, in the frame "
            "cross_section_frame(verts) returns. Before version 0.12 it was a "
            "grid in absolute coordinates."
        )
        raise ValueError(msg)
    u, v = cross_section_frame(verts, axis=axis, angle=angle)
    closed = _is_closed(verts)
    # 2D inputs are lifted to z=0 (the plane intersection is 3D-only) and
    # sliced back at the end.
    is_2d = dim == 2
    if is_2d:
        verts = np.column_stack([verts, np.zeros(len(verts))])
        grid = np.column_stack([np.zeros(len(grid)), grid[:, 0]])
    out = _sweep(verts, verts[0] + grid[:, :1] * u + grid[:, 1:] * v, closed)
    if closed and not is_2d:
        out = _undo_closing_turn(verts, grid, u, v, out)
    if closed:
        out[:, 0] = out[:, -1]
    return out[..., :2] if is_2d else out


def _check_segments(verts: np.ndarray, closed: bool) -> None:
    """Refuse what has no mitre: a segment of zero length, a turn within
    0.1° of 180°, whose mitre would be over 1,000 times wider than the
    cross-section."""
    d = np.diff(verts, axis=0)
    length = np.linalg.norm(d, axis=1)
    short = np.flatnonzero(length <= 1e-12 * length.max())
    if short.size:
        i = short[0]
        msg = f"verts[{i}] and verts[{i + 1}] are the same point."
        raise ValueError(msg)
    t = d / length[:, np.newaxis]
    turns = np.einsum("ij,ij->i", t, np.roll(t, -1, axis=0))
    turns = turns if closed else turns[:-1]
    back = np.flatnonzero(turns < -1 + 1e-6)
    if back.size:
        i = (back[0] + 1) % len(t)
        msg = (
            f"The polyline turns back on itself at verts[{i}]: a turn within "
            "0.1° of 180° has no usable mitre to carry the cross-section "
            "through."
        )
        raise ValueError(msg)


def _sweep(verts: np.ndarray, start: np.ndarray, closed: bool) -> np.ndarray:
    """`start`, points across the first segment, carried along the polyline:
    along each segment to the mitre plane at its end. Shape (m, n, 3)."""
    pts = [start]
    normals = _bisectors(verts)
    if closed:
        v_ext = np.concatenate([verts[-2:], verts[1:2]])
        last_normal = _bisectors(v_ext)
    else:
        last_normal = [verts[-1] - verts[-2]]
    normals = np.concatenate([normals, last_normal])
    for i in range(len(verts) - 1):
        line_points = pts[-1]
        line_directions = np.tile(verts[i + 1] - verts[i], (line_points.shape[0], 1))
        pts.append(
            _line_plane_intersection(
                verts[i + 1], normals[i], line_points, line_directions
            )
        )
    return np.array(pts).swapaxes(0, 1)


def _undo_closing_turn(
    verts: np.ndarray,
    grid: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """Close every winding onto itself on a loop that is not flat.

    The mitres turn the cross-section about each corner's bend axis, never
    about the wire. On a flat loop the bend axes are parallel and the turns
    cancel; on any other the cross-section comes back turned about the wire,
    and each winding would jump onto another's start. The turn is undone in
    steps, in proportion to arc length, so no one segment takes it all.
    """
    # The sweep is affine in the cross-section's coordinates: O + a*U + b*V.
    o, a_end, b_end = _sweep(verts, verts[0] + np.array([0 * u, u, v]), closed=True)
    a_vec, b_vec = a_end - o, b_end - o
    # where u and v come back, read across the first segment
    back = np.array([[a_vec[-1] @ u, b_vec[-1] @ u], [a_vec[-1] @ v, b_vec[-1] @ v]])
    turn = np.arctan2(back[1, 0] - back[0, 1], back[0, 0] + back[1, 1])
    if abs(turn) < 1e-9:  # flat, or turns that cancel: nothing to undo
        return out
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(verts, axis=0), axis=1))]
    theta = -turn * s / s[-1]
    c, sn = np.cos(theta), np.sin(theta)
    a = grid[:, :1] * c - grid[:, 1:] * sn
    b = grid[:, :1] * sn + grid[:, 1:] * c
    return o + a[..., np.newaxis] * a_vec + b[..., np.newaxis] * b_vec
