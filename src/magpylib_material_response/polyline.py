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
    max_ratio : float, optional, default: 0.5
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
        logger.debug(f"r: {r}, d: {d}, norm_ab: {norm_ab}, norm_bc: {norm_bc}")
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
        np.dot(start - center, end - center)
        / (np.linalg.norm(start - center) * np.linalg.norm(end - center))
    )
    angles = np.linspace(0, angle_diff, n_points)
    v = start - center
    w = np.cross(v, end - start)
    w /= np.linalg.norm(w)
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
    points = np.array(polyline)
    radius = max_radius
    if radius == 0 or N == 0:
        return points

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
    return np.array(filleted_points)


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


def move_grid_along_polyline(verts: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Move a grid along a polyline, defined by the vertices.

    Parameters
    ----------
    verts : np.ndarray, shape (n, d)
        Array of polyline vertices, where n is the number of vertices and d is the dimension.
    grid : np.ndarray, shape (m, d)
        Array of grid points to move along the polyline, where m is the number of points.

    Returns
    -------
    np.ndarray, shape (m, n, d)
        Array of moved grid points along the polyline, with the same dimensions as the input grid.
    """
    grid = grid.copy()
    pts = [grid]
    normals = _bisectors(verts)
    closed = np.allclose(verts[0], verts[-1])
    if closed:
        v_ext = np.concatenate([verts[-2:], verts[1:2]])
        last_normal = _bisectors(v_ext)
    else:
        last_normal = [verts[-1] - verts[-2]]
    normals = np.concatenate([normals, last_normal])
    for i in range(len(verts) - 1):
        plane_point = verts[i + 1]
        plane_normal = normals[i]
        line_points = pts[-1]
        line_directions = np.tile(verts[i + 1] - verts[i], (line_points.shape[0], 1))
        pts1 = _line_plane_intersection(
            plane_point, plane_normal, line_points, line_directions
        )
        pts.append(pts1)
    if closed:
        pts[0] = pts[-1]
    return np.array(pts).swapaxes(0, 1)
