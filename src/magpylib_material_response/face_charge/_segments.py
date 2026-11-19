"""The field of CylinderSegment magnets at a solved part's Gauss points.

A magnet touching a solved part has the part's Gauss points on its face, and
segments that meet put them beside each other's planes. There magpylib's
segment kernel goes wrong: within about 1e-8 m of the planes and cylinders that
bound it, wherever a point's coordinate meets one of them, on the face or beside
it, and it fails outright at 1e-10 m (magpylib#1073). Points that close are
stepped straight off first.
"""

from __future__ import annotations

import magpylib as magpy
import numpy as np

#: a CylinderSegment's field is read this share of the parts' size off its
#: faces: closer in magpylib's kernel goes wrong (magpylib#1073), and at 1e-10
#: m it fails outright. B's normal part, all the load takes, is the same on both
#: sides of a face, and the step is taken straight off the face.
SEGMENT_CLEARANCE = 1e-6


def _segment_frame(segment):
    """``segment``'s position and rotation, of its last pose on a path."""
    position = np.reshape(segment.position, (-1, 3))[-1]
    rotation = segment.orientation
    return position, rotation if rotation.single else rotation[-1]


def _segment_surface_distances(segment, local):
    """Each point's distance (n, 5) to the surfaces that bound ``segment``
    (points in its own frame): the planes of its ends and cuts and the
    cylinders of its radii, all of them unbounded, since magpylib's kernel goes
    wrong where a point's coordinate meets one of them, on the face or beside
    it; and the unit normal (n, 5, 3) that leads off each on the point's own
    side."""
    x, y, z = local.T
    r1, r2, h, phi1, phi2 = segment.dimension
    r = np.hypot(x, y)
    radial = np.stack([x, y, np.zeros_like(x)], 1) / np.maximum(r, 1e-300)[:, None]
    radial[r == 0] = (1.0, 0.0, 0.0)  # on the axis: any way off it
    dist = [np.abs(np.abs(z) - h / 2)]
    normal = [
        np.outer(np.where(np.abs(z) >= h / 2, np.sign(z), -np.sign(z)), [0, 0, 1])
    ]
    for radius in (r1, r2):
        dist.append(np.abs(r - radius))
        normal.append(radial * np.where(r >= radius, 1.0, -1.0)[:, None])
    for phi in np.radians([phi1, phi2]):
        n = np.array([-np.sin(phi), np.cos(phi), 0.0])
        side = local @ n
        dist.append(np.abs(side))
        normal.append(np.outer(np.where(side >= 0, 1.0, -1.0), n))
    return np.stack(dist, 1), np.stack(normal, 1)


def segments_field(segments, points, size):
    """B of the CylinderSegments at ``points``. A point closer than
    ``SEGMENT_CLEARANCE`` of ``size`` to a surface of any of them is first
    stepped straight off it, to that clearance: a magnet touching a solved
    part has the part's Gauss points on its face, and segments that meet put
    them beside each other's faces, where one's field alone is large and only
    the sum is right. One position per point serves every segment, the step
    being far too short to matter to the others, so all are read in one call;
    magpylib is never called closer in."""
    tol = SEGMENT_CLEARANCE * size
    at = np.array(points, float)
    poses = [_segment_frame(seg) for seg in segments]
    for _ in range(4):  # off each surface in turn: an edge, a corner, a seam
        moved = False
        for seg, (pos, rot) in zip(segments, poses, strict=True):
            local = rot.inv().apply(at - pos)
            dist, normal = _segment_surface_distances(seg, local)
            close = dist < tol
            rows = np.flatnonzero(close.any(axis=1))
            if not rows.size:
                continue
            face = np.argmin(np.where(close[rows], dist[rows], np.inf), axis=1)
            local[rows] += normal[rows, face] * (tol - dist[rows, face])[:, None]
            at[rows] = rot.apply(local[rows]) + pos
            moved = True
        if not moved:
            break
    return magpy.getB(segments, at, sumup=True, squeeze=False).reshape(-1, 3)
