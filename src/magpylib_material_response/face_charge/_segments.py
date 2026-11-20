"""The field of CylinderSegment magnets at a solved part's Gauss points.

A magnet touching a solved part has the part's Gauss points on its face, and
segments that meet put them beside each other's planes. There magpylib's
segment kernel goes wrong (magpylib#1082): within about 1e-7 of the segment's
own size of the planes and cylinders that bound it, wherever a point's
coordinate meets one of them, on the face or beside it, and it gives NaN closer
in still. Points that close are stepped straight off first.
"""

from __future__ import annotations

import warnings

import magpylib as magpy
import numpy as np

#: a CylinderSegment's field is read at least this share of its own size off
#: its surfaces: the kernel's error scales with the segment, about 3e-5 of J at
#: 1e-6 and 3e-3 at 1e-7 of its size (magpylib#1082). B's normal part, all the
#: load takes, is the same on both sides of a face, and the step is taken
#: straight off the face, so it costs about this share too.
SEGMENT_CLEARANCE = 1e-6
#: passes over the segments while a step off one surface lands within another's
#: clearance: near the axis of many wedges each step meets the next cut plane
PASSES = 16


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
    up = np.where(z < 0, -1.0, 1.0)  # z = 0 on a segment thinner than the clearance
    normal = [np.outer(np.where(np.abs(z) >= h / 2, up, -up), [0, 0, 1])]
    for radius in (r1, r2):
        dist.append(np.abs(r - radius))
        normal.append(radial * np.where(r >= radius, 1.0, -1.0)[:, None])
    for phi in np.radians([phi1, phi2]):
        n = np.array([-np.sin(phi), np.cos(phi), 0.0])
        side = local @ n
        dist.append(np.abs(side))
        normal.append(np.outer(np.where(side >= 0, 1.0, -1.0), n))
    return np.stack(dist, 1), np.stack(normal, 1)


def _clearance(segment):
    """``SEGMENT_CLEARANCE`` of ``segment``'s own size: its outer radius or its
    height, the larger."""
    _, r2, h, *_ = segment.dimension
    return SEGMENT_CLEARANCE * max(r2, h)


def segments_field(segments, points):
    """B of the CylinderSegments at ``points``. A point closer than a segment's
    clearance (``_clearance``) to one of its surfaces is first stepped straight
    off it, to twice that: a magnet touching a solved part has the part's
    Gauss points on its face, and segments that meet put them beside each
    other's faces, where one's field alone is large and only the sum is right.
    One position per point serves every segment, the step being far too short
    to matter to the others, so all are read in one call. A point no step
    clears, or a field that is not finite, is a warning, not a silent NaN."""
    at = np.array(points, float)
    poses = [_segment_frame(seg) for seg in segments]
    tols = [_clearance(seg) for seg in segments]
    for _ in range(PASSES):  # off each surface in turn: an edge, a corner, a seam
        moved = False
        for seg, (pos, rot), tol in zip(segments, poses, tols, strict=True):
            local = rot.inv().apply(at - pos)
            dist, normal = _segment_surface_distances(seg, local)
            close = dist < tol
            rows = np.flatnonzero(close.any(axis=1))
            if not rows.size:
                continue
            face = np.argmin(np.where(close[rows], dist[rows], np.inf), axis=1)
            local[rows] += normal[rows, face] * (2 * tol - dist[rows, face])[:, None]
            at[rows] = rot.apply(local[rows]) + pos
            moved = True
        if not moved:
            break
    else:
        warnings.warn(
            "Some load points could not be stepped clear of the CylinderSegments' "
            "surfaces; their field may be off (magpylib#1082).",
            stacklevel=4,
        )
    B = magpy.getB(segments, at, sumup=True, squeeze=False).reshape(-1, 3)
    if not np.isfinite(B).all():
        warnings.warn(
            "A CylinderSegment's field is not finite at some load points "
            "(magpylib#1082); the solve that uses it is not to be trusted.",
            stacklevel=4,
        )
    return B
