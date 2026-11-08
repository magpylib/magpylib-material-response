"""Tests for polyline fillet and grid-sweep helpers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from magpylib_material_response.polyline import (
    create_polyline_fillet,
    cross_section_frame,
    move_grid_along_polyline,
)


def test_fillet_3d_right_angle():
    poly = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
    out = create_polyline_fillet(poly, max_radius=0.2, N=5)
    assert out.shape[1] == 3
    assert np.isfinite(out).all()
    # endpoints preserved, sharp corner replaced by arc points
    np.testing.assert_allclose(out[0], poly[0])
    np.testing.assert_allclose(out[-1], poly[-1])
    assert not any(np.allclose(p, [1, 0, 0]) for p in out)
    # arc points keep the fillet radius to the circle centre (0.8, 0.2, 0)
    arc = out[1:-1]
    radii = np.linalg.norm(arc - np.array([0.8, 0.2, 0.0]), axis=1)
    np.testing.assert_allclose(radii, 0.2, rtol=1e-12)


def test_fillet_2d_matches_3d():
    """Regression: 2D input crashed in np.cross, masked as 'radius too large'."""
    poly2 = np.array([[0, 0], [1, 0], [1, 1]], dtype=float)
    poly3 = np.column_stack([poly2, np.zeros(len(poly2))])
    out2 = create_polyline_fillet(poly2, max_radius=0.2, N=5)
    out3 = create_polyline_fillet(poly3, max_radius=0.2, N=5)
    assert out2.shape[1] == 2
    np.testing.assert_allclose(out2, out3[:, :2])


def test_fillet_radius_autoreduced():
    """An oversized radius is reduced instead of raising."""
    poly = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
    out = create_polyline_fillet(poly, max_radius=10.0, N=5)
    assert np.isfinite(out).all()


def test_fillet_closed_loop():
    square = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=float
    )
    out = create_polyline_fillet(square, max_radius=0.1, N=4)
    np.testing.assert_allclose(out[0], out[-1])
    assert np.isfinite(out).all()


def test_fillet_collinear_points_pass_through():
    """Collinear middle vertices (degenerate arc) must not produce NaNs."""
    poly = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    out = create_polyline_fillet(poly, max_radius=0.2, N=5)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[0], poly[0])
    np.testing.assert_allclose(out[-1], poly[-1])


_L = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]  # turns left: vector area +z/2


@pytest.mark.parametrize(
    ("verts", "axis", "angle", "u", "v"),
    [
        # u along the vector area, v = t x u
        (_L, None, 0, [0, 0, 1], [0, -1, 0]),
        # turning right flips the area, so u: the moment follows the current
        ([[0, 0, 0], [1, 0, 0], [1, -1, 0]], None, 0, [0, 0, -1], [0, 1, 0]),
        # a small dent to the right before the left turn barely moves the area
        (
            [
                [0, 0, 0],
                [0.5, 0, 0],
                [0.6, -0.02, 0],
                [0.7, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
            ],
            None,
            0,
            [0, 0, 1],
            [0, -1, 0],
        ),
        # axis in place of the area, only its part across the first segment
        ([[0, 0, 0], [1, 0, 0]], [1, 0, 1], 0, [0, 0, 1], [0, -1, 0]),
        # angle turns the frame about t, from u toward v
        (_L, None, 90, [0, -1, 0], [0, 0, -1]),
        # 2D: u = +z for a left turn, and for a straight line: v right of t
        ([[0, 0], [1, 0], [1, 1]], None, 0, [0, 0, 1], [0, -1, 0]),
        ([[0, 0], [1, 0]], None, 0, [0, 0, 1], [0, -1, 0]),
    ],
)
def test_cross_section_frame(verts, axis, angle, u, v):
    """The documented frame, and the grid placed in it at verts[0]."""
    verts = np.array(verts, float)
    fu, fv = cross_section_frame(verts, axis=axis, angle=angle)
    np.testing.assert_allclose(fu, u, atol=1e-12)
    np.testing.assert_allclose(fv, v, atol=1e-12)
    if verts.shape[1] == 2:
        grid, start = [0.2], verts[0] + 0.2 * fv[:2]
    else:
        grid, start = [[0.1, 0.2]], verts[0] + 0.1 * fu + 0.2 * fv
    out = move_grid_along_polyline(verts, grid, axis=axis, angle=angle)
    assert out.shape == (1, len(verts), verts.shape[1])
    np.testing.assert_allclose(out[:, 0], [start], atol=1e-12)


def test_move_grid_follows_the_polyline():
    """Regression #18: moving and turning the polyline moves the windings
    with it, wherever it starts."""
    rng = np.random.default_rng(0)
    verts = rng.normal(size=(6, 3))
    verts = np.concatenate([verts, verts[:1]])  # closed and non-planar
    grid = rng.normal(scale=0.05, size=(5, 2))
    rot = Rotation.from_rotvec([0.3, -1.2, 2.0])
    shift = np.array([5.0, -3.0, 2.0])
    out = move_grid_along_polyline(verts, grid)
    moved = move_grid_along_polyline(rot.apply(verts) + shift, grid)
    np.testing.assert_allclose(
        moved, rot.apply(out.reshape(-1, 3)).reshape(out.shape) + shift
    )


def test_move_grid_closed_loop_closes_onto_its_start():
    """A loop that is not flat brings the cross-section back turned by 86°;
    each winding must still close onto its own start, not another's."""
    verts = np.array([[0, 0, 0], [2, 0, 0], [1, 2, 0.5], [1, 2, -0.5], [0, 0, 0]])
    grid = np.array([[a, b] for a in (-0.3, 0, 0.3) for b in (0, 0.3)])
    out = move_grid_along_polyline(verts, grid)
    # the documented frame: t = +x, vector area (-1, 0, 2), so u = +z, v = -y
    u, v = np.array([0, 0, 1.0]), np.array([0, -1.0, 0])
    # the closing section, read across the first segment, is the grid again
    back = np.column_stack([out[:, 0] @ u, out[:, 0] @ v])
    np.testing.assert_allclose(back, grid, atol=1e-12)
    np.testing.assert_array_equal(out[:, 0], out[:, -1])


@pytest.mark.parametrize(
    ("verts", "grid", "axis", "match"),
    [
        (_L, [[0, 0, 0.1]], None, "Before version 0.12"),
        ([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0.1]], None, "straight polyline"),
        (
            [[0, 0, 0], [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0], [0, 0, 0]],
            [[0, 0.1]],
            None,
            "figure of eight",
        ),
        ([[0, 0, 0], [1, 0, 0]], [[0, 0.1]], [2, 0, 0], "parallel"),
        ([[0, 0, 0], [1, 0, 0]], [[0, 0.1]], [0, 0, 0], "zero vector"),
        ([[0, 0], [1, 0]], [0.1], [0, 0, 1], "3D polylines"),
        ([[0, 0, 0], [2, 0, 0], [0, 0, 0]], [[0, 0.1]], [0, 0, 1], "turns back"),
        ([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]], [[0, 0.1]], None, "same point"),
    ],
)
def test_move_grid_errors(verts, grid, axis, match):
    with pytest.raises(ValueError, match=match):
        move_grid_along_polyline(np.array(verts, float), grid, axis=axis)
