"""Tests for polyline fillet and grid-sweep helpers."""

from __future__ import annotations

import numpy as np

from magpylib_material_response.polyline import (
    create_polyline_fillet,
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


def test_move_grid_along_polyline_3d():
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
    grid = np.array([[0, 0, -0.1], [0, 0, 0.1]])
    out = move_grid_along_polyline(verts, grid)
    assert out.shape == (2, 3, 3)
    assert np.isfinite(out).all()


def test_move_grid_along_polyline_2d():
    """Regression: 2D verts crashed in the 3D-only plane intersection."""
    verts = np.array([[0, 0], [1, 0], [1, 1]], dtype=float)
    grid = np.array([[0, -0.1], [0, 0.1]])
    out = move_grid_along_polyline(verts, grid)
    assert out.shape == (2, 3, 2)
    assert np.isfinite(out).all()
