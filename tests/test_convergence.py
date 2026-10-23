from __future__ import annotations

import magpylib as magpy
import numpy as np
import pytest

from magpylib_material_response.convergence import SAFETY_FACTOR, MeshStudy, mesh_study
from magpylib_material_response.meshing import mesh_Cuboid


@pytest.mark.parametrize(
    ("limit", "scale"),
    [(1.0, -0.5), (1.0, 3.0), (np.ones((4, 3)), np.linspace(-1, 1, 12).reshape(4, 3))],
    ids=["from-below", "from-above", "field"],
)
def test_estimate_recovers_a_power_law(limit, scale):
    """The order and the limit of an exact power law, and the error of the finest
    value relative to itself, times the safety factor: a quantity falling toward
    its limit is not measured against its larger coarse values."""
    cells = np.array([100, 400, 2000])
    values = [limit + scale * n**-0.5 for n in cells]
    study = MeshStudy.from_values(cells, values)
    assert study.order == pytest.approx(1.5, rel=1e-9)  # N**-0.5 is h**1.5
    np.testing.assert_allclose(study.estimate, limit, rtol=1e-9, atol=1e-12)
    finest = np.asarray(values[-1])
    assert study.error == pytest.approx(
        SAFETY_FACTOR * np.linalg.norm(finest - limit) / np.linalg.norm(finest)
    )
    size = (lambda a: np.linalg.norm(a, axis=-1)) if finest.ndim else abs
    np.testing.assert_allclose(
        study.errors, SAFETY_FACTOR * size(finest - limit) / size(finest), rtol=1e-9
    )


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 1.1, 1.05],
        [0.9, 0.99, 0.99001],
        [1.0, 2.0, 2.0 + 1 / (1 + 1e-6)],
        [[0.9, 0.01], [0.95, 0.02], [0.975, 0.0]],
    ],
    ids=["oscillating", "implausibly-fast", "barely-shrinking", "one-component-turns"],
)
def test_estimate_declines_unsteady_results(values):
    study = MeshStudy.from_values([27, 216, 1728], values)
    assert study.order is None
    np.testing.assert_array_equal(study.estimate, values[-1])
    assert np.isnan(study.error)
    assert np.all(np.isnan(study.errors))


def test_estimate_lets_a_few_turning_entries_through():
    """Many sensors converge steadily while a weak one turns: one entry of 20
    carrying a few percent of the change does not veto the estimate."""
    cells = np.array([100, 400, 2000])
    values = [1.0 + np.linspace(1, 2, 20) * n**-0.5 for n in cells]
    values[1][0] += 0.004  # the weakest entry overshoots on the middle mesh
    study = MeshStudy.from_values(cells, values)
    assert study.order is not None
    assert study.errors.shape == (20,)
    assert np.isfinite(study.error)


def test_estimate_handles_a_limit_of_zero():
    study = MeshStudy.from_values([1, 8, 64], [-0.5, -0.25, -0.125])
    assert study.order == pytest.approx(1.0)
    assert study.estimate == pytest.approx(0.0, abs=1e-12)
    assert study.error == pytest.approx(SAFETY_FACTOR)


def _soft_cube():
    cube = magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0))
    cube.susceptibility = 1000
    cube.H_ext = (0, 0, 0.01)
    return cube


def test_mesh_study_of_a_soft_cube():
    """A soft cube converges from below: the estimate lies above the finest
    value, at an order of about 1.5 in the cell size, within a percent or two."""

    def moment(solved):
        cells = solved.sources_all
        return sum(c.polarization[2] * np.prod(c.dimension) for c in cells) / magpy.mu_0

    study = mesh_study(magpy.Collection(_soft_cube()), moment, levels=(3, 6, 12))
    np.testing.assert_array_equal(study.cells, [27, 216, 1728])
    assert 0.9 < study.order < 2.1
    assert study.estimate > study.values[-1]
    assert 0.005 < study.error < 0.03
    assert study.last_change == pytest.approx(
        (study.values[-1] - study.values[-2]) / study.values[-1]
    )


@pytest.mark.parametrize(
    "observers",
    [
        np.array([(0, 0, 0.01), (0.01, 0, 0)]),
        magpy.Sensor(pixel=[(0, 0, 0.01), (0.01, 0, 0)]),
    ],
    ids=["positions", "sensor"],
)
def test_mesh_study_counts_cells_that_respond_and_errs_per_observer(observers):
    hard = magpy.magnet.Cuboid(
        dimension=(0.005,) * 3, position=(0.05, 0, 0), polarization=(0, 0, 1)
    )
    hard.susceptibility = 0
    study = mesh_study(magpy.Collection(_soft_cube(), hard), observers, (3, 6, 12))
    np.testing.assert_array_equal(study.cells, [27, 216, 1728])
    assert study.errors.shape == (2,)


def test_mesh_study_checks_the_meshes():
    cube = magpy.Collection(_soft_cube())
    with pytest.raises(ValueError, match="not more than"):
        mesh_study(cube, [(0, 0, 0.01)], levels=(2, 2))
    with pytest.raises(ValueError, match="inplace"):
        mesh_study(cube, [(0, 0, 0.01)], (1, 2), demag_kwargs={"inplace": True})
    with pytest.raises(ValueError, match="one of each"):
        MeshStudy.from_values([1, 8, 64], [1.0, 2.0])
    with pytest.warns(UserWarning, match="grew only"):
        mesh_study(
            magpy.Collection(_soft_cube()),
            [(0, 0, 0.01)],
            levels=(8, 9, 10),
            mesher=lambda coll, n: mesh_Cuboid(coll.sources[0], (n, 1, 1)),
        )
