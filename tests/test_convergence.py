from __future__ import annotations

import magpylib as magpy
import numpy as np
import pytest

from magpylib_material_response.convergence import SAFETY_FACTOR, _gci, mesh_study
from magpylib_material_response.meshing import mesh_Cuboid


@pytest.mark.parametrize(
    ("limit", "scale"),
    [(1.0, -0.5), (1.0, 3.0), (np.ones((4, 3)), np.linspace(-1, 1, 12).reshape(4, 3))],
    ids=["from-below", "from-above", "field"],
)
def test_gci_recovers_a_power_law(limit, scale):
    """The order and the limit of an exact power law, and the error of the finest
    value relative to itself, times the safety factor: a quantity falling toward
    its limit is not measured against its larger coarse values."""
    cells = np.array([100, 400, 2000])
    values = [limit + scale * n**-0.5 for n in cells]
    order, estimate, error, errors = _gci(cells, values)
    assert order == pytest.approx(0.5, rel=1e-9)
    np.testing.assert_allclose(estimate, limit, rtol=1e-9, atol=1e-12)
    finest = np.asarray(values[-1])
    assert error == pytest.approx(
        SAFETY_FACTOR * np.linalg.norm(finest - limit) / np.linalg.norm(finest)
    )
    size = (lambda a: np.linalg.norm(a, axis=-1)) if finest.ndim else abs
    np.testing.assert_allclose(
        errors, SAFETY_FACTOR * size(finest - limit) / size(finest), rtol=1e-9
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
def test_gci_declines_unsteady_results(values):
    order, estimate, error, errors = _gci([27, 216, 1728], values)
    assert order is None
    np.testing.assert_array_equal(estimate, values[-1])
    assert np.isnan(error)
    assert np.all(np.isnan(errors))


def test_gci_lets_a_few_turning_entries_through():
    """Many sensors converge steadily while a weak one turns: one entry of 20
    carrying a few percent of the change does not veto the estimate."""
    cells = np.array([100, 400, 2000])
    values = [1.0 + np.linspace(1, 2, 20) * n**-0.5 for n in cells]
    values[1][0] += 0.004  # the weakest entry overshoots on the middle mesh
    order, _, error, errors = _gci(cells, values)
    assert order is not None
    assert errors.shape == (20,)
    assert np.isfinite(error)


def test_gci_handles_a_limit_of_zero():
    order, estimate, error, _ = _gci([1, 8, 64], [-0.5, -0.25, -0.125])
    assert order == pytest.approx(1 / 3)
    assert estimate == pytest.approx(0.0, abs=1e-12)
    assert error == pytest.approx(SAFETY_FACTOR)


def _soft_cube():
    cube = magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0))
    cube.susceptibility = 1000
    cube.H_ext = (0, 0, 0.01)
    return cube


def test_mesh_study_of_a_soft_cube():
    """A soft cube converges from below: the estimate lies above the finest
    value, at an order of about 0.5 in the cell count, within a percent or two."""

    def moment(solved):
        cells = solved.sources_all
        return sum(c.polarization[2] * np.prod(c.dimension) for c in cells) / magpy.mu_0

    study = mesh_study(
        magpy.Collection(_soft_cube()),
        moment,
        target_elems=(27, 216, 1728),
        mesher=lambda coll, n: mesh_Cuboid(coll.sources[0], n),
    )
    np.testing.assert_array_equal(study.cells, [27, 216, 1728])
    assert 0.3 < study.order < 0.7
    assert study.estimate > study.values[-1]
    assert 0.005 < study.error < 0.03
    assert study.last_change == pytest.approx(
        (study.values[-1] - study.values[-2]) / study.values[-1]
    )


def test_mesh_study_checks_the_meshes():
    with pytest.raises(ValueError, match="increasing"):
        mesh_study(magpy.Collection(), [(0, 0, 0)], target_elems=(100, 100))
    with pytest.warns(UserWarning, match="grew only"):
        mesh_study(
            magpy.Collection(_soft_cube()),
            [(0, 0, 0.01)],
            target_elems=(8, 9, 10),
            mesher=lambda coll, n: mesh_Cuboid(coll.sources[0], (n, 1, 1)),
        )
