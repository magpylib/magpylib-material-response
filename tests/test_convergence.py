from __future__ import annotations

import magpylib as magpy
import numpy as np
import pytest

from magpylib_material_response.convergence import _richardson, mesh_study
from magpylib_material_response.meshing import mesh_Cuboid


@pytest.mark.parametrize("shape", [(), (4, 3)], ids=["scalar", "field"])
def test_richardson_recovers_a_power_law(shape):
    rng = np.random.default_rng(0)
    limit, scale = rng.normal(size=shape), rng.normal(size=shape)
    cells = np.array([100, 400, 2000])
    values = [limit - scale * n**-0.5 for n in cells]
    order, estimate, error = _richardson(cells, values)
    assert order == pytest.approx(0.5, rel=1e-9)
    np.testing.assert_allclose(estimate, limit, rtol=1e-9, atol=1e-12)
    largest = max(np.linalg.norm(v) for v in values)
    assert error == pytest.approx(np.linalg.norm(limit - values[-1]) / largest)


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
def test_richardson_declines_unsteady_results(values):
    order, estimate, error = _richardson([27, 216, 1728], values)
    assert order is None
    np.testing.assert_array_equal(estimate, values[-1])
    assert np.isnan(error)


def test_richardson_handles_a_limit_of_zero():
    order, estimate, error = _richardson([1, 8, 64], [-0.5, -0.25, -0.125])
    assert order == pytest.approx(1 / 3)
    assert estimate == pytest.approx(0.0, abs=1e-12)
    assert error == pytest.approx(0.125 / 0.5)


def test_mesh_study_of_a_soft_cube():
    """A soft cube converges from below: the estimate lies above the finest
    value, at an order of about 0.5 in the cell count."""
    cube = magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0))
    cube.susceptibility = 1000
    cube.H_ext = (0, 0, 0.01)

    def moment(solved):
        cells = solved.sources_all
        return sum(c.polarization[2] * np.prod(c.dimension) for c in cells) / magpy.mu_0

    study = mesh_study(
        magpy.Collection(cube),
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


def test_mesh_study_needs_increasing_targets():
    with pytest.raises(ValueError, match="increasing"):
        mesh_study(magpy.Collection(), [(0, 0, 0)], target_elems=(100, 100))
