"""A uniform applied field as a source, in all three solvers."""

from __future__ import annotations

import copy
import pickle

import magpylib as magpy
import numpy as np
import pytest

from magpylib_material_response import UniformField, from_json, to_json
from magpylib_material_response.cells import solve_cells
from magpylib_material_response.face_charge import solve_face_charge
from magpylib_material_response.meshing import mesh_all, mesh_Cuboid, mesh_uniform

B0 = np.array([0.0, 0.02, 0.1])
POINTS = np.array([[0.02, 0.001, 0.002], [0.0, 0.0, 0.03]])


def _soft():
    part = magpy.magnet.Cuboid(dimension=(0.01, 0.012, 0.008), polarization=(0, 0, 0))
    part.susceptibility = 50.0
    return part


def _cells(model):
    return solve_cells(mesh_uniform(model, 3))


def _fem(model):
    pytest.importorskip("ngsolve")
    from magpylib_material_response.fem import solve_fem  # noqa: PLC0415

    return solve_fem(model, order=2)


SOLVERS = [_cells, lambda model: solve_face_charge(model, level=1), _fem]
IDS = ["cells", "face charge", "fem"]


@pytest.mark.parametrize("solve", SOLVERS, ids=IDS)
def test_it_drives_a_part_as_h_ext_does_and_shows_in_get_b(solve):
    with_field = magpy.Collection(_soft(), UniformField(B=B0))
    with_attribute = magpy.Collection(_soft())
    with_attribute.H_ext = tuple(B0)
    new, old = solve(with_field), solve(with_attribute)
    scale = np.abs(magpy.getB(old, POINTS)).max()
    atol = 1e-3 * scale  # fem's mesh is its own at each solve
    np.testing.assert_allclose(
        magpy.getB(new, POINTS) - B0, magpy.getB(old, POINTS), atol=atol
    )
    np.testing.assert_allclose(
        magpy.getH(new, POINTS) - B0 / magpy.mu_0,
        magpy.getH(old, POINTS),
        atol=atol / magpy.mu_0,
    )


@pytest.mark.parametrize("solve", SOLVERS, ids=IDS)
def test_several_fields_add_up_also_turned_and_nested(solve):
    turned = UniformField(B=(0, 0, 0.1))
    turned.rotate_from_angax(90, "x")  # (0, -0.1, 0) in the global frame
    other = UniformField(B=(0.05, 0, 0))
    several = magpy.Collection(_soft(), turned, magpy.Collection(other))
    one = magpy.Collection(_soft(), UniformField(B=(0.05, -0.1, 0)))
    new, reference = solve(several), solve(one)
    scale = np.abs(magpy.getB(reference, POINTS)).max()
    np.testing.assert_allclose(
        magpy.getB(new, POINTS), magpy.getB(reference, POINTS), atol=1e-3 * scale
    )


@pytest.mark.parametrize("solve", SOLVERS, ids=IDS)
@pytest.mark.parametrize("above", [False, True], ids=["on it", "above it"])
def test_a_field_and_h_ext_together_are_refused(solve, above):
    model = magpy.Collection(_soft(), UniformField(B=B0))
    holder = magpy.Collection(model) if above else model
    holder.H_ext = tuple(B0)
    with pytest.raises(ValueError, match="count twice"):
        solve(model)


def test_it_turns_and_takes_h_and_is_kept_through_copy_and_json():
    field = UniformField(H=(0, 0, 1000.0))
    np.testing.assert_allclose(field.B, (0, 0, 1000 * magpy.mu_0))
    field.rotate_from_angax(90, "x")
    np.testing.assert_allclose(
        field.getB((5, 6, 7)), (0, -1000 * magpy.mu_0, 0), atol=1e-12
    )
    np.testing.assert_allclose(field.getJ((1, 1, 1)), 0)
    np.testing.assert_allclose(
        field.copy().getB((1, 1, 1)), field.getB((1, 1, 1)), atol=1e-15
    )
    back = from_json(to_json(field))[0]
    np.testing.assert_allclose(back.getB((1, 1, 1)), field.getB((1, 1, 1)), atol=1e-15)


@pytest.mark.parametrize(
    "bad", [{}, {"B": B0, "H": B0}, {"B": (1, 2)}, {"B": (0, 0, np.nan)}]
)
def test_a_bad_field_is_refused(bad):
    with pytest.raises(ValueError, match="UniformField"):
        UniformField(**bad)


def test_its_value_can_be_set_and_given_to_a_copy():
    field = UniformField(B=(0, 0, 0.1))
    field.B = (0, 0, 0.2)
    np.testing.assert_allclose(field.getB((1, 1, 1)), (0, 0, 0.2))
    field.H = (0, 0, 1000.0)
    np.testing.assert_allclose(field.B, (0, 0, 1000 * magpy.mu_0))
    other = field.copy(B=(0.3, 0, 0))
    np.testing.assert_allclose(other.getB((1, 1, 1)), (0.3, 0, 0), atol=1e-15)
    np.testing.assert_allclose(field.B, (0, 0, 1000 * magpy.mu_0))
    with pytest.raises(ValueError, match="three finite"):
        field.B = (1, 2)


@pytest.mark.parametrize(
    "duplicate",
    [
        copy.copy,
        copy.deepcopy,
        lambda f: pickle.loads(pickle.dumps(f)),
        lambda f: f.copy(),
    ],
    ids=["copy.copy", "deepcopy", "pickle", ".copy"],
)
def test_every_copy_reads_its_own_field_and_keeps_it_read_only(duplicate):
    field = UniformField(B=(0, 0, 0.1))
    other = duplicate(field)
    other.B = (0, 0, 0.5)
    np.testing.assert_allclose(other.getB((1, 1, 1)), (0, 0, 0.5))
    np.testing.assert_allclose(field.getB((1, 1, 1)), (0, 0, 0.1))
    with pytest.raises(ValueError, match="read-only"):
        other.B[0] = 1.0


def test_a_field_in_a_unit_of_h_warns():
    with pytest.warns(UserWarning, match="H in A/m") as caught:
        UniformField(B=(0, 0, 1e5))
    assert caught[0].filename == __file__
    UniformField(H=(0, 0, 1e7))  # 12.6 T, and clearly H: no warning


def test_meshers_and_reuse_keep_it():
    model = magpy.Collection(_soft(), UniformField(B=B0))
    meshed = mesh_all(model, 27)
    assert sum(isinstance(s, UniformField) for s in meshed.sources_all) == 1
    # with the susceptibility on the parent, as with a coil
    part = magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0))
    parent = magpy.Collection(part, UniformField(B=B0))
    parent.susceptibility = 50.0
    assert (
        sum(isinstance(s, UniformField) for s in mesh_uniform(parent, 2).sources_all)
        == 1
    )
    first = solve_face_charge(magpy.Collection(_soft(), UniformField(B=B0)), reuse=True)
    again = solve_face_charge(
        magpy.Collection(_soft(), UniformField(B=2 * B0)), reuse=first
    )
    assert again.face_charge_info["reused"]
    np.testing.assert_allclose(
        magpy.getB(again, POINTS) - 2 * B0,
        2 * (magpy.getB(first, POINTS) - B0),
        rtol=1e-9,
    )


def test_h_ext_warns_that_it_goes():
    part = _soft()
    model = magpy.Collection(mesh_Cuboid(part, (2, 2, 2)))
    model.H_ext = tuple(B0)
    with pytest.warns(DeprecationWarning, match="UniformField"):
        solve_cells(model)
