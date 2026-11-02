"""The flat names at the top, and the old ones, for one release."""

from __future__ import annotations

import importlib
import subprocess
import sys

import magpylib as magpy
import pytest

import magpylib_material_response as mmr


@pytest.mark.parametrize(
    ("module", "old", "new"),
    [
        ("demag", "apply_demag", "solve_cells"),
        ("face_charge", "apply_face_charge", "solve_face_charge"),
        ("convergence", "mesh_study", "study_cells"),
        ("fem", "fem_solve", "solve_fem"),
        ("fem", "fem_study", "study_fem"),
    ],
)
def test_old_names_warn_and_give_the_new_function(module, old, new):
    if module == "fem":
        pytest.importorskip("ngsolve")
    loaded = importlib.import_module(f"magpylib_material_response.{module}")
    with pytest.warns(DeprecationWarning, match=new):
        function = getattr(loaded, old)
    assert function is getattr(mmr, new)


@pytest.mark.parametrize(
    ("module", "old"),
    [
        ("demag", "apply_demag"),
        ("face_charge", "apply_face_charge"),
        ("convergence", "mesh_study"),
        ("fem", "fem_solve"),
    ],
)
def test_a_star_import_still_gives_the_old_names(module, old):
    if module == "fem":
        pytest.importorskip("ngsolve")
    namespace = {}
    with pytest.warns(DeprecationWarning):
        exec(f"from magpylib_material_response.{module} import *", namespace)  # noqa: S102
    assert old in namespace


def test_the_old_module_keeps_its_other_names():
    from magpylib_material_response import cells, demag  # noqa: PLC0415

    with pytest.warns(DeprecationWarning, match="cells.demag_tensor"):
        assert demag.demag_tensor is cells.demag_tensor
    with pytest.raises(AttributeError):
        demag.no_such_name  # noqa: B018


def test_importing_the_package_loads_no_solver():
    """The top level is lazy: it stays fast, and NGSolve optional."""
    code = (
        "import sys, magpylib_material_response as mmr;"
        "names = ('cells', 'face_charge', 'fem', 'convergence');"
        "loaded = [n for n in names if 'magpylib_material_response.' + n in sys.modules];"
        "assert not loaded, loaded; assert 'ngsolve' not in sys.modules;"
        "mmr.solve_cells; assert 'magpylib_material_response.cells' in sys.modules;"
        "assert 'ngsolve' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_without_ngsolve_the_package_can_be_listed_and_star_imported():
    code = (
        "import importlib.util as u, inspect;"
        "find = u.find_spec;"
        "u.find_spec = lambda name, *a: None if name == 'ngsolve' else find(name, *a);"
        "import magpylib_material_response as mmr;"
        "inspect.getmembers(mmr);"
        "namespace = {};"
        "exec('from magpylib_material_response import *', namespace);"
        "assert 'solve_cells' in namespace;\n"
        "try:\n    mmr.solve_fem\n"
        "except ImportError as error:\n    assert 'pip install' in str(error)\n"
        "else:\n    raise AssertionError('no install hint')"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_file_saved_with_the_old_name_reads_as_cells():
    magnet = magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 1))
    magnet.solved_by = "demag"
    assert mmr.from_json(mmr.to_json(magnet))[0].solved_by == "cells"


def test_an_unknown_name_is_an_attribute_error():
    with pytest.raises(AttributeError, match="no_such_name"):
        mmr.no_such_name  # noqa: B018
    assert {"solve_cells", "meshing"} <= set(dir(mmr))
