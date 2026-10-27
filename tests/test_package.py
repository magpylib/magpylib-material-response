from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util

import pytest

import magpylib_material_response as m


def test_version() -> None:
    assert importlib.metadata.version("magpylib_material_response") == m.__version__


def test_fem_without_ngsolve_says_how_to_install() -> None:
    if importlib.util.find_spec("ngsolve") is not None:
        pytest.skip("NGSolve is installed")
    with pytest.raises(ImportError, match=r"magpylib-material-response\[fem\]"):
        importlib.import_module("magpylib_material_response.fem")
