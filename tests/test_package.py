from __future__ import annotations

import importlib.metadata

import magpylib_material_response as m


def test_version():
    assert importlib.metadata.version("magpylib_material_response") == m.__version__
