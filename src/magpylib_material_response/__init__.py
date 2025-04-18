"""
Copyright (c) 2025 Alexandre Boisselet. All rights reserved.

magpylib-material-response: Python package extending the Magpylib library by providing magnetic field analysis for soft materials and demagnetization of hard magnets.
"""

from __future__ import annotations

from magpylib_material_response._data import get_dataset

from ._version import version as __version__

__all__ = ["__version__", "get_dataset"]
