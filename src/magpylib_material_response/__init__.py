"""
Copyright (c) 2025 Alexandre Boisselet. All rights reserved.

magpylib-material-response: Python package extending the Magpylib library by providing magnetic field analysis for soft materials and demagnetization of hard magnets.
"""

from __future__ import annotations

from magpylib_material_response._data import get_dataset
from magpylib_material_response.logging_config import configure_logging, disable_logging

from ._version import version as __version__

__all__ = ["__version__", "configure_logging", "disable_logging", "get_dataset"]
