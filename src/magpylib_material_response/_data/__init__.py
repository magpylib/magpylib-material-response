"""
Built-in datasets for demonstration, educational and test purposes.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path


def get_dataset(name):
    name = Path(name).with_suffix("").with_suffix(".json")
    with (
        importlib.resources.path(
            "magpylib_material_response.package_data", "datasets"
        ) as resources_path,
        (resources_path / name).open() as fp,
    ):
        return json.load(fp)
