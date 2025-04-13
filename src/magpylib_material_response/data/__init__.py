"""
Built-in datasets for demonstration, educational and test purposes.
"""

from __future__ import annotations


def get_dataset(name):
    import importlib
    import json
    from pathlib import Path

    name = Path(name).with_suffix("").with_suffix(".json")
    with (
        importlib.resources.path(
            "magpylib_material_response.package_data", "datasets"
        ) as resources_path,
        (resources_path / name).open() as fp,
    ):
        return json.load(fp)
