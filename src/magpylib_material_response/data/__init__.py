"""
Built-in datasets for demonstration, educational and test purposes.
"""

from __future__ import annotations

def get_dataset(name):
    import importlib
    import json
    from pathlib import Path

    name = Path(name).with_suffix("").with_suffix(".json")
    with importlib.resources.path(
        "magpylib_material_response.package_data", "datasets"
    ) as resources_path:
        with open(resources_path / name) as fp:
            sim = json.load(fp)
    return sim
