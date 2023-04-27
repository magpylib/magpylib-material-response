#!/usr/bin/env python
"""The setup script."""
from setuptools import find_packages
from setuptools import setup

with open("magpylib_material_response/__init__.py") as handle:
    for line in handle:
        if "__version__" in line:
            version = line.split(" = ")[-1].replace('"', "")
            break

with open("./README.md") as handle:
    readme_text = handle.read()

with open("./requirements.txt") as handle:
    requirements = [lr.strip() for lr in handle.read().splitlines() if lr.strip()]

with open("./requirements_dev.txt") as handle:
    requirements_dev = [lv.strip() for lv in handle.read().splitlines() if lv.strip()]

with open("./requirements_doc.txt") as handle:
    requirements_doc = [ld.strip() for ld in handle.read().splitlines() if ld.strip()]

_short_description = (
    "An extension to the Magpylib library, providing magnetic field analysis "
    "for soft materials and demagnetization of hard magnets."
)
setup(
    name="magpylib-material-response",
    version=version,
    description=_short_description,
    long_description=readme_text,
    long_description_content_type="text/markdown",
    author="Alexandre Boisselet",
    author_email="magpylib@gmail.com",
    url=("https://github.com/" "magpylib/magpylib-material-response"),
    license="MIT",
    packages=find_packages(),
    # include anything specified in Manifest.in
    include_package_data=True,
    package_data={"": ["package_data/datasets/*.json"]},
    install_requires=requirements,
    extras_require={
        "code_style": ["flake8", "black", "pre-commit"],
        "testing": requirements_dev,
        "docs": requirements_doc,
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="",
)
