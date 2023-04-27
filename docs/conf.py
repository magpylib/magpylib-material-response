# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# Add package to the path so it can be imported without installing it
sys.path.append(os.path.join(os.path.split(__file__)[0], os.pardir))
from magpylib_material_response import __version__  # noqa: E402

# -- Project information -----------------------------------------------------

project = "magpylib-material-response"
copyright = "2023, Alexandre Boisselet"
author = "Alexandre Boisselet"

# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------

master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "magpylib-material-response.tex",
        "magpylib-material-response Documentation",
        "Alexandre Boisselet",
        "manual",
    )
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        master_doc,
        "magpylib-material-response",
        "magpylib-material-response Documentation",
        [author],
        1,
    )
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "magpylib-material-response",
        "magpylib-material-response Documentation",
        author,
        "magpylib-material-response",
        "Magpylib-Material-Response is an extension to the Magpylib library, providing magnetic "
        "field analysis for soft materials and demagnetization of hard magnets. "
        "Leveraging the Method of Moments, it calculates magnetic material response "
        "with high precision.",
        "Miscellaneous",
    )
]

suppress_warnings = ["mystnb.unknown_mime_type"]

html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js",
    # "https://unpkg.com/thebe@latest/lib/index.js",
]

mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"
mathjax2_config = {
    "tex2jax": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "processEscapes": True,
        "ignoreClass": "document",
        "processClass": "math|output_area",
    }
}

# needed for sphinx >= 4.3
mathjax_options = {
    "async": "async",
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    # "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
