from __future__ import annotations

import datetime
import importlib.metadata
import os
import re
from typing import Any

# Make plotly output self-contained HTML (text/html) instead of
# application/vnd.plotly.v1+json, which myst-nb cannot render with plotly>=6.
# This env var is inherited by the notebook execution kernel.
# See: https://github.com/executablebooks/MyST-NB/issues/667
os.environ["PLOTLY_RENDERER"] = "sphinx_gallery"

project = "magpylib-material-response"
copyright = f"{datetime.datetime.now(tz=datetime.UTC).year}, Alexandre Boisselet"
author = "Alexandre Boisselet"
version = release = importlib.metadata.version("magpylib_material_response")

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

# Benchmark cells in the examples (e.g. solver_performance) run for tens of
# seconds locally; give slower CI builders comfortable headroom over the
# myst-nb default of 30 s per cell.
nb_execution_timeout = 120

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
    # jupytext-paired notebooks (gitignored, created by IDE pairing) — the
    # MyST .md files are the single documentation source
    "**/*.ipynb",
]

html_theme = "pydata_sphinx_theme"

html_theme_options: dict[str, Any] = {
    # "announcement": announcement,
    "logo": {
        "text": "Magpylib-Material-Response",
        "image_light": "_static/images/magpylib_logo.png",
        "image_dark": "_static/images/magpylib_logo.png",
    },
    "header_links_before_dropdown": 4,
    "show_version_warning_banner": True,
    "navbar_align": "content",  # [left, content, right] For testing that the navbar items align properly
    "navbar_center": ["navbar-nav"],
    "check_switcher": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/magpylib/magpylib-material-response",
            "icon": "https://img.shields.io/github/stars/magpylib/magpylib-material-response?style=social",
            "type": "url",
        },
    ],
    "navigation_with_keys": False,
    "footer_start": ["copyright"],
    "footer_end": [],
    "use_edit_page_button": True,
    "navigation_depth": 3,
    "collapse_navigation": False,
}

html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise site
    "github_user": "magpylib",
    "github_repo": "magpylib-material-response",
    "github_version": "main",
    "doc_path": "docs/",
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

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "magpylib": ("https://magpylib.readthedocs.io/en/stable", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

# The API docstrings use NumPy-style type strings ("array_like, shape (n, 3)",
# "int, optional, default=8", "magpy.Collection", ...). Nitpicky mode tries to
# resolve every fragment as a class reference; silence those without masking
# genuinely broken cross-references to fully-qualified targets.
nitpick_ignore_regex = [
    ("py:class", r"[^.]*"),  # any dot-free fragment: array_like, optional, n, 3, ...
    ("py:class", r"default.*"),  # "default=0.5", "default 1.5"
    ("py:class", r"(np|magpy|magnet)\..*"),  # docstring shorthand aliases
    ("py:class", r".*\bobject\b.*"),  # free text like "magpylib.Collection object ..."
]

always_document_param_types = True

suppress_warnings = ["mystnb.unknown_mime_type"]

html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js",
]

# Static files (CSS/JS)
html_static_path = ["_static"]
html_css_files = ["fullwidth.css"]

# ── Fix MathJax on pages with plotly figures ────────────────────────────────
# The plotly "sphinx_gallery" renderer embeds MathJax *v2* synchronously in
# the page body (for TeX in chart labels). Sphinx's own MathJax v3 loads
# deferred in <head>, so on any page with a plotly figure the v2 script runs
# first, clobbers ``window.MathJax``, and v3 never typesets the page math —
# raw ``\(...\)`` delimiters appear. We use no TeX inside chart labels, so
# strip the injected v2 tags at build time; plotly is unaffected
# (``PlotlyConfig.MathJaxConfig = 'local'`` is set by the renderer itself).
_PLOTLY_MATHJAX2_SCRIPTS = re.compile(
    r'<script src="https://[^"]*/mathjax/2[^"]*"></script>'
    r"(\s*<script>if \(window\.MathJax.*?</script>)?",
    re.DOTALL,
)


def _strip_plotly_mathjax2(_app, _pagename, _templatename, context, _doctree):
    body = context.get("body")
    if body and "/mathjax/2" in body:
        context["body"] = _PLOTLY_MATHJAX2_SCRIPTS.sub("", body)


def setup(app):
    app.connect("html-page-context", _strip_plotly_mathjax2)
