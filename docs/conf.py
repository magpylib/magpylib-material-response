from __future__ import annotations

import importlib.metadata
from typing import Any

project = "magpylib-material-response"
copyright = "2025, Alexandre Boisselet"
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

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
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
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

always_document_param_types = True

suppress_warnings = ["mystnb.unknown_mime_type"]

html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js",
]

# Static files (CSS/JS)
html_static_path = ["_static"]
html_css_files = ["fullwidth.css"]
