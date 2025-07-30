"""Sphinx configuration for autoformalize-math-lab documentation."""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# -- Project information -----------------------------------------------------

project = "autoformalize-math-lab"
copyright = "2025, Autoformalize Team"
author = "Autoformalize Team"

# The full version, including alpha/beta/rc tags
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "myst_parser",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_rtd_theme",
    "sphinx.ext.githubpages",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    ".pytest_cache",
    "__pycache__",
]

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": None,
    ".md": "myst_parser",
    ".ipynb": "nbsphinx",
}

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom CSS
html_css_files = [
    "custom.css",
]

# Custom JS
html_js_files = [
    "custom.js",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for autodoc ----------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

autodoc_typehints = "description"
autodoc_preserve_defaults = True

# -- Options for autosummary ------------------------------------------------

autosummary_generate = True
autosummary_generate_overwrite = True

# -- Options for Napoleon (Google/NumPy style docstrings) -------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "transformers": ("https://huggingface.co/docs/transformers/", None),
}

# -- Options for MathJax -----------------------------------------------------

mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
        "processEnvironments": True,
        "tags": "ams",
        "macros": {
            "RR": "\\mathbb{R}",
            "NN": "\\mathbb{N}",
            "ZZ": "\\mathbb{Z}",
            "QQ": "\\mathbb{Q}",
            "CC": "\\mathbb{C}",
            "lean": "\\text{Lean}",
            "isabelle": "\\text{Isabelle/HOL}",
            "coq": "\\text{Coq}",
        },
    },
    "options": {
        "ignoreHtmlClass": "tex2jax_ignore",
        "processHtmlClass": "tex2jax_process",
    },
}

# -- Options for MyST parser ------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

myst_heading_anchors = 3
myst_footnote_transition = True
myst_dmath_double_inline = True

# -- Options for nbsphinx ---------------------------------------------------

nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True
nbsphinx_kernel_name = "python3"

# Custom notebook cell timeout
nbsphinx_timeout = 300

# -- Options for copybutton -------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True

# -- Custom configuration ---------------------------------------------------

# Logo
html_logo = "_static/logo.png"  # Add logo if available
html_favicon = "_static/favicon.ico"  # Add favicon if available

# Footer
html_last_updated_fmt = "%b %d, %Y"
html_show_sourcelink = True
html_show_sphinx = True

# Search
html_search_language = "en"

# Custom roles
def setup(app):
    """Set up custom Sphinx configuration."""
    app.add_css_file("custom.css")
    app.add_js_file("custom.js")
    
    # Custom roles for mathematical concepts
    app.add_role("lean", lambda name, rawtext, text, lineno, inliner, options={}, content=[]: 
                 ([], []))
    app.add_role("isabelle", lambda name, rawtext, text, lineno, inliner, options={}, content=[]: 
                 ([], []))
    app.add_role("coq", lambda name, rawtext, text, lineno, inliner, options={}, content=[]: 
                 ([], []))

# -- Options for LaTeX output -----------------------------------------------

latex_engine = "pdflatex"
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{mathtools}
\usepackage{unicode-math}
""",
}

# Grouping the document tree into LaTeX files. List of tuples
latex_documents = [
    (master_doc, "autoformalize-math-lab.tex", "autoformalize-math-lab Documentation",
     "Autoformalize Team", "manual"),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
man_pages = [
    (master_doc, "autoformalize-math-lab", "autoformalize-math-lab Documentation",
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
texinfo_documents = [
    (master_doc, "autoformalize-math-lab", "autoformalize-math-lab Documentation",
     author, "autoformalize-math-lab", "Automated Mathematical Formalization",
     "Miscellaneous"),
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]