# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "sphinx_rtd_theme"
html_theme = "classic"
html_static_path = ["_static"]

master_doc = "index"

import sphinx_rtd_theme

# Add the project directory to the system path
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "NIF"
author = "Shaowu Pan, Steven Brunton, J. Nathan Kutz"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx_rtd_theme",
]
# Templates
templates_path = ["_templates"]

# Static files

# Theme settings
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    "style_nav_header_background": "#2f5470",
}

# Output file base name for HTML help builder
htmlhelp_basename = "NIF"

# Document formatting
napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
napoleon_use_keyword = False

# Exclude modules
exclude_patterns = []

# Syntax highlighting
highlight_language = "python"

# Configure LaTeX output
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "12pt",
    "preamble": "",
    "figure_align": "htbp",
}

# Document hierarchy
autodoc_member_order = "bysource"

# Exclude members
exclude_members = []

# Source suffix
source_suffix = [".rst", ".md"]


html_context = {"module": "nif.model", "module": "nif.layers"}
html_sidebars = {"**": ["localtoc.html", "sourcelink.html", "searchbox.html"]}
