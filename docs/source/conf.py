"""Build the current NIF API with the original Read the Docs theme."""
import ast
import os
import re
import sys
from pathlib import Path


ROOT = Path(os.environ.get("NIF_DOCS_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
about = ast.parse((ROOT / "nif/__about__.py").read_text(encoding="utf-8"))
release = next(
    ast.literal_eval(node.value)
    for node in about.body
    if isinstance(node, ast.Assign)
    and any(isinstance(item, ast.Name) and item.id == "__version__" for item in node.targets)
)
version = release
project = "NIF"
author = "Shaowu Pan, Steven Brunton, J. Nathan Kutz"
copyright = "2026, " + author
language = "en"
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_templates", "__pycache__"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
napoleon_use_keyword = False
myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 6
html_theme = "sphinx_rtd_theme"
html_title = f"NIF {release} — PyTorch documentation"
html_static_path = ["_static"]
html_theme_options = {
    "style_nav_header_background": "#2f5470",
    "collapse_navigation": False,
    "navigation_depth": 3,
}
html_baseurl = "https://www.shaowupan.com/nif/"
htmlhelp_basename = "NIF"


def stable_optimizer_signature(app, what, name, obj, options, signature, return_annotation):
    """Display the actual omitted-epsilon sentinel without a process address."""
    if name in {"nif.optimizers.AdaBeliefOptimizer", "nif.optimizers.L4Adam"} and signature:
        signature = re.sub(r"eps=<object object(?: at 0x[0-9A-Fa-f]+)?>", "eps=_UNSET", signature)
    return signature, return_annotation


def setup(app):
    app.connect("autodoc-process-signature", stable_optimizer_signature)
    app.connect("autodoc-process-docstring", normalization_docstrings)


def normalization_docstrings(app, what, name, obj, options, lines):
    """Correct inherited normalization prose in this documentation build only."""
    standard = """Normalize raw columns using population means and standard deviations.

The transform is ``normalized = (raw_data - offset) / scale``. Ordinary
columns use the raw mean as offset and population standard deviation
(``ddof=0``) as scale. With ``area_weighted=True``, the last column contains
weights: its offset is zero and its scale is the raw mean weight.

Returns
-------
tuple
    ``(data, offset, scale)`` without weights; otherwise
    ``(data_without_weights, offset, scale, normalized_weights)``.
    The offset and scale vectors still include the weight column.

Zero or nonfinite scales are rejected. The historical return names ``mean``
and ``std`` denote these raw-data offset and scale vectors.
"""
    minmax = """Normalize parameter/coordinate ranges and target amplitudes.

The transform is ``normalized = (raw_data - offset) / scale``. Parameter and
coordinate columns use the raw midpoint ``(min + max) / 2`` and half-range
``(max - min) / 2``. Target columns use the raw mean as offset and
``max(abs(raw_target))`` as scale. With ``area_weighted=True``, the last column
contains weights: its offset is zero and its scale is the raw mean weight.

Returns
-------
tuple
    ``(data, offset, scale)`` without weights; otherwise
    ``(data_without_weights, offset, scale, normalized_weights)``.
    The offset and scale vectors still include the weight column.

Zero or nonfinite scales are rejected. The historical return names ``mean``
and ``std`` denote these raw-data offset and scale vectors, not statistics
recomputed from normalized data.
"""
    replacements = {"standard_normalize": standard, "minmax_normalize": minmax}
    for prefix in ("nif.data.PointWiseData.", "nif.data.point_wise_data.PointWiseData."):
        if name.startswith(prefix) and name[len(prefix):] in replacements:
            lines[:] = replacements[name[len(prefix):]].splitlines()
            return
    if name in {"nif.demo.CylinderFlow", "nif.demo.cylinderflow.CylinderFlow"}:
        lines[:] = [
            "Load bundled cylinder-flow data: time, two coordinates, two velocity",
            "components and sample weights. Offset and scale follow",
            "``PointWiseData.minmax_normalize``; normalized weights are stored",
            "separately in ``sample_weight``.",
        ]
