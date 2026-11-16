"""Render tutorial entry points from tutorial/catalog.json without changing notebooks."""
import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/pswpswpsw/nif"
COLAB = "https://colab.research.google.com/github/pswpswpsw/nif"
README_START = "## Google Colab Tutorial\n"
README_END = "## Issues, bugs, requests, ideas\n"


def validate_catalog(catalog, root=ROOT):
    """Reject ambiguous entries and paths before any generated file is written."""
    root = Path(root).resolve()
    if catalog.get("schema_version") != 1 or not catalog.get("tutorials"):
        raise ValueError("Expected a nonempty version-1 tutorial catalog")
    links = catalog.get("links", {})
    for key in (
        "readme_ref",
        "released_docs_ref",
        "experimental_docs_ref",
        "historical_docs_ref",
    ):
        value = links.get(key, "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value) or ".." in value:
            raise ValueError("Invalid source ref: " + key)
    ids, notebooks = [], set()
    for entry in catalog["tutorials"]:
        number = entry.get("id")
        if not isinstance(number, int) or number < 1 or number in ids:
            raise ValueError("Tutorial ids must be unique positive integers")
        ids.append(number)
        if entry.get("status") not in {"released", "experimental"}:
            raise ValueError("Unknown tutorial status")
        for key in ("title", "case", "concepts"):
            if (
                not isinstance(entry.get(key), str)
                or not entry[key].strip()
                or "\n" in entry[key]
            ):
                raise ValueError("Missing or multiline tutorial field: " + key)
        if not entry.get("goals") or not all(
            isinstance(goal, str) and goal.strip() and "\n" not in goal
            for goal in entry["goals"]
        ):
            raise ValueError("Tutorial goals must be nonempty single-line strings")
        keys = (
            ("notebook", "original", "compatibility")
            if entry["status"] == "released"
            else ("notebook",)
        )
        for key in keys:
            name = entry.get(key, "")
            path = Path(name)
            if (
                not isinstance(name, str)
                or "\\" in name
                or path.is_absolute()
                or ".." in path.parts
                or not name.startswith("tutorial/")
                or not (root / path).resolve().is_relative_to(root)
                or not (root / path).is_file()
            ):
                raise ValueError("Invalid or missing tutorial file: " + str(name))
        if entry["status"] == "released":
            for key in ("original", "compatibility"):
                published = entry.get("historical_" + key, "")
                if (
                    not isinstance(published, str)
                    or "\\" in published
                    or Path(published).parent != Path("tutorial")
                    or Path(published).name != Path(entry[key]).name
                    or not published.endswith(".ipynb")
                ):
                    raise ValueError(
                        "Invalid historical tutorial path: " + str(published)
                    )
        if entry["notebook"] in notebooks or not entry["notebook"].endswith(
            "_pytorch.ipynb"
        ):
            raise ValueError("Native notebook filenames must be unique")
        notebooks.add(entry["notebook"])
    if ids != sorted(ids):
        raise ValueError("Tutorials must be ordered by id")
    return catalog


def load_catalog(root=ROOT):
    root = Path(root)
    return validate_catalog(
        json.loads((root / "tutorial/catalog.json").read_text(encoding="utf-8")), root
    )


def tutorial_entries(root=ROOT, status=None):
    return [
        entry
        for entry in load_catalog(root)["tutorials"]
        if status is None or entry["status"] == status
    ]


def source_ref(catalog, entry, audience):
    if audience == "readme":
        return catalog["links"]["readme_ref"]
    if audience == "docs":
        return catalog["links"][entry["status"] + "_docs_ref"]
    raise ValueError("Unknown tutorial audience: " + audience)


def readme_outer_sections(text):
    start = text.index(README_START) + len(README_START)
    end = text.index(README_END, start)
    return text[:start], text[end:]


def replace_readme_section(text, listing):
    before, after = readme_outer_sections(text)
    return before + "\n" + listing.rstrip() + "\n\n" + after


def render_readme(catalog):
    lines = [
        "<!-- Generated from tutorial/catalog.json by scripts/tutorial_catalog.py. -->",
        "",
        "The notebooks below run the native PyTorch examples. Their scientific workloads "
        "are the default; set `NIF_SMOKE=1` before running only for a short development check.",
        "Google Colab runs these same PyTorch notebooks; the framework and platform are separate choices.",
        "",
    ]
    for entry in catalog["tutorials"]:
        url = (
            COLAB
            + "/blob/"
            + source_ref(catalog, entry, "readme")
            + "/"
            + entry["notebook"]
        )
        lines.append(
            f'{entry["id"]}. **{entry["title"]}** '
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"({url})"
        )
        lines.extend("    - " + goal for goal in entry["goals"])
        lines.append("")
    return "\n".join(lines)


def rst_table(catalog, entries, title):
    lines = [
        ".. list-table:: " + title,
        "   :header-rows: 1",
        "   :widths: 12 29 39 20",
        "   :class: nif-tutorial-table",
        "",
        "   * - Tutorial",
        "     - Scientific case",
        "     - Main implementation concepts",
        "     - Open",
    ]
    for entry in entries:
        suffix = "/blob/" + source_ref(catalog, entry, "docs") + "/" + entry["notebook"]
        lines.extend(
            [
                f'   * - {entry["id"]}',
                "     - " + entry["case"],
                "     - " + entry["concepts"],
                f"     - `Notebook <{REPOSITORY + suffix}>`__ / `Colab <{COLAB + suffix}>`__",
            ]
        )
    return "\n".join(lines) + "\n"


def render_rst(catalog):
    released = [
        entry for entry in catalog["tutorials"] if entry["status"] == "released"
    ]
    experimental = [
        entry for entry in catalog["tutorials"] if entry["status"] == "experimental"
    ]
    lines = [
        "Scientific tutorials",
        "====================",
        "",
        ".. Generated from tutorial/catalog.json; update with scripts/tutorial_catalog.py --write.",
        "",
        f"The {len(released)} released notebooks contain their scientific cases, "
        "training, evaluation and figures.",
        f'Open the immutable ``{catalog["links"]["released_docs_ref"]}`` versions below '
        "in GitHub or Google Colab.",
        "Full workloads are the default; ``NIF_SMOKE=1`` selects a short development check.",
        "",
        rst_table(catalog, released, "Released PyTorch tutorials"),
        "The :doc:`validation` and :doc:`comparison_records` pages retain the numerical",
        "results, user acceptance and evaluation-grid definitions.",
        "",
        "Experimental tutorials",
        "----------------------",
        "",
        "These post-release experiments extend the scientific cases. The",
        ":doc:`self_scaled_optimizers` guide gives the optimizer comparison's methods,",
        "budgets, measured results and execution receipts.",
        "",
        rst_table(catalog, experimental, "Experimental PyTorch tutorials"),
        "Original cases and compatibility notebooks",
        "------------------------------------------",
        "",
        "Original notebooks preserve the scientific definitions and historical outputs.",
        "Compatibility copies retain the pinned setup for the historical backend.",
        "Their current checkout location is ``tutorial/legacy/tensorflow/``; the links",
        "below keep their original paths in the immutable release.",
        "",
        ".. list-table:: Historical references",
        "   :header-rows: 1",
        "   :class: nif-tutorial-table",
        "",
        "   * - Tutorial",
        "     - Original notebook",
        "     - Compatibility notebook",
    ]
    for entry in released:
        prefix = REPOSITORY + "/blob/" + catalog["links"]["historical_docs_ref"] + "/"
        lines.extend(
            [
                f'   * - {entry["id"]}',
                f'     - `Original <{prefix + entry["historical_original"]}>`__',
                f'     - `Compatibility <{prefix + entry["historical_compatibility"]}>`__',
            ]
        )
    lines.extend(
        [
            "",
            "See the `original API documentation <legacy/tensorflow/index.html>`_ for the",
            "historical workflow. Use the native PyTorch links above for current execution.",
            "",
        ]
    )
    return "\n".join(lines)


def render_tutorial_readme(catalog):
    released = [
        entry for entry in catalog["tutorials"] if entry["status"] == "released"
    ]
    experimental = [
        entry for entry in catalog["tutorials"] if entry["status"] == "experimental"
    ]
    lines = [
        "# Native PyTorch tutorials",
        "",
        "<!-- Generated from catalog.json by scripts/tutorial_catalog.py. -->",
        "",
        f"The [catalog](catalog.json) lists {len(released)} released tutorials "
        f"and {len(experimental)} experimental tutorial.",
        "Use the [tutorial index](../docs/source/tutorials.rst) for scientific cases and",
        "versioned notebook links, or the [README Colab links](../README.md#google-colab-tutorial).",
        "",
        "Colab is a notebook execution platform. These same PyTorch notebooks run",
        "locally or in Colab; each notebook includes its own environment setup.",
        "",
        "Released notebooks run their full scientific workloads by default.",
        "`NIF_SMOKE=1` selects a short development check.",
        "",
    ]
    lines.extend(
        f'- [Tutorial {entry["id"]}: {entry["title"]}]({Path(entry["notebook"]).name}) '
        "extends the released cases; see the "
        "[experiment guide](../docs/SELF_SCALED_OPTIMIZERS.md)."
        for entry in experimental
    )
    lines.extend(
        [
            "",
            "The [TensorFlow archive](legacy/tensorflow/README.md) contains the eight",
            "original notebooks, their eight `_colab.ipynb` compatibility copies and",
            "historical setup files. See the [native API guide](../docs/PYTORCH_API.md)",
            "for current package usage.",
            "",
        ]
    )
    return "\n".join(lines)


def render_legacy_readme(catalog):
    lines = [
        "# Historical TensorFlow tutorials",
        "",
        "<!-- Generated from ../../catalog.json by scripts/tutorial_catalog.py. -->",
        "",
        "The current [PyTorch tutorials](../../README.md) run locally or in Google Colab.",
        "This archive preserves the original TensorFlow notebooks and the later",
        "Colab compatibility copies, including their scientific code, comments and outputs.",
        "",
        "| Tutorial | Original TensorFlow notebook | TensorFlow compatibility copy |",
        "| --- | --- | --- |",
    ]
    for entry in catalog["tutorials"]:
        if entry["status"] == "released":
            original = Path(entry["original"]).name
            compatibility = Path(entry["compatibility"]).name
            lines.append(
                f'| {entry["id"]} | [{original}]({original}) | '
                f"[{compatibility}]({compatibility}) |"
            )
    lines.extend(
        [
            "",
            "`_colab.ipynb` is a historical name for the TensorFlow dependency-compatibility",
            "copies. Colab is a platform; the current PyTorch notebooks also include Colab setup.",
            "",
            "The [compatibility guide](colab_support/README.md) records the pinned historical",
            "environment. These notebooks load their recorded older NIF implementation; use",
            "`*_pytorch.ipynb` in the parent tutorial index with the current package.",
            "",
            "The [catalog](../../catalog.json) records current archive paths separately from",
            "immutable `v2.0.0` links, where these files retain their original published paths.",
            "",
        ]
    )
    return "\n".join(lines)


def update_outputs(root=ROOT, check=False):
    root = Path(root)
    catalog = load_catalog(root)
    outputs = {
        "README.md": replace_readme_section(
            (root / "README.md").read_text(encoding="utf-8"), render_readme(catalog)
        ),
        "docs/source/tutorials.rst": render_rst(catalog),
        "tutorial/README.md": render_tutorial_readme(catalog),
        "tutorial/legacy/tensorflow/README.md": render_legacy_readme(catalog),
    }
    changed = []
    for name, expected in outputs.items():
        path = root / name
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            changed.append(name)
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(expected, encoding="utf-8", newline="\n")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    changed = update_outputs(args.root, check=args.check)
    if args.check and changed:
        print("Out-of-date tutorial entry points: " + ", ".join(changed))
        return 1
    print(
        "Tutorial entry points are current."
        if args.check
        else "Updated: " + (", ".join(changed) or "none")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
