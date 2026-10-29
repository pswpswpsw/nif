"""Build PyTorch documentation and preserve the original static site as an archive.

Markdown pages use the repository's existing README and guides, so executable
snippets have one source. HTML never overwrites the tracked legacy site.
"""
import argparse
import ast
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote
from urllib.parse import unquote
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/pswpswpsw/nif"
GUIDES = {
    "README.md": "quickstart.md",
    "docs/PYTORCH_API.md": "api_guide.md",
    "docs/PYTORCH_VALIDATION.md": "validation.md",
    "docs/PYTORCH_TENSORFLOW_PARITY.md": "comparison_records.md",
    "docs/PYTORCH_TRAINING_EXAMPLE.md": "training_example.md",
}
README_IMAGES = ("myimage.gif", "compare.gif")


def package_version(root):
    tree = ast.parse((Path(root) / "nif/__about__.py").read_text(encoding="utf-8"))
    return next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(item, ast.Name) and item.id == "__version__"
            for item in node.targets
        )
    )


def markdown_sections(text):
    """Separate fenced examples without changing contents or line endings."""
    buffered = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence is None and marker:
            if buffered:
                yield False, "".join(buffered)
            buffered = [line]
            fence = marker.group(1)
        elif fence is not None:
            buffered.append(line)
            if re.match(r"^\s*" + re.escape(fence) + r"\s*$", line):
                yield True, "".join(buffered)
                buffered = []
                fence = None
        else:
            buffered.append(line)
    if buffered:
        yield fence is not None, "".join(buffered)


def rewrite_markdown(text, source_name, tag):
    """Resolve site links while leaving fenced executable examples verbatim."""

    def target_for(match):
        target = match.group(1)
        for prefix in (
            REPOSITORY + "/blob/main/",
            REPOSITORY + "/tree/main/",
            "https://colab.research.google.com/github/pswpswpsw/nif/blob/main/",
        ):
            if target.startswith(prefix):
                return "](" + target.replace("/main/", "/" + tag + "/", 1) + ")"
        parts = urlsplit(target)
        if parts.scheme or parts.netloc or not parts.path or target.startswith("/"):
            return match.group(0)
        name = posixpath.normpath(
            posixpath.join(posixpath.dirname(source_name), unquote(parts.path))
        )
        suffix = ("?" + parts.query if parts.query else "") + (
            "#" + parts.fragment if parts.fragment else ""
        )
        if name in GUIDES:
            resolved = GUIDES[name]
        elif name in {"misc/" + image for image in README_IMAGES}:
            resolved = "_static/readme/" + posixpath.basename(name)
        else:
            resolved = f"{REPOSITORY}/blob/{tag}/" + quote(name, safe="/")
        return "](" + resolved + suffix + ")"

    def prose(content):
        content = re.sub(
            r'<p align="center">\s*<img src="([^"]+)" alt="([^"]+)"\s*/>\s*</p>',
            lambda match: f"![{match.group(2)}]({match.group(1)})",
            content,
        )
        return re.sub(r"\]\(([^)]+)\)", target_for, content)

    return "".join(
        content if code else prose(content) for code, content in markdown_sections(text)
    )


def validate_output(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if (
        output == root
        or root.is_relative_to(output)
        or output.is_relative_to(root / "docs")
    ):
        raise ValueError(
            "Build output must not overwrite the original documentation or checkout"
        )
    return output


def prepare_source(root, source):
    root, source = Path(root), Path(source)
    shutil.copytree(
        root / "docs/source",
        source,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    tag = "v" + package_version(root)
    for name, destination in GUIDES.items():
        text = (root / name).read_bytes().decode("utf-8")
        text = rewrite_markdown(text, name, tag)
        if destination == "validation.md":
            text = (
                ":::{note}\nThis page retains the dated validation record. Historical branch names and "
                "status descriptions below refer to that recorded snapshot. The comparison records "
                "contain the later tutorial acceptance decisions.\n:::\n\n" + text
            )
        (source / destination).write_bytes(text.encode("utf-8"))
    images = source / "_static/readme"
    images.mkdir(parents=True, exist_ok=True)
    for name in README_IMAGES:
        shutil.copyfile(root / "misc" / name, images / name)
    return source


def copy_legacy_site(root, output):
    """Copy only the previously published HTML site and referenced assets."""
    docs = Path(root) / "docs"
    legacy = Path(output) / "legacy/tensorflow"
    legacy.mkdir(parents=True, exist_ok=True)
    for path in docs.iterdir():
        if path.is_file() and path.suffix in {".html", ".js", ".inv"}:
            shutil.copyfile(path, legacy / path.name)
    for name in ("_static", "_sources", "source/_static"):
        shutil.copytree(docs / name, legacy / name, dirs_exist_ok=True)
    return legacy


def build(root, output):
    root = Path(root).resolve()
    output = validate_output(root, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="nif-docs-", dir=output.parent
    ) as temporary:
        source = Path(temporary).resolve()
        if not source.is_relative_to(output.parent):
            raise ValueError("Temporary documentation source escaped its output parent")
        prepare_source(root, source)
        environment = dict(os.environ, NIF_DOCS_ROOT=str(root), MPLBACKEND="Agg")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-b",
                "html",
                "-W",
                "--keep-going",
                "-d",
                str(output.parent / "doctrees"),
                str(source),
                str(output),
            ],
            cwd=root,
            env=environment,
            check=True,
        )
    copy_legacy_site(root, output)
    (output / ".nojekyll").write_bytes(b"")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dev_tests/artifacts/docs_v2_20261028_build/html",
    )
    args = parser.parse_args()
    print("Built documentation:", build(args.root, args.output))


if __name__ == "__main__":
    main()
