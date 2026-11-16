"""Build PyTorch documentation and preserve the original static site as an archive.

Markdown pages use the repository's existing README and guides, so executable
snippets have one source. HTML never overwrites the tracked legacy site.
"""
import argparse
import ast
import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from pathlib import PurePosixPath
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
    "docs/SELF_SCALED_OPTIMIZERS.md": "self_scaled_optimizers.md",
}
README_IMAGES = ("myimage.gif", "compare.gif")
POST_RELEASE_SOURCES = (
    "examples/self_scaled",
    "dev_tests/provenance",
    "scripts/tutorial_catalog.py",
    "tutorial/catalog.json",
    "tutorial/legacy/tensorflow",
    "docs/legacy-assets.json",
    "tutorial/9_self_scaled_optimizers_pytorch.ipynb",
    "docs/SELF_SCALED_OPTIMIZERS.md",
)


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


def rewrite_markdown(text, source_name, tag, destination_name="", guides=None):
    """Resolve site links while leaving fenced executable examples verbatim."""
    pages = {**GUIDES, **(guides or {})}

    def source_ref(name):
        if name.startswith("docs/archive/"):
            return "main"
        # These experimental files were added after the immutable v2.0.0 tag.
        if any(
            name == prefix or name.startswith(prefix + "/")
            for prefix in POST_RELEASE_SOURCES
        ):
            return "main"
        return tag

    def target_for(match):
        target = match.group(1)
        for prefix in (
            REPOSITORY + "/blob/main/",
            REPOSITORY + "/tree/main/",
            "https://colab.research.google.com/github/pswpswpsw/nif/blob/main/",
        ):
            if target.startswith(prefix):
                name = unquote(urlsplit(target[len(prefix) :]).path)
                return (
                    "]("
                    + target.replace("/main/", "/" + source_ref(name) + "/", 1)
                    + ")"
                )
        parts = urlsplit(target)
        if parts.scheme or parts.netloc or not parts.path or target.startswith("/"):
            return match.group(0)
        name = posixpath.normpath(
            posixpath.join(posixpath.dirname(source_name), unquote(parts.path))
        )
        suffix = ("?" + parts.query if parts.query else "") + (
            "#" + parts.fragment if parts.fragment else ""
        )
        if name in pages:
            resolved = pages[name]
        elif name.startswith("docs/archive/") and name.endswith(".md"):
            resolved = name.removeprefix("docs/")
        elif name in {"misc/" + image for image in README_IMAGES}:
            resolved = "_static/readme/" + posixpath.basename(name)
        elif name.startswith("docs/source/_static/"):
            resolved = name.removeprefix("docs/source/")
        else:
            resolved = f"{REPOSITORY}/blob/{source_ref(name)}/" + quote(name, safe="/")
            return "](" + resolved + suffix + ")"
        if posixpath.dirname(destination_name):
            resolved = posixpath.relpath(resolved, posixpath.dirname(destination_name))
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
        (source / destination).write_bytes(text.encode("utf-8"))
    prepare_archives(root, source)
    images = source / "_static/readme"
    images.mkdir(parents=True, exist_ok=True)
    for name in README_IMAGES:
        shutil.copyfile(root / "misc" / name, images / name)
    return source


def checked_relative_path(name, label):
    """Reject platform-specific and escaping paths in public asset manifests."""
    if not isinstance(name, str):
        raise ValueError("Unsafe " + label + " path")
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in name
        or ":" in name
        or name != relative.as_posix()
        or not relative.parts
    ):
        raise ValueError("Unsafe " + label + " path: " + name)
    return relative


def prepare_archives(root, source):
    """Render dated Markdown snapshots without changing their retained bytes."""
    root, source = Path(root), Path(source)
    for path in sorted((root / "docs/archive").glob("*/manifest.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        prefix = path.parent.relative_to(root / "docs").as_posix()
        pages = {}
        verified = {}
        for name, entry in record["records"].items():
            checked_relative_path(name, "archive source")
            relative = checked_relative_path(entry["archive"], "archive asset")
            snapshot = path.parent / relative
            if not snapshot.resolve().is_relative_to(path.parent.resolve()):
                raise ValueError("Unsafe archive asset path: " + entry["archive"])
            data = snapshot.read_bytes()
            if (
                hashlib.sha256(data).hexdigest() != entry["sha256"]
                or len(data) != entry["bytes"]
            ):
                raise ValueError("Changed archived document: " + entry["archive"])
            if snapshot.suffix == ".md":
                destination = prefix + "/" + relative.as_posix()
                pages[name] = destination
                link_source = entry.get("link_source", name)
                checked_relative_path(link_source, "archive source")
                verified[name] = data, link_source
        for name, (data, link_source) in verified.items():
            destination = pages[name]
            current = posixpath.relpath("index.rst", posixpath.dirname(destination))
            notice = (
                "---\norphan: true\n---\n\n:::{note}\n"
                + "Historical snapshot recorded on "
                + record["snapshot_date"]
                + " at commit `"
                + record["source_commit"]
                + "`. See the [current documentation]("
                + current
                + ") for the maintained API and tutorial status.\n:::\n\n"
            )
            rendered = rewrite_markdown(
                data.decode("utf-8"),
                link_source,
                record["source_commit"],
                destination_name=destination,
                guides=pages,
            )
            target = source / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((notice + rendered).encode("utf-8"))
        index = path.parent / "README.md"
        if index.is_file() and "docs/" + prefix + "/README.md" not in pages:
            destination = prefix + "/README.md"
            rendered = rewrite_markdown(
                index.read_text(encoding="utf-8"),
                "docs/" + destination,
                record["source_commit"],
                destination_name=destination,
            )
            (source / destination).write_text(
                "---\norphan: true\n---\n\n" + rendered, encoding="utf-8"
            )


def copy_legacy_site(root, output):
    """Copy the frozen legacy site without importing later documentation assets."""
    docs = Path(root) / "docs"
    output = validate_output(root, output)
    manifest = json.loads((docs / "legacy-assets.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(
        manifest.get("files"), dict
    ):
        raise ValueError("Unsupported legacy asset manifest")
    verified = {}
    for name, entry in manifest["files"].items():
        checked_relative_path(name, "legacy asset")
        source = docs / name
        if not source.resolve().is_relative_to(docs.resolve()):
            raise ValueError("Unsafe legacy asset path: " + name)
        data = source.read_bytes()
        canonical = data
        if entry.get("text_lf") and b"\r" in data:
            # Accept a uniform Windows checkout, never mixed or lone CRs.
            if b"\r" in data.replace(b"\r\n", b"") or data.count(b"\n") != data.count(
                b"\r\n"
            ):
                raise ValueError("Changed legacy asset: " + name)
            canonical = data.replace(b"\r\n", b"\n")
        if (
            hashlib.sha256(canonical).hexdigest() != entry["sha256"]
            or len(canonical) != entry["bytes"]
        ):
            raise ValueError("Changed legacy asset: " + name)
        verified[name] = data
    legacy = output / "legacy/tensorflow"
    # Reset only this generated namespace after verifying every original asset.
    # Resolve before deletion so a symlink cannot redirect cleanup outside output.
    if legacy.is_symlink() or not legacy.resolve().is_relative_to(output):
        raise ValueError("Legacy output escaped the build output directory")
    if legacy.exists():
        shutil.rmtree(legacy)
    legacy.mkdir(parents=True, exist_ok=True)
    for name, data in verified.items():
        target = legacy / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
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
