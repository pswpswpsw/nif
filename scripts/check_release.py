"""Check release identity and distribution contents without importing NIF."""
import argparse
import ast
import json
import re
import tarfile
import zipfile
from datetime import date
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/pswpswpsw/nif"
REQUIRED_PACKAGE_FILES = {
    "nif/__about__.py",
    "nif/model.py",
    "nif/demo/dataset/cylinderflow.npz",
    "nif/demo/dataset/traveling_wave.npz",
    "nif/demo/dataset/traveling_wave_high_freq.npz",
    "nif/optimizers/TFP_LICENSE.txt",
    "nif/optimizers/TFP_NOTICE.txt",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def citation_scalar(text, field):
    """Read required top-level scalar fields; cffconvert validates the full schema."""
    values = re.findall(r"^" + re.escape(field) + r":[ \t]*(.+)$", text, re.MULTILINE)
    require(len(values) == 1, f"CITATION.cff requires one top-level {field}")
    value = values[0].strip()
    if value[:1] in ("'", '"'):
        require(value[-1:] == value[:1], f"Invalid CITATION.cff {field}")
        value = value[1:-1]
    return value


def check_metadata(root=ROOT, tag=None):
    root = Path(root)
    source = ast.parse((root / "nif/__about__.py").read_text(encoding="utf-8"))
    versions = [
        ast.literal_eval(node.value)
        for node in source.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    ]
    require(len(versions) == 1, "Package must declare one __version__")
    version = versions[0]
    require(
        isinstance(version, str) and re.fullmatch(r"\d+\.\d+\.\d+", version),
        "Release version must use MAJOR.MINOR.PATCH",
    )
    if tag is not None:
        require(tag == "v" + version, "Release tag does not match package version")
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    require(
        citation_scalar(citation, "type") == "software",
        "Citation must identify software",
    )
    require(
        citation_scalar(citation, "version") == version, "Citation version mismatch"
    )
    released = citation_scalar(citation, "date-released")
    require(
        date.fromisoformat(released).isoformat() == released, "Invalid release date"
    )
    require(
        citation_scalar(citation, "repository-code") == REPOSITORY,
        "Citation repository mismatch",
    )
    require(
        re.fullmatch(r"10\.5281/zenodo\.\d+", citation_scalar(citation, "doi")),
        "Software DOI must identify a Zenodo record, not the paper",
    )
    zenodo = json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
    require(zenodo.get("version") == version, "Zenodo version mismatch")
    require(zenodo.get("upload_type") == "software", "Zenodo must identify software")
    require(zenodo.get("publication_date") == released, "Zenodo release date mismatch")
    require(
        zenodo.get("title") == citation_scalar(citation, "title"),
        "Zenodo title mismatch",
    )
    require(
        "doi" not in zenodo, "Do not reuse an existing DOI for a new Zenodo version"
    )
    return version


def check_members(names, required):
    names = set(names)
    require(
        not required - names,
        "Missing distribution files: " + ", ".join(sorted(required - names)),
    )
    forbidden = {"experiments", "artifacts", ".git", "__pycache__"}
    require(
        not any(forbidden.intersection(Path(name).parts) for name in names),
        "Distribution contains local/generated files",
    )


def check_distributions(directory, version):
    directory = Path(directory)
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    require(
        len(wheels) == len(sources) == 1, "Expected exactly one wheel and one sdist"
    )
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        check_members(names, REQUIRED_PACKAGE_FILES)
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        require(len(metadata_names) == 1, "Wheel must contain one package METADATA")
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
        require(metadata["Name"].lower() == "nif", "Wheel package name mismatch")
        require(metadata["Version"] == version, "Wheel version mismatch")
        dependencies = metadata.get_all("Requires-Dist", [])
        require(
            any(item.lower().startswith("torch") for item in dependencies),
            "Missing PyTorch dependency",
        )
        require(
            not any(
                re.match(r"(?:tensorflow|keras|tf-keras)\b", item, re.I)
                for item in dependencies
            ),
            "Wheel unexpectedly requires the historical backend",
        )
    with tarfile.open(sources[0], "r:gz") as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
        roots = {name.split("/", 1)[0] for name in names}
        require(len(roots) == 1, "Sdist must have one top-level package directory")
        stripped = [name.split("/", 1)[1] for name in names if "/" in name]
        check_members(
            stripped,
            REQUIRED_PACKAGE_FILES
            | {"CITATION.cff", ".zenodo.json", "LICENSE", "README.md", "PKG-INFO"},
        )
        metadata = BytesParser().parsebytes(
            archive.extractfile(next(iter(roots)) + "/PKG-INFO").read()
        )
        require(metadata["Version"] == version, "Sdist version mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tag", help="Release tag, including its v prefix")
    parser.add_argument(
        "--dist", type=Path, help="Also validate one wheel and one sdist"
    )
    args = parser.parse_args()
    try:
        version = check_metadata(args.root, args.tag)
        if args.dist is not None:
            check_distributions(args.dist, version)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Release validation failed: {error}\n")
    print(f"Validated NIF {version}" + (" and distributions" if args.dist else ""))


if __name__ == "__main__":
    main()
