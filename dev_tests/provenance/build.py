"""Rebuild the frozen public baseline from its immutable Git commit."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


BASE = "73b2ffcf71d9f0d019bd86d62cbf80989e9f7eac"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = {".png", ".gif", ".npz", ".woff", ".woff2", ".ttf", ".eot"}


def build():
    files, objects = {}, {}
    tree = subprocess.check_output(["git", "ls-tree", "-r", "-z", BASE], cwd=ROOT)
    with subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    ) as reader:
        for record in tree.split(b"\0"):
            if not record:
                continue
            metadata, _, name = record.partition(b"\t")
            _, kind, object_id = metadata.split()
            if kind != b"blob":
                raise ValueError("Historical tree contains a non-file object")
            name = name.decode("utf-8")
            reader.stdin.write(object_id + b"\n")
            reader.stdin.flush()
            identity, blob_kind, size = reader.stdout.readline().split()
            if identity != object_id or blob_kind != b"blob":
                raise ValueError("Git did not return the requested historical blob")
            data = reader.stdout.read(int(size))
            if len(data) != int(size) or reader.stdout.read(1) != b"\n":
                raise ValueError("Truncated historical Git blob")
            digest = hashlib.sha256(data).hexdigest()
            path = Path(name)
            protected = path.suffix in ASSETS or (
                name.startswith("tutorial/")
                and name.endswith(".ipynb")
                and not name.endswith("_pytorch.ipynb")
            )
            storage = "backing" if protected else "bundle"
            files[name] = dict(sha256=digest, size=len(data), storage=storage)
            if name.startswith("tutorial/") and path.suffix == ".ipynb" and protected:
                files[name]["checkout_eol"] = "lf-or-crlf"
            if not protected:
                objects[digest] = data
        reader.stdin.close()
        if reader.wait():
            raise RuntimeError("Historical Git blob reader failed")
    bundle = HERE / "baseline-73b2ffc.zip"
    with zipfile.ZipFile(
        bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for digest, data in sorted(objects.items()):
            entry = zipfile.ZipInfo("sha256/" + digest, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, data, compresslevel=9)
    archived = [entry for entry in files.values() if entry["storage"] == "bundle"]
    manifest = dict(
        format_version=1,
        base_commit=BASE,
        bundle=bundle.name,
        bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        files=files,
        archived_logical_files=len(archived),
        content_objects=len(objects),
        deduplicated_bytes=sum(entry["size"] for entry in archived)
        - sum(map(len, objects.values())),
        protected_backing_files=sum(
            entry["storage"] == "backing" for entry in files.values()
        ),
    )
    target = HERE / "baseline-73b2ffc.json"
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "bundle_bytes": bundle.stat().st_size,
                **{key: value for key, value in manifest.items() if key != "files"},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    build()
