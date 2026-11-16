"""Read the immutable pre-cleanup public tree without importing its old code."""
import hashlib
import json
import re
import zipfile
from pathlib import Path
from pathlib import PurePosixPath


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST_SHA256 = "392facc286eb33caf90ec16ead2052ea1ae4aaa4c41bf009635e94cbace1f7c3"
BACKING_RELOCATIONS = {
    "tutorial/" + name: "tutorial/legacy/tensorflow/" + name
    for name in (
        "1_simple_1d_wave.ipynb",
        "1_simple_1d_wave_colab.ipynb",
        "2_multi_scale_NIF.ipynb",
        "2_multi_scale_NIF_colab.ipynb",
        "3_multi_scale_linear_NIF.ipynb",
        "3_multi_scale_linear_NIF_colab.ipynb",
        "4_get_gradients_by_wrapping_model_with_layer.ipynb",
        "4_get_gradients_by_wrapping_model_with_layer_colab.ipynb",
        "5_large_scale_training_on_tensorflow_record_data.ipynb",
        "5_large_scale_training_on_tensorflow_record_data_colab.ipynb",
        "6_revisit_multi_scale_NIF_with_L1_L2_regularization.ipynb",
        "6_revisit_multi_scale_NIF_with_L1_L2_regularization_colab.ipynb",
        "7_model_pruning_and_quantization.ipynb",
        "7_model_pruning_and_quantization_colab.ipynb",
        "8_revisit_multi_scale_NIF_with_sobolov_training.ipynb",
        "8_revisit_multi_scale_NIF_with_sobolov_training_colab.ipynb",
    )
}


def relative_name(name):
    """Accept only portable repository-relative logical paths."""
    name = str(name)
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or path.is_absolute()
        or any(part in ("", ".", "..", ".git") for part in name.split("/"))
    ):
        raise ValueError("Expected a safe relative POSIX archive path: " + name)
    return path.as_posix()


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Snapshot:
    """Verified text objects plus hash-checked unchanged scientific assets.

    Large original notebooks, plots and datasets are backing assets rather than
    duplicate ZIP payloads. Every mutable source and historical test is stored in
    the bundle, so current implementation changes cannot alter the old audit.
    """

    def __init__(self, root=ROOT, manifest_path=None, bundle_path=None):
        self.root = Path(root).resolve()
        self.manifest_path = Path(manifest_path or HERE / "baseline-73b2ffc.json")
        raw = self.manifest_path.read_bytes()
        if digest(raw) != MANIFEST_SHA256:
            raise ValueError(
                "Historical manifest digest differs from its recorded value"
            )
        self.manifest = json.loads(raw)
        if self.manifest["format_version"] != 1:
            raise ValueError("Unsupported historical archive format")
        self.bundle_path = Path(bundle_path or HERE / self.manifest["bundle"])
        if digest(self.bundle_path.read_bytes()) != self.manifest["bundle_sha256"]:
            raise ValueError("Historical bundle digest differs from its recorded value")
        expected = set()
        for name, entry in self.manifest["files"].items():
            relative_name(name)
            if not re.fullmatch("[0-9a-f]{64}", entry["sha256"]):
                raise ValueError("Invalid historical content digest")
            if entry["storage"] not in ("bundle", "backing"):
                raise ValueError("Invalid historical storage kind")
            if entry["storage"] == "bundle":
                expected.add("sha256/" + entry["sha256"])
        self._objects = {}
        with zipfile.ZipFile(self.bundle_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != expected:
                raise ValueError("Historical bundle member inventory differs")
            for name in names:
                data = archive.read(name)
                if digest(data) != name.removeprefix("sha256/"):
                    raise ValueError("Historical content object digest differs")
                self._objects[name.removeprefix("sha256/")] = data

    def read(self, name):
        name = relative_name(name)
        entry = self.manifest["files"][name]
        if entry["storage"] == "bundle":
            data = self._objects[entry["sha256"]]
        else:
            source = self.root / relative_name(BACKING_RELOCATIONS.get(name, name))
            if not source.resolve().is_relative_to(self.root):
                raise ValueError("Historical backing path escapes repository")
            data = source.read_bytes()
            if (
                digest(data) != entry["sha256"]
                and entry.get("checkout_eol") == "lf-or-crlf"
            ):
                without_pairs = data.replace(b"\r\n", b"")
                if (
                    b"\r\n" in data
                    and b"\r" not in without_pairs
                    and b"\n" not in without_pairs
                ):
                    data = data.replace(b"\r\n", b"\n")
            if digest(data) != entry["sha256"]:
                raise ValueError("Historical backing digest differs: " + name)
        if len(data) != entry["size"]:
            raise ValueError("Historical content size differs: " + name)
        return data

    def materialize(self, destination, names=None):
        destination = Path(destination).resolve()
        selected = list(self.manifest["files"] if names is None else names)
        selected = [relative_name(name) for name in selected]
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError("Historical projection destination must be empty")
        destination.mkdir(parents=True, exist_ok=True)
        for name in selected:
            target = destination / name
            if not target.resolve().is_relative_to(destination):
                raise ValueError("Historical projection path escapes destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.read(name))
        return destination
