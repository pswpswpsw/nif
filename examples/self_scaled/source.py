"""Fetch a byte-verified, pinned research optimizer without changing SciPy."""
import hashlib
import importlib.util
import os
import sys
import tempfile
import urllib.request
from pathlib import Path


AUTHOR_COMMIT = "990a82c1353cbda9f5afee3200b222fdb821c521"
AUTHOR_SHA256 = "65958e6bba33631ca5aeab9ecc309536199c08a7e380fc0487d404e93ae2687e"
AUTHOR_BYTES = 160175
AUTHOR_URL = (
    "https://raw.githubusercontent.com/EliKiani/Optimizing_the_Optimizer_PINNs/"
    + AUTHOR_COMMIT
    + "/_optimize.py"
)
SCIPY_VERSION = "1.16.3"


def verify_source(content):
    """Validate the original raw bytes before they can be imported."""
    digest = hashlib.sha256(content).hexdigest()
    if len(content) != AUTHOR_BYTES or digest != AUTHOR_SHA256:
        raise ValueError(
            "Pinned author source SHA256/length mismatch; source was not executed"
        )
    return digest


def fetch_source(cache_dir=None):
    """Download once into a versioned cache, rejecting altered cached bytes."""
    root = (
        Path(cache_dir)
        if cache_dir is not None
        else Path(tempfile.gettempdir()) / "nif-self-scaled"
    )
    path = root / AUTHOR_COMMIT / "_optimize.py"
    if path.exists():
        verify_source(path.read_bytes())
        return path
    request = urllib.request.Request(
        AUTHOR_URL, headers={"User-Agent": "NIF-self-scaled-tutorial/1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read(AUTHOR_BYTES + 1)
    verify_source(content)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix="verified-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def load_author(cache_dir=None):
    """Import a separate module; the installed scipy.optimize stays unchanged."""
    import scipy

    if scipy.__version__ != SCIPY_VERSION:
        raise RuntimeError(
            "The experimental author adapter requires scipy=="
            + SCIPY_VERSION
            + "; use examples/self_scaled/requirements.txt in a separate environment."
        )
    path = fetch_source(cache_dir)
    content = path.read_bytes()
    verify_source(content)
    module_name = "scipy.optimize._nif_self_scaled_" + AUTHOR_COMMIT
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        # Compile the exact verified bytes. SourceFileLoader may otherwise use
        # timestamp-valid .pyc bytes that were never checked against this hash.
        exec(compile(content, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def provenance():
    return {
        "repository": "EliKiani/Optimizing_the_Optimizer_PINNs",
        "commit": AUTHOR_COMMIT,
        "url": AUTHOR_URL,
        "sha256": AUTHOR_SHA256,
        "bytes": AUTHOR_BYTES,
        "scipy": SCIPY_VERSION,
        "distribution": "Runtime download; author modifications are not redistributed in this repository",
    }
