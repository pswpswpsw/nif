"""Execute released native tutorials in clean kernels and a fresh result directory."""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path
from uuid import uuid4

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tutorial_catalog import tutorial_entries  # noqa: E402


def select_tutorials(root=ROOT, numbers=None):
    released = tutorial_entries(root, status="released")
    if numbers is None:
        return released
    try:
        requested = [int(part.strip()) for part in numbers.split(",")]
    except ValueError as exc:
        raise ValueError(
            "--numbers must contain comma-separated released tutorial ids"
        ) from exc
    by_id = {entry["id"]: entry for entry in released}
    if len(requested) != len(set(requested)) or any(
        number not in by_id for number in requested
    ):
        raise ValueError(
            "Select each released tutorial at most once; experimental tutorials run separately"
        )
    return [by_id[number] for number in requested]


def new_output(root, mode, output=None):
    if output is None:
        name = (
            mode
            + "-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid4().hex[:8]
        )
        output = Path(root) / "dev_tests/artifacts/native-notebooks" / name
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("The output directory must be new or empty: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    return output


def run(root=ROOT, numbers=None, full=False, output=None):
    """Run selected released cases; retain failed notebooks and restore caller env."""
    root = Path(root).resolve()
    entries = select_tutorials(root, numbers)
    mode = "full" if full else "smoke"
    artifact_root = new_output(root, mode, output)
    updates = dict(
        NIF_SMOKE="0" if full else "1",
        NIF_OUTPUT_DIR=str(artifact_root),
        MPLBACKEND="Agg",
    )
    previous = {name: os.environ.get(name) for name in updates}
    summary = []
    try:
        os.environ.update(updates)
        for entry in entries:
            path = root / entry["notebook"]
            notebook = nbformat.read(path, as_version=4)
            source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            started = time.monotonic()
            try:
                NotebookClient(
                    notebook,
                    timeout=7200,
                    kernel_name="python3",
                    resources={"metadata": {"path": str(root)}},
                ).execute()
                result = dict(
                    notebook=path.name,
                    status="passed",
                    seconds=time.monotonic() - started,
                )
            except Exception as exc:
                result = dict(
                    notebook=path.name,
                    status="failed",
                    seconds=time.monotonic() - started,
                    error=str(exc),
                )
                (artifact_root / (path.stem + ".error.txt")).write_text(
                    str(exc), encoding="utf-8"
                )
            nbformat.write(notebook, artifact_root / path.name)
            result["source_sha256"] = source_sha256
            result["output_directory"] = str(artifact_root)
            summary.append(result)
            print(json.dumps(result), flush=True)
            (artifact_root / "execution_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return artifact_root, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full scientific workloads instead of SMOKE",
    )
    parser.add_argument(
        "--numbers",
        help="Comma-separated released tutorial ids; default: all released cases",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New or empty result directory; default: a unique timestamped directory",
    )
    args = parser.parse_args(argv)
    try:
        _, summary = run(numbers=args.numbers, full=args.full, output=args.output)
    except ValueError as exc:
        parser.error(str(exc))
    return int(any(result["status"] == "failed" for result in summary))


if __name__ == "__main__":
    raise SystemExit(main())
