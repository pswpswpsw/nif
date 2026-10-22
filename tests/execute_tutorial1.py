"""Execute every cell of a NIF tutorial in an isolated working directory.

--smoke applies tutorial-1-specific reductions only to the executed QA copy.
Other tutorials use this runner at their full settings without --smoke.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(args.notebook, as_version=4)
    if args.smoke:
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.source = cell.source.replace("nepoch = 5000", "nepoch = 2")
                cell.source = cell.source.replace("ckpt-4999/ckpt", "ckpt-1/ckpt")
                cell.source = cell.source.replace(
                    "rounds=200, max_iter=1000", "rounds=1, max_iter=2"
                )
    kernel_root = args.workdir / "kernels"
    kernel_dir = kernel_root / "nif-colab-qa"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "NIF Colab QA",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    manager = KernelManager(
        kernel_name="nif-colab-qa",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]),
    )
    events = []
    start = time.monotonic()

    def cell_start(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            lines = cell.source.splitlines()
            first_line = lines[0] if lines else "<empty>"
            print(f"START cell {cell_index}: {first_line[:90]}", flush=True)

    def cell_done(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            events.append(
                {
                    "cell_index": cell_index,
                    "execution_count": cell.execution_count,
                    "elapsed_seconds": round(time.monotonic() - start, 2),
                }
            )
            print(
                f"DONE cell {cell_index} at {events[-1]['elapsed_seconds']}s",
                flush=True,
            )
            nbformat.write(notebook, args.output)
            args.output.with_suffix(".events.json").write_text(
                json.dumps(events, indent=2), encoding="utf-8"
            )

    client = NotebookClient(
        notebook,
        km=manager,
        timeout=7200,
        startup_timeout=120,
        allow_errors=False,
        resources={"metadata": {"path": str(args.workdir)}},
        on_cell_start=cell_start,
        on_cell_executed=cell_done,
    )
    try:
        client.execute()
        code_cells = [
            c for c in notebook.cells if c.cell_type == "code" and c.source.strip()
        ]
        assert all(c.execution_count is not None for c in code_cells)
        assert not any(
            o.output_type == "error" for c in code_cells for o in c.get("outputs", [])
        )
        print(
            f"PASS: all {len(code_cells)} code cells executed ({'SMOKE' if args.smoke else 'FULL'}).",
            flush=True,
        )
    finally:
        nbformat.write(notebook, args.output)
        args.output.with_suffix(".events.json").write_text(
            json.dumps(events, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
