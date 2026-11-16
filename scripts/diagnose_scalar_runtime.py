"""Compare scalar reference backends on the same CPU, without changing NIF.

By default, report the runtime and run the two AdaBelief scalar/moment test
files first with automatic dispatch, then with NumPy's AVX512 power target
disabled. This diagnostic reports failures and returns zero after collecting
both phases; use --require-tests-pass to require both phases to pass.
"""
import argparse
import difflib
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = (
    "dev_tests/test_adabelief_scalar_dtype.py",
    "dev_tests/test_adabelief_moment_dtype.py",
)


def runtime_probe(reference_root=None):
    import numpy as np
    import torch
    from numpy._core import _multiarray_umath

    sys.path.insert(0, str(ROOT))
    if reference_root is not None:
        sys.path.insert(0, str(reference_root))
    from dev_tests import test_adabelief_scalar_dtype as scalar_reference
    from dev_tests.test_adabelief_scalar_dtype import coefficients

    def value_bits(value, scalar):
        value = scalar(value)
        integer = np.uint32 if scalar is np.float32 else np.uint64
        return {
            "value": float(value) if np.isfinite(value) else None,
            "bits": hex(int(value.view(integer))),
        }

    rows = []
    for scalar, dtype in ((np.float32, torch.float32), (np.float64, torch.float64)):
        for beta_kind, betas in (
            ("nominal", (0.9, 0.999)),
            ("legacy-f32", tuple(float(np.float32(b)) for b in (0.9, 0.999))),
        ):
            for step in (1, 5, 6, 7, 8, 12, 20, 24, 50):
                powers = []
                for beta in betas:
                    b, t = scalar(beta), scalar(step)
                    powers.append(
                        {
                            "ufunc": value_bits(np.power(b, t), scalar),
                            "scalar_operator": value_bits(b**t, scalar),
                            "torch": value_bits(
                                torch.tensor(float(b), dtype=dtype)
                                .pow(torch.tensor(float(t), dtype=dtype))
                                .item(),
                                scalar,
                            ),
                        }
                    )
                with np.errstate(invalid="ignore", divide="ignore"):
                    reference = coefficients(dtype, step, betas)
                rows.append(
                    {
                        "dtype": str(dtype),
                        "beta_kind": beta_kind,
                        "step": step,
                        "powers": powers,
                        "reference_coefficients": [
                            value_bits(v, scalar) for v in reference
                        ],
                    }
                )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "coefficient_reference_file": scalar_reference.__file__,
        "coefficient_reference_sha256": hashlib.sha256(
            Path(scalar_reference.__file__).read_bytes()
        ).hexdigest(),
        "numpy_cpu_features": {
            key: bool(value)
            for key, value in _multiarray_umath.__cpu_features__.items()
        },
        "numpy_dispatched_functions": np.lib.introspect.opt_func_info(
            func_name="power|sqrt"
        ),
        "environment": {
            key: os.environ.get(key)
            for key in ("NPY_DISABLE_CPU_FEATURES", "ATEN_CPU_CAPABILITY")
        },
        "primitive_mismatch_rows": [
            row
            for row in rows
            if any(
                p["ufunc"]["bits"] != p["scalar_operator"]["bits"]
                for p in row["powers"]
            )
        ],
        "coefficient_rows": rows,
    }


def captured(command, environment, cwd=ROOT):
    result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.decode("utf-8", errors="backslashreplace"),
        "stderr": result.stderr.decode("utf-8", errors="backslashreplace"),
    }


def scalar_operator_copy(destination):
    package = destination / "dev_tests"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    original = (ROOT / TESTS[0]).read_bytes()
    old = b"power1, power2 = np.power(b1, t), np.power(b2, t)"
    new = b"power1, power2 = b1**t, b2**t"
    assert original.count(old) == 1, "Expected one exact power-primitive line"
    candidate = original.replace(old, new, 1)
    (destination / TESTS[0]).write_bytes(candidate)
    (destination / TESTS[1]).write_bytes((ROOT / TESTS[1]).read_bytes())
    return {
        "reference_root": str(destination),
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate).hexdigest(),
        "moment_test_file_unchanged": True,
        "diff": "".join(
            difflib.unified_diff(
                original.decode("utf-8").splitlines(keepends=True),
                candidate.decode("utf-8").splitlines(keepends=True),
                fromfile=TESTS[0],
                tofile="candidate/" + TESTS[0],
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe", action="store_true", help="Print runtime/coefficient JSON only"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--probe-only", action="store_true", help="Collect both runtimes without pytest"
    )
    parser.add_argument("--require-tests-pass", action="store_true")
    parser.add_argument("--reference-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(runtime_probe(args.reference_root), indent=2, allow_nan=False))
        return
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    temporary = tempfile.TemporaryDirectory(prefix="nif-scalar-reference-")
    candidate_root = (
        args.output_dir.resolve() / "scalar_operator_candidate"
        if args.output_dir is not None
        else Path(temporary.name)
    )
    reference_patch = scalar_operator_copy(candidate_root)
    phases = []
    for name, disabled, reference_root in (
        ("automatic", None, None),
        ("numpy-scalar-power", "AVX512_SKX", None),
        ("numpy-scalar-operators", None, candidate_root),
    ):
        environment = os.environ.copy()
        environment.pop("NPY_DISABLE_CPU_FEATURES", None)
        if disabled:
            environment["NPY_DISABLE_CPU_FEATURES"] = disabled
        probe_command = [sys.executable, "-B", str(Path(__file__).resolve()), "--probe"]
        if reference_root is not None:
            probe_command += ["--reference-root", str(reference_root)]
            environment["PYTHONPATH"] = (
                str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
            )
        probe = captured(probe_command, environment)
        if probe["returncode"]:
            raise RuntimeError(probe["stdout"] + probe["stderr"])
        runtime = json.loads(probe["stdout"])
        result = (
            None
            if args.probe_only
            else captured(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "pytest",
                    *TESTS,
                    "-q",
                    "-p",
                    "no:cacheprovider",
                ],
                environment,
                cwd=reference_root or ROOT,
            )
        )
        phase = {
            "phase": name,
            "runtime": runtime,
            "tests": result,
            "probe_stderr": probe["stderr"],
        }
        if reference_root is not None:
            phase["reference_patch"] = reference_patch
        phases.append(phase)
        if args.output_dir is not None:
            (args.output_dir / (name + ".json")).write_text(
                json.dumps(phase, indent=2), encoding="utf-8"
            )
        print(
            json.dumps(
                {
                    "phase": name,
                    "python": runtime["python"],
                    "numpy": runtime["numpy"],
                    "torch": runtime["torch"],
                    "torch_cpu_capability": runtime["torch_cpu_capability"],
                    "numpy_dispatched_functions": runtime["numpy_dispatched_functions"],
                    "primitive_mismatch_rows": runtime["primitive_mismatch_rows"],
                    "test_returncode": None if result is None else result["returncode"],
                    "reference_patch": phase.get("reference_patch"),
                },
                indent=2,
            )
        )
        if result is not None:
            print(result["stdout"])
            print(result["stderr"])
    temporary.cleanup()
    if args.require_tests_pass and any(
        p["tests"] and p["tests"]["returncode"] for p in phases
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
