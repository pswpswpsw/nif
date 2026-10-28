"""Preserve the native helper's existing positional call convention."""
import inspect

from nif.optimizers.lbfgs_V2 import lbfgs_minimize


def test_original_parallel_iterations_positional_slot_is_preserved():
    # Original public native signature ends ..., max_line_search_iterations,
    # parallel_iterations, **kwargs. Adding absolute-f must not move that slot.
    arguments = (
        inspect.signature(lbfgs_minimize)
        .bind(None, None, None, 50, 10, 1e-8, 0.0, 0.0, 50, 1)
        .arguments
    )
    assert arguments.get("parallel_iterations") == 1
    assert "f_absolute_tolerance" not in arguments
