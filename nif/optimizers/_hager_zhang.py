# Copyright 2018 The TensorFlow Probability Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See TFP_LICENSE.txt and TFP_NOTICE.txt in this directory.
# Modified 2026: scalar, eager PyTorch adaptation; no TensorFlow dependency.
"""Scalar Hager--Zhang search, adapted from TFP 0.25's public implementation.

This preserves its bracket/secant-square/bisection predicates, including the
unbounded internal bisection (terminated by representable-point stagnation).
Only scalar real f32/f64 searches are supported; no batched or differentiable
optimizer interface is provided. Backend arithmetic is not bitwise identical.
"""
from dataclasses import dataclass

import torch


@dataclass
class Point:
    x: torch.Tensor
    f: torch.Tensor
    df: torch.Tensor
    gradient: torch.Tensor


@dataclass
class SearchResult:
    left: Point
    right: Point
    converged: bool = False
    failed: bool = False
    iterations: int = 0
    evaluations: int = 0


def divide_no_nan(numerator, denominator):
    if bool(denominator == 0):
        return torch.zeros_like(numerator)
    return numerator / denominator


def _finite(point):
    return bool(torch.isfinite(point.f) & torch.isfinite(point.df))


def _negative_inf(value):
    return bool(value == -torch.inf)


def _bad_nan(point):
    return bool(
        torch.isnan(point.f) | (torch.isfinite(point.f) & torch.isnan(point.df))
    )


def _rising(point):
    return bool(torch.isfinite(point.f) & (point.df >= 0))


def _needs_bisect(point, limit):
    return bool((point.f == torch.inf) | ((point.df < 0) & (point.f > limit)))


def _close(a, b):
    # Literal TFP predicate, including equality and reversed arguments.
    return bool(torch.nextafter(a, b) >= b)


def _secant(a, b):
    return divide_no_nan(a.x * b.df - b.x * a.df, b.df - a.df)


def _wolfe(zero, point, limit):
    curvature = point.df >= 0.9 * zero.df
    exact = (0.1 * zero.df >= divide_no_nan(point.f - zero.f, point.x)) & curvature
    approximate = (point.f <= limit) & ((2 * 0.1 - 1) * zero.df >= point.df) & curvature
    return bool(exact | approximate)


def _bisect(evaluate, left, right, limit, result):
    """TFP U3; these evaluations do not increment the search iteration count."""
    while True:
        mid = evaluate((left.x + right.x) / 2)
        result.evaluations += 1
        finished = _negative_inf(mid.f)
        if not finished and (
            _bad_nan(mid) or bool(mid.x == left.x) or bool(mid.x == right.x)
        ):
            result.failed = True
            return left, right
        if finished:
            return mid, mid
        if bool((mid.df < 0) & (mid.f <= limit)):
            left = mid
        else:
            right = mid
        if _rising(right):
            return left, right


def _update(evaluate, left, right, trial, limit, result):
    if not bool((left.x < trial.x) & (trial.x < right.x)):
        return left, right
    valid_left = bool((trial.df < 0) & (trial.f <= limit))
    needs_bisect = bool((trial.df < 0) & (trial.f > limit))
    if valid_left:
        left = trial
    else:
        right = trial
    if needs_bisect:
        left, right = _bisect(evaluate, left, right, limit, result)
    return left, right


def _secant_square(evaluate, zero, limit, result):
    old_left, old_right = result.left, result.right
    trial = evaluate(_secant(old_left, old_right))
    result.evaluations += 1
    finished = _negative_inf(trial.f)
    if not finished and not _finite(trial):
        result.failed = True
        return
    if finished or _wolfe(zero, trial, limit):
        result.left = result.right = trial
        result.converged = True
        return
    left, right = _update(evaluate, old_left, old_right, trial, limit, result)
    # TFP's first update retains the prior endpoints on failure.
    if result.failed:
        return
    result.left, result.right = left, right
    updated_left = bool(left.x == trial.x)
    updated_right = bool(right.x == trial.x)
    next_x = trial.x
    if updated_left:
        next_x = _secant(old_left, left)
    if updated_right:
        next_x = _secant(old_right, right)
    if not bool((left.x <= next_x) & (next_x <= right.x)):
        return
    if updated_left or updated_right:
        trial = evaluate(next_x)
        result.evaluations += 1
    if not _finite(trial):
        result.failed = True
    elif _wolfe(zero, trial, limit):
        result.left = result.right = trial
        result.converged = True
    else:
        result.left, result.right = _update(evaluate, left, right, trial, limit, result)


def hager_zhang(evaluate, zero, max_iterations, *, inactive=False):
    """Return the literal final LEFT endpoint, including algorithmic failure.

    Even an inactive search evaluates its initial unit trial, as TFP does. The
    caller distinguishes a line-search failure from a raised closure exception.
    """
    initial = evaluate(zero.x.new_ones(()))
    result = SearchResult(
        zero, zero if inactive else initial, converged=inactive, evaluations=1
    )
    valid = (
        _finite(zero)
        and bool(zero.df < 0)
        and bool(torch.isfinite(initial.x) & (initial.x > 0))
    )
    result.failed = not inactive and not valid
    if inactive or result.failed:
        return result
    limit = zero.f + 1e-6 * zero.f.abs()
    left, right = zero, initial
    # Bracketing B0--B3. A finite-limit exit still passes through U3 when needed.
    while True:
        if _bad_nan(right):
            result.failed = True
            break
        if (
            _negative_inf(right.f)
            or _rising(right)
            or _needs_bisect(right, limit)
            or result.iterations >= max_iterations
        ):
            break
        left, right = right, evaluate(5.0 * right.x)
        result.evaluations += 1
        result.iterations += 1
    if _negative_inf(right.f):
        left = right
    else:
        if _needs_bisect(right, limit):
            left = zero
        # Literal reference transition: exhaustion alone does NOT mark U3
        # stopped. An unbracketed exhausted interval still bisects, possibly
        # ending with both `failed` and representable-endpoint `converged`.
        if not result.failed and not _rising(right):
            left, right = _bisect(evaluate, left, right, limit, result)
    result.left, result.right = left, right
    result.converged = _close(left.x, right.x)
    result.failed = result.failed or (
        not result.converged and result.iterations >= max_iterations
    )

    while (
        not result.converged
        and not result.failed
        and result.iterations < max_iterations
    ):
        old_left, old_right = result.left, result.right
        _secant_square(evaluate, zero, limit, result)
        result.iterations += 1
        if not result.converged and not result.failed:
            shrunk = bool(
                result.right.x - result.left.x < 0.66 * (old_right.x - old_left.x)
            )
            flat = _close(old_left.f, old_right.f) and _close(
                result.left.f, result.right.f
            )
            if shrunk and flat:
                result.converged = True
            elif not shrunk:
                mid = evaluate((result.left.x + result.right.x) / 2)
                result.evaluations += 1
                if not _finite(mid):
                    result.failed = True
                else:
                    result.left, result.right = _update(
                        evaluate, result.left, result.right, mid, limit, result
                    )
        if not result.failed and _close(result.left.x, result.right.x):
            result.converged = True
    return result
