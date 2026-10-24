"""Runnable phase-0 oracle/provenance checks, without importing a future solver."""
import ast
import hashlib
import json
from pathlib import Path
import struct

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'dev_tests/fixtures/lbfgs_reference_phase0'
ORACLE = json.loads((FIXTURES / 'oracle.json').read_bytes())


def case(name, dtype):
    return next(c for c in ORACLE['cases'] if c['case'] == name and c['dtype'] == dtype)


def test_frozen_public_inputs_and_pending_recipe_are_byte_exact():
    record = json.loads((FIXTURES / 'provenance.json').read_bytes())
    for path, expected in record['preserved_inputs'].items():
        data = (ROOT / path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected['sha256'], path
        assert len(data) == expected['bytes'], path
    for path, expected in record['frozen_phase0_inputs'].items():
        data = (ROOT / path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected['sha256'], path
        assert len(data) == expected['bytes'], path
    recipe = FIXTURES / 'test_reference_lbfgs_contract.py.txt'
    ast.parse(recipe.read_text(encoding='utf-8'))
    assert recipe.name.endswith('.py.txt')


def test_recorded_cpu_environment_and_original_options():
    assert ORACLE['cuda_visible_devices'] == '-1'
    assert ORACLE['physical_gpus'] == ORACLE['logical_gpus'] == []
    assert ORACLE['threads_intra'] == ORACLE['threads_inter'] == 1
    assert ORACLE['configured_outer_rounds'] == 200
    assert ORACLE['calls_per_simple_case'] == 1
    assert ORACLE['options_read_from_original_wrapper'] == dict(
        num_correction_pairs=20, tolerance=1e-15, x_tolerance=1e-15,
        f_relative_tolerance=1e-15, parallel_iterations=1,
        max_iterations=1000, max_line_search_iterations=100)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_recorded_raw_tensor_bytes_and_mathematical_state(dtype):
    def validate(value):
        if isinstance(value, dict) and 'raw_hex' in value:
            fmt = '<' + ('f' if value['dtype'] == 'float32' else 'd') * len(value['values'])
            unpacked = struct.unpack(fmt, bytes.fromhex(value['raw_hex']))
            for raw, number in zip(unpacked, value['values']):
                if isinstance(number, str):
                    assert str(raw) == {'+inf': 'inf', '-inf': '-inf', 'nan': 'nan'}[number]
                else:
                    assert raw == number
        elif isinstance(value, dict):
            for child in value.values():
                validate(child)
        elif isinstance(value, list):
            for child in value:
                validate(child)
    for item in ORACLE['cases']:
        if item['dtype'] == dtype:
            validate(item)
    for name in ('unit_quadratic', 'small_quadratic'):
        result = case(name, dtype)['result']
        assert result['converged'] and not result['failed']
        assert abs(result['objective_gradient']['values'][0]) <= 1e-15
    flat = case('flat_offset_quartic', dtype)['result']
    assert flat['converged'] and not flat['failed']
    assert flat['objective_gradient']['values'][0] > 1e-15
    assert flat['objective_value']['values'] == [2. ** 20]


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_recorded_resume_flags_history_and_cumulative_counts(dtype):
    item = case('cumulative_resume', dtype)
    assert item['first']['num_iterations'] == 1
    assert item['resumed']['num_iterations'] == 2
    assert item['resumed'] == item['fresh_two']
    for result in (item['first'], item['resumed']):
        assert not result['converged'] and not result['failed']
        assert result['position_deltas']['shape'] == [20, 2]
        assert result['gradient_deltas']['shape'] == [20, 2]
    assert case('positive_relative_stop', dtype)['result']['converged']
    assert not case('negative_relative_continues', dtype)['result']['converged']


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_recorded_failure_and_negative_infinity_contract(dtype):
    for name in ('positive_infinity', 'nan_gradient'):
        result = case(name, dtype)['result']
        assert result['failed'] and not result['converged']
    negative = case('negative_infinity', dtype)['result']
    assert negative['converged'] and not negative['failed']
    linear = case('unbounded_linear', dtype)['result']
    assert linear['converged'] == (dtype == 'float32')
    assert linear['failed'] == (dtype == 'float64')


def test_recorded_four_independent_stopping_conditions_and_signed_relative_f():
    for row in ORACLE['convergence_boundaries']:
        expected = (abs(row['gradient']) <= row['gradient_tolerance']
            or abs(row['dx']) <= row['x_tolerance']
            or abs(row['new_f'] - row['old_f']) <= row['relative_tolerance'] * row['old_f']
            or abs(row['new_f'] - row['old_f']) <= row['absolute_tolerance'])
        assert row['converged'] == expected
    assert next(r for r in ORACLE['convergence_boundaries']
                if r['case'] == 'relative_negative')['converged'] is False
