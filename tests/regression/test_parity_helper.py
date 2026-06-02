"""Unit tests for the shared parity_gate helper."""
import math
import pytest

from tests.regression.parity import parity_gate


def _row(seed, **metrics):
    return {'_seed': seed, **metrics}


def test_parity_gate_pass_means_overlap():
    v2 = [_row(s, m=10.0 + 0.1 * s) for s in range(5)]
    v3 = [_row(s, m=10.0 + 0.1 * s) for s in range(5)]
    failures = parity_gate(v3, v2, z_threshold=3.0)
    assert failures == []


def test_parity_gate_fail_large_drift():
    v2 = [_row(s, m=10.0 + 0.1 * s) for s in range(10)]
    v3 = [_row(s, m=100.0 + 0.1 * s) for s in range(10)]
    failures = parity_gate(v3, v2, z_threshold=3.0)
    assert len(failures) == 1
    name, z = failures[0]
    assert name == 'm'
    assert abs(z) > 3.0


def test_parity_gate_skip_keys_ignored():
    v2 = [_row(s, m=10.0, _total_pop=1e6) for s in range(5)]
    v3 = [_row(s, m=10.0, _total_pop=2e6) for s in range(5)]
    failures = parity_gate(v3, v2, z_threshold=3.0,
                           skip_keys=frozenset({'_total_pop'}))
    assert failures == []


def test_parity_gate_degenerate_distributions_exact_match_passes():
    v2 = [_row(s, m=10.0) for s in range(5)]
    v3 = [_row(s, m=10.0) for s in range(5)]
    failures = parity_gate(v3, v2, z_threshold=3.0)
    assert failures == []


def test_parity_gate_degenerate_distributions_mismatch_fails():
    v2 = [_row(s, m=10.0) for s in range(5)]
    v3 = [_row(s, m=11.0) for s in range(5)]
    failures = parity_gate(v3, v2, z_threshold=3.0)
    assert len(failures) == 1
    assert failures[0][0] == 'm'
    assert math.isinf(failures[0][1])
