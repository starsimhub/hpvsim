"""Tests for the v2 -> v3 regression harness.

Smoke for the M01 1-genotype HPV16 anchor and unit tests for the drift
computation in tests/regression/compare.py.
"""

import sys
from pathlib import Path

# tests/ is on sys.path when pytest is invoked from tests/, but be robust:
sys.path.insert(0, str(Path(__file__).parent))


# --- Unit tests for tests/regression/compare.py:compute_drift -----------------

from regression.compare import compute_drift  # noqa: E402


def test_compute_drift_within_threshold():
    baseline = {'a': 100.0, 'b': 50.0}
    current  = {'a': 105.0, 'b': 49.0}
    rows = compute_drift(baseline, current, threshold=0.10)
    by_key = {r['key']: r for r in rows}
    assert by_key['a']['rel_diff'] == 0.05
    assert by_key['a']['over_threshold'] is False
    assert by_key['b']['rel_diff'] == -0.02
    assert by_key['b']['over_threshold'] is False


def test_compute_drift_over_threshold():
    baseline = {'a': 100.0}
    current  = {'a': 120.0}
    rows = compute_drift(baseline, current, threshold=0.10)
    assert rows[0]['rel_diff'] == 0.20
    assert rows[0]['over_threshold'] is True


def test_compute_drift_zero_baseline_flagged():
    baseline = {'a': 0.0}
    current  = {'a': 1.0}
    rows = compute_drift(baseline, current, threshold=0.10)
    assert rows[0]['rel_diff'] is None
    assert rows[0]['abs_diff'] == 1.0
    assert rows[0]['over_threshold'] is True


def test_compute_drift_skips_keys_missing_from_current():
    baseline = {'a': 1.0, 'b': 2.0}
    current  = {'a': 1.05}  # 'b' missing
    rows = compute_drift(baseline, current)
    keys = [r['key'] for r in rows]
    assert keys == ['a']


# --- M01 1-genotype HPV16 anchor smoke ----------------------------------------

from regression.anchor_hpv16 import run_and_summarize as run_anchor_hpv16  # noqa: E402


def test_anchor_hpv16_runs():
    """Tier-2 smoke: M01 anchor runs end-to-end with finite shaped summary.
    Runtime ~3 minutes; no baseline file required."""
    short, total_pop = run_anchor_hpv16()
    expected_keys = {
        'total HPV infections',
        'mean HPV prevalence (%)',
        'mean age of infection (years)',
    }
    missing = expected_keys - set(short.keys())
    assert not missing, f'short_summary missing keys: {missing}'
    assert total_pop > 0, f'total_pop should be positive, got {total_pop}'
    for k, v in short.items():
        assert v >= 0 and v == v, f'summary value {k}={v} should be finite and non-negative'