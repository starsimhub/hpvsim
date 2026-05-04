"""Tests for the v2 -> v3 regression harness.

Smoke tests for the M00 4-genotype anchor (skipped until M03 multi-genotype
support lands) and the M01 1-genotype HPV16 anchor, plus unit tests for the
drift computation in tests/regression/compare.py and a gated drift test
that runs when the v2 baseline file is present.
"""

import sys
from pathlib import Path

import pytest

# tests/ is on sys.path when pytest is invoked from tests/, but be robust:
sys.path.insert(0, str(Path(__file__).parent))

from regression.anchor import run_and_summarize  # noqa: E402


@pytest.mark.skip(
    reason='Multi-genotype not yet ported to v3; restored in M03 when '
           'genotypes=[16, 18, hi5, ohr] is supported again.'
)
def test_anchor_runs():
    short, total_pop = run_and_summarize()
    expected_keys = {
        'total HPV infections',
        'total cancers',
        'total cancer deaths',
        'mean HPV prevalence (%)',
        'mean cancer incidence (per 100k)',
        'mean age of infection (years)',
        'mean age of cancer (years)',
        'mean age of cancer death (years)',
    }
    missing = expected_keys - set(short.keys())
    assert not missing, f'short_summary missing keys: {missing}'
    assert total_pop > 0, f'total_pop should be positive, got {total_pop}'


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
    # All summary values should be finite and non-negative for a clean run.
    for k, v in short.items():
        assert v >= 0 and v == v, f'summary value {k}={v} should be finite and non-negative'


# --- M01 anchor drift vs. v2 1-gt baseline (gated on baseline file) -----------

import json
from pathlib import Path


_BASELINE_HPV16 = Path(__file__).resolve().parent / 'regression_baselines' / 'anchor_hpv16.json'


@pytest.mark.skipif(not _BASELINE_HPV16.exists(),
                    reason='anchor_hpv16.json baseline not present (gitignored; '
                           'see tests/regression/README.md for generation procedure)')
def test_anchor_hpv16_drift():
    """Tier-3 informational drift test against v2 1-genotype baseline.
    10% relative threshold; does NOT fail the build (mirrors M00 drift)."""
    short, total_pop = run_anchor_hpv16()
    current_summary = {**{k: float(v) for k, v in short.items()},
                       'total population': total_pop}

    with open(_BASELINE_HPV16) as f:
        baseline = json.load(f)
    rows = compute_drift(baseline['summary'], current_summary, threshold=0.10)
    n_over = sum(1 for r in rows if r['over_threshold'])
    print(f'\nM01 anchor drift: {n_over}/{len(rows)} keys exceed 10% relative drift')
    for r in rows:
        rel = f"{r['rel_diff']*100:+.2f}%" if r['rel_diff'] is not None else 'n/a'
        flag = 'YES' if r['over_threshold'] else ''
        print(f'  {r["key"]:<40} {r["baseline"]:>12.4g} {r["current"]:>12.4g} {rel:>10} {flag}')
    # Informational only; no assertion.
