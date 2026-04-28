"""Tests for the v2 -> v3 regression harness.

Smoke test for the anchor sim (this file) plus unit tests for the drift
computation in tests/regression/compare.py (added in Task 2).
"""

import sys
from pathlib import Path

# tests/ is on sys.path when pytest is invoked from tests/, but be robust:
sys.path.insert(0, str(Path(__file__).parent))

from regression.anchor import run_and_summarize  # noqa: E402


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