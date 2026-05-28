"""Unit test: M01 short summary returns the right keys + plausible values."""
import pytest

import hpvsim as hpv

from tests.regression.anchor_m01 import make_sim
from tests.regression.short_summary_m01 import build_summary_m01, METRIC_KEYS_M01


def test_build_summary_m01_keys_and_shape():
    sim = make_sim()
    sim.run()
    out = build_summary_m01(sim)
    assert set(out.keys()) == set(METRIC_KEYS_M01)
    assert out['total HPV infections'] >= 0
    assert 0.0 <= out['mean HPV prevalence (%)'] <= 100.0
    assert out['total population'] > 0
