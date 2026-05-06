"""Unit tests for the M03 40-entry short-summary builder."""
import numpy as np
import pytest

import hpvsim as hpv
from tests.regression.short_summary import build_summary, METRIC_KEYS


def test_metric_keys_are_eight():
    """Each genotype contributes exactly 8 metrics (matches M02 short_summary set)."""
    assert len(METRIC_KEYS) == 8


def test_build_summary_keys_are_40_for_four_genotypes():
    """4-genotype build = 32 per-genotype + 8 aggregate = 40 entries."""
    sim = hpv.Sim(
        n_agents=500, location='nigeria',
        start=1990, stop=1995, dt=1.0, rand_seed=0,
        genotypes=[16, 18, 'hi5', 'ohr'],
    )
    sim.run()
    out = build_summary(sim, genotypes=('hpv16', 'hpv18', 'hi5', 'ohr'))
    assert len(out) == 40
    for g in ('hpv16', 'hpv18', 'hi5', 'ohr'):
        for m in METRIC_KEYS:
            assert f'{g}.{m}' in out
    for m in METRIC_KEYS:
        assert f'any.{m}' in out


def test_build_summary_per_genotype_zero_safe():
    """Zero-cancer trajectory yields 0.0 (not NaN) for mean-age-of-cancer."""
    sim = hpv.Sim(
        n_agents=200, location='nigeria',
        start=1990, stop=1991, dt=1.0, rand_seed=0,
        genotypes=[16],
    )
    sim.run()
    out = build_summary(sim, genotypes=('hpv16',))
    # 8 + 8 = 16 entries (1 genotype + aggregate).
    assert len(out) == 16
    val = out['hpv16.mean age of cancer (years)']
    assert val == 0.0 or not np.isnan(val), \
        f'expected 0.0 or non-NaN; got {val}'