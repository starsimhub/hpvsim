"""Verification tests: ss.MultiSim works with hpv.Sim.

These tests confirm Starsim's MultiSim machinery composes correctly with
hpv.Sim. They pin down behavior
hpvsim depends on so future Starsim upgrades that change the contract
break here loudly.

Tiny n_agents and short dur keep each test under ~5 s.
"""
import numpy as np
import pytest

import starsim as ss
import hpvsim as hpv


def _tiny_sim(rand_seed=0, label=None):
    return hpv.Sim(
        n_agents=300,
        location='nigeria',
        genotypes=[16],
        start=2000,
        stop=2003,
        dt=0.25,
        rand_seed=rand_seed,
        verbose=0,
        label=label,
    )


def test_multisim_n_runs_distinct_seeds():
    """ss.MultiSim(sim, n_runs=N).run() produces N sims with distinct seeds."""
    sim = _tiny_sim(rand_seed=0)
    msim = ss.MultiSim(sim, n_runs=4).run(verbose=0)
    assert len(msim.sims) == 4
    seeds = [int(s.pars.rand_seed) for s in msim.sims]
    assert len(set(seeds)) == 4, f'Expected 4 distinct seeds, got {seeds}'
    # Every run produced finite total infections.
    for s in msim.sims:
        cum_inf = float(s.results.hpv16.cum_infections[-1])
        assert np.isfinite(cum_inf)
        assert cum_inf >= 0


def test_multisim_median_reduce_shape_and_nonneg():
    """msim.median() exposes median + 10/90 quantile arrays; non-negative for counts."""
    sim = _tiny_sim(rand_seed=0)
    msim = ss.MultiSim(sim, n_runs=5).run(verbose=0)
    pre_len = len(msim.sims[0].results.hpv16.cum_infections)
    msim.median()
    # Reduced results are flattened to top-level keys (hpv16_cum_infections).
    reduced = np.asarray(msim.results.hpv16_cum_infections)
    assert reduced.shape == (pre_len,)
    assert np.all(reduced >= 0), 'cum_infections must be non-negative post-median'
    low = np.asarray(msim.results.hpv16_cum_infections.low)
    high = np.asarray(msim.results.hpv16_cum_infections.high)
    assert low.shape == (pre_len,), f'Expected low.shape {(pre_len,)}, got {low.shape}'
    assert high.shape == (pre_len,), f'Expected high.shape {(pre_len,)}, got {high.shape}'
    assert np.all(low >= 0), 'q10 lower bound must be non-negative'
    assert np.all(high >= low), 'q90 must be >= q10 everywhere'


def test_multisim_mean_reduce_smoke():
    """msim.mean() functions (smoke test). Don't use on bounded metrics in production."""
    sim = _tiny_sim(rand_seed=0)
    msim = ss.MultiSim(sim, n_runs=5).run(verbose=0)
    pre_len = len(msim.sims[0].results.hpv16.cum_infections)
    msim.mean()
    reduced = np.asarray(msim.results.hpv16_cum_infections)
    assert reduced.shape == (pre_len,), f'Expected shape {(pre_len,)}, got {reduced.shape}'
    assert np.isfinite(reduced).all()
    assert hasattr(msim.results.hpv16_cum_infections, 'low'), 'mean() result must carry .low attribute'
    assert hasattr(msim.results.hpv16_cum_infections, 'high'), 'mean() result must carry .high attribute'


def test_multisim_labels_propagate():
    """Labels on input sims survive to msim.sims[i].label after run()."""
    sims = [_tiny_sim(rand_seed=s, label=f'rep-{s}') for s in range(3)]
    msim = ss.MultiSim(sims).run(verbose=0)
    assert [s.label for s in msim.sims] == ['rep-0', 'rep-1', 'rep-2']
