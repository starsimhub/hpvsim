"""Verification tests: ss.parallel + hpv.Sim with correct RNG semantics.

Locks in four properties hpvsim relies on:
  1. Same rand_seed → identical short summaries (CRN guarantee).
  2. Different rand_seeds → plausibly distinct, plausibly distributed results.
  3. ss.parallel(inplace=True) (default) populates the original sims.
  4. copy_inputs=True (sim default) ensures shared module references are
     independent per sim post-construction — the [[feedback_sim_input_copy]]
     discipline reified as a regression test. Module state must be accessed
     via each sim's own copy (e.g. sim.get_module(name)), not via the
     original reference.
"""
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


def _short_summary(sim):
    """Tiny summary suitable for bit-equality assertions."""
    return {
        'cum_inf': float(sim.results.hpv16.cum_infections[-1]),
        'n_alive': float(sim.results['n_alive'][-1]),
    }


def test_parallel_seed_reproducibility():
    """Same rand_seed → identical short summary across two ss.parallel calls."""
    s1 = _tiny_sim(rand_seed=42, label='a')
    s2 = _tiny_sim(rand_seed=42, label='b')
    ss.parallel(s1, s2, verbose=0)
    assert _short_summary(s1) == _short_summary(s2)


def test_parallel_seed_separation():
    """Different rand_seeds → different short summaries (sanity)."""
    s1 = _tiny_sim(rand_seed=0, label='a')
    s2 = _tiny_sim(rand_seed=1, label='b')
    ss.parallel(s1, s2, verbose=0)
    a = _short_summary(s1)
    b = _short_summary(s2)
    # Pin: the two summaries must not be byte-identical. (Theoretically
    # could be equal by chance — astronomically unlikely for a 300-agent
    # 3-year run.)
    assert a != b


def test_parallel_inplace_true_default_mutates_originals():
    """ss.parallel(s1, s2) (default inplace=True) leaves s1/s2 with populated results."""
    s1 = _tiny_sim(rand_seed=0, label='a')
    s2 = _tiny_sim(rand_seed=1, label='b')
    ss.parallel(s1, s2, verbose=0)
    # Both originals carry results after the call.
    assert float(s1.results.hpv16.cum_infections[-1]) >= 0
    assert float(s2.results.hpv16.cum_infections[-1]) >= 0


def test_parallel_inplace_false_leaves_originals_unrun():
    """ss.parallel(inplace=False) preserves originals as unrun."""
    s1 = _tiny_sim(rand_seed=0, label='a')
    s2 = _tiny_sim(rand_seed=1, label='b')
    msim = ss.parallel(s1, s2, verbose=0, inplace=False)
    # The originals are not run; check that s1.ti is still None (uninitialized).
    assert s1.ti is None
    assert s2.ti is None
    # The MultiSim's internal copies DID run.
    assert float(msim.sims[0].results.hpv16.cum_infections[-1]) >= 0


def test_parallel_copy_inputs_independent_after_construction():
    """Shared module reference → independent per-sim copies after Sim construction.

    Memory: hpv.Sim deep-copies interventions/analyzers/diseases in __init__.
    After construction, each sim's diseases are independent, even when
    initialized from the same reference. ss.parallel may replace modules
    post-run with fresh copies (ss.parallel creates new sims internally);
    what matters is that the pre-run modules were distinct.
    """
    shared_hpv_list = [hpv.HPV(genotype='hpv16')]
    s1 = hpv.Sim(
        n_agents=300, location='nigeria', diseases=shared_hpv_list,
        start=2000, stop=2003, dt=0.25, rand_seed=0, verbose=0, label='a',
    )
    s2 = hpv.Sim(
        n_agents=300, location='nigeria', diseases=shared_hpv_list,
        start=2000, stop=2003, dt=0.25, rand_seed=1, verbose=0, label='b',
    )
    # After construction, each sim's module is its OWN copy (deep-copy discipline).
    s1_mod = s1.get_module('hpv16')
    s2_mod = s2.get_module('hpv16')
    assert s1_mod is not s2_mod, 'Each Sim must deep-copy its diseases'
    assert s1_mod is not shared_hpv_list[0], 'Sim must not reuse input module reference'
    assert s2_mod is not shared_hpv_list[0], 'Sim must not reuse input module reference'

    ss.parallel(s1, s2, verbose=0)
    # Post-run, each sim has final state without interference from the other.
    assert float(s1.results.hpv16.cum_infections[-1]) >= 0
    assert float(s2.results.hpv16.cum_infections[-1]) >= 0
    r1 = float(s1.results.hpv16.cum_infections[-1])
    r2 = float(s2.results.hpv16.cum_infections[-1])
    assert r1 != r2, 'Sims with different seeds must not contaminate each other'
