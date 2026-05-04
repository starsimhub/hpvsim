"""Functional tests for hpvsim.hpv.HPV via a minimal ss.Sim."""

import numpy as np
import pytest
import starsim as ss

from hpvsim.hpv import HPV


def _minimal_sim(genotype='hpv16', n_agents=1000, init_prev=0.05, beta=0.3,
                 dur_years=2.0, n_steps=4):
    """Build a minimal Sim with HPV(genotype=...) and a stock random network.

    Returns ``(sim, hpv)`` where ``hpv`` is the live module reference inside
    the sim. ``ss.Sim`` deep-copies its pars by default, so we use
    ``copy_inputs=False`` to keep the original instance addressable from tests.
    """
    hpv = HPV(
        genotype=genotype,
        init_prev=ss.bernoulli(p=init_prev),
        beta=ss.peryear(beta),
        dur_precin=ss.lognorm_ex(mean=ss.years(dur_years)),
    )
    sim = ss.Sim(
        diseases=hpv,
        networks='random',
        n_agents=n_agents,
        dur=ss.years(n_steps * 0.5),
        dt=ss.years(0.5),
        verbose=0,
        copy_inputs=False,
    )
    return sim, hpv


def test_genotype_attribute_set():
    hpv = HPV(genotype='hpv16')
    assert hpv.genotype == 'hpv16'


def test_unknown_genotype_rejected():
    with pytest.raises(ValueError, match='hpv16'):
        HPV(genotype='hpv99')


def test_init_prev_seeds_initial_cases():
    """init_prev=0.05 + n_agents=1000 yields ~50 initial cases (Bernoulli +/-5 sigma)."""
    sim, hpv = _minimal_sim(init_prev=0.05, n_agents=1000)
    sim.init()
    n_initial = int(hpv.infected.sum())
    expected = 0.05 * 1000
    sigma = (0.05 * 0.95 * 1000) ** 0.5
    assert abs(n_initial - expected) < 5 * sigma, \
        f'initial cases {n_initial} far from expected {expected:.0f} +/- {sigma:.1f}'


def test_set_prognoses_flips_state():
    """set_prognoses moves agents from susceptible to infected, sets ti_clearance > ti."""
    sim, hpv = _minimal_sim()
    sim.init()
    sus_uids = hpv.susceptible.uids
    assert len(sus_uids) >= 3
    target = sus_uids[:3]
    hpv.set_prognoses(target, sources=None)
    assert (~hpv.susceptible[target]).all()
    assert hpv.infected[target].all()
    assert (hpv.ti_clearance[target] > hpv.ti).all()


def test_step_state_clears_at_ti_clearance():
    """An agent whose ti_clearance is reached returns to susceptible (SIS)."""
    sim, hpv = _minimal_sim(init_prev=0.0)
    sim.init()
    target = ss.uids([0])
    hpv.set_prognoses(target, sources=None)
    hpv.ti_clearance[target] = hpv.ti
    hpv.step_state()
    assert hpv.susceptible[target].all()
    assert (~hpv.infected[target]).all()


def test_runs_a_few_timesteps():
    """End-to-end: minimal sim with random net runs without error."""
    sim, hpv = _minimal_sim(init_prev=0.05, n_agents=500, n_steps=4)
    sim.run()
    assert 'n_infected' in sim.results.hpv16
    assert sim.results.hpv16.n_infected[-1] >= 0