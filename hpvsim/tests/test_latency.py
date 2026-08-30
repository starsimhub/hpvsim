"""Tests for HPV latency: entry (hpv_control_prob), reactivation
(hpv_reactivation), and the HIV rel_reactivation connector effect."""
import numpy as np
import starsim as ss
import hpvsim as hpv


def test_hpv_has_latency_pars_and_states():
    """HPV defines hpv_control_prob/hpv_reactivation pars and the latency states."""
    mod = hpv.HPV(genotype='hpv16')
    p = mod.pars
    assert p.hpv_control_prob == 0.0
    assert p.hpv_reactivation.value == 0.025
    for name in ('to_latent', 'ti_latent', 'ti_reactivation'):
        assert hasattr(mod, name), f'HPV missing state {name!r}'
    assert hasattr(mod, '_latent_bern')
    assert hasattr(mod, '_reactivation_bern')


def test_latency_states_default_correctly():
    """to_latent defaults False; ti_latent/ti_reactivation default nan; latent (existing) defaults False."""
    sim = hpv.Sim(n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0)
    sim.init()
    mod = sim.diseases.hpv16
    uids = sim.people.auids
    assert not mod.to_latent[uids].any()
    assert not mod.latent[uids].any()
    assert np.all(np.isnan(mod.ti_latent[uids]))
    assert np.all(np.isnan(mod.ti_reactivation[uids]))
