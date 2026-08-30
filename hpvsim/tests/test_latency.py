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


def _sim_with_forced_latency(control_prob=1.0, n_agents=500, stop=1995):
    """A short sim with hpv_control_prob forced high so clearing females
    reliably enter latency instead. Single genotype, no HIV (isolates the
    base latency mechanism from the HIV rel_reactivation effect)."""
    sim = hpv.Sim(n_agents=n_agents, location='nigeria', genotypes=[16],
                  start=1975, stop=stop, dt=1.0, rand_seed=0,
                  pars=dict(hpv_control_prob=control_prob))
    sim.run()
    return sim


def test_forced_control_prob_produces_latent_agents():
    """With hpv_control_prob=1.0, every female clearance becomes latency instead."""
    sim = _sim_with_forced_latency()
    mod = sim.diseases.hpv16
    uids = sim.people.auids
    latent_mask = mod.latent[uids]
    assert latent_mask.any(), 'expected at least one latent agent with hpv_control_prob=1.0'
    latent_uids = uids[latent_mask]
    assert np.all(~np.isnan(mod.ti_latent[latent_uids]))
    # Latent agents are neither susceptible nor (normally) infected.
    assert not mod.susceptible[latent_uids].any()
    assert not mod.to_latent[latent_uids].any(), 'to_latent should clear once latent is stamped'


def test_males_never_enter_latency():
    """hpv_control_prob only applies to females; males always clear normally."""
    sim = _sim_with_forced_latency()
    mod = sim.diseases.hpv16
    female_mask = sim.people.female[sim.people.auids]
    male_uids = sim.people.auids[~female_mask]
    assert not mod.latent[male_uids].any()


def test_zero_control_prob_is_a_no_op():
    """Default hpv_control_prob=0.0 must never produce latent agents."""
    sim = hpv.Sim(n_agents=500, location='nigeria', genotypes=[16],
                  start=1975, stop=1995, dt=1.0, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    uids = sim.people.auids
    assert not mod.latent[uids].any()
    assert not mod.to_latent[uids].any()
