"""Multiscale-agents feature tests."""
import numpy as np
import hpvsim as hpv

ANCHOR = dict(location='nigeria', genotypes=['hpv16'], start=1990, stop=2030,
              dt=0.25, rand_seed=0, verbose=0)


def test_ms_agent_ratio_defaults_to_one():
    """ms_agent_ratio defaults to 1 and is exposed on the HPV module pars."""
    mod = hpv.HPV(genotype='hpv16')
    assert int(mod.pars.ms_agent_ratio) == 1


def test_ms_agent_ratio_forwarded_from_sim():
    """hpv.Sim(ms_agent_ratio=N) forwards to every auto-built genotype module."""
    sim = hpv.Sim(n_agents=200, ms_agent_ratio=5, **ANCHOR)
    sim.init()
    assert int(sim.diseases.hpv16.pars.ms_agent_ratio) == 5


def test_ratio_one_is_bit_identical():
    """ms_agent_ratio=1 must reproduce the pre-feature results bit-for-bit."""
    base = hpv.Sim(n_agents=2000, **ANCHOR)
    base.run()
    one = hpv.Sim(n_agents=2000, ms_agent_ratio=1, **ANCHOR)
    one.run()
    a = np.asarray(base.results.hpv16.new_cancers)
    b = np.asarray(one.results.hpv16.new_cancers)
    assert np.array_equal(a, b)
