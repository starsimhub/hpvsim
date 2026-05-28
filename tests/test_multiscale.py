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


def test_cancer_count_is_scale_weighted():
    """new_cancers weights by people.scale, not raw agent count."""
    sim = hpv.Sim(n_agents=3000, **ANCHOR)
    sim.init()
    mod = sim.diseases.hpv16
    ppl = sim.people

    # Force two agents into the CIN->cancerous transition this step at
    # known scales, and verify the recorded tally is the scale sum (1.5),
    # not the raw count (2).
    uids = ppl.auids[:2]
    ppl.scale[uids] = np.array([1.0, 0.5])
    mod.cin[uids] = True
    mod.cancerous[uids] = False
    mod.ti_cancerous[uids] = sim.t.ti
    mod.ti_clearance[uids] = np.nan
    mod.infected[uids] = True
    mod.step_state()
    ti = sim.t.ti
    assert np.isclose(float(mod.results.new_cancers[ti]), 1.5)
    # age tally also scale-weighted: sum(age*scale)
    expected_age = float((np.asarray(ppl.age[uids]) * np.array([1.0, 0.5])).sum())
    assert np.isclose(float(mod.results.sum_age_at_cancer[ti]), expected_age)


def test_hpvtotal_counts_are_scale_weighted():
    """HPVTotal union counts weight by people.scale: halving every agent's
    scale halves the counts (n_susceptible, n_infected, cum_infections_unique)."""
    sim = hpv.Sim(n_agents=1000, **ANCHOR)
    sim.init()
    ppl = sim.people
    tot = [a for a in sim.analyzers.values()
           if a.__class__.__name__ == 'HPVTotal'][0]
    ti = sim.t.ti

    def counts():
        tot.step()
        return (float(tot.results['n_susceptible'][ti]),
                float(tot.results['n_infected'][ti]),
                float(tot.results['cum_infections_unique'][ti]))

    ppl.scale[ppl.auids] = 1.0
    base = counts()
    ppl.scale[ppl.auids] = 0.5
    half = counts()
    assert base[0] > 0
    for b, h in zip(base, half):
        assert np.isclose(h, 0.5 * b, rtol=1e-6)
