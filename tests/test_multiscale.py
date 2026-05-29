"""Multiscale-agents feature tests."""
import numpy as np
import pytest
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


def test_split_shrinks_cancer_agents_to_fractional_scale():
    """Multiscale resolution (binomial-on-original, Task 6) leaves no separate
    fine agents; instead it shrinks the resolved cancer agents to a fractional
    scale ``k/ratio`` (the progressing fraction). Over a multi-year run some
    cancer agents must therefore carry scale strictly below 1.0."""
    sim = hpv.Sim(n_agents=5000, ms_agent_ratio=10, **ANCHOR)
    sim.run()
    mod = sim.diseases.hpv16
    ppl = sim.people
    cancer = mod.cancerous.uids
    assert len(cancer) > 0, 'expected some cancer agents over a 40-year run'
    cancer_scale = np.asarray(ppl.scale[cancer])
    # At least some cancer agents resolved to a sub-unit (fractional) scale.
    assert np.any(cancer_scale < 0.9999), 'expected fractional-scale cancer agents'
    # No agent may carry a negative or >1 relative scale from the split.
    assert np.all(cancer_scale > 0) and np.all(cancer_scale <= 1.0 + 1e-9)


def test_split_is_reproducible():
    """Same seed twice -> identical scaled cancer totals under splitting."""
    def total(seed):
        s = hpv.Sim(n_agents=5000, ms_agent_ratio=10,
                    **{**ANCHOR, 'rand_seed': seed})
        s.run()
        return float(np.asarray(s.results.hpv16.new_cancers).sum())
    assert total(7) == total(7)


def test_split_preserves_array_integrity():
    """All module/people arrays stay length-consistent after growth; no NaN ages."""
    sim = hpv.Sim(n_agents=5000, ms_agent_ratio=10, **ANCHOR)
    sim.run()
    ppl = sim.people
    assert len(ppl.age.raw) == len(ppl.scale.raw) == len(sim.diseases.hpv16.cancerous.raw)
    assert int(np.isnan(np.asarray(ppl.age[ppl.auids])).sum()) == 0


def test_ratio_one_still_identical_after_split_code():
    """ms_agent_ratio=1 remains bit-identical (split is a no-op at ratio 1)."""
    base = hpv.Sim(n_agents=2000, **ANCHOR)
    base.run()
    one = hpv.Sim(n_agents=2000, ms_agent_ratio=1, **ANCHOR)
    one.run()
    assert np.array_equal(np.asarray(base.results.hpv16.new_cancers),
                          np.asarray(one.results.hpv16.new_cancers))


def test_split_does_not_inflate_infections_via_network():
    """Total people-space infections must be ~scale-invariant across
    ms_agent_ratio (within noise), not inflated or deflated by the ratio.

    Live test (Task 6). Two accounting fixes make this hold:
      (a) ``HPV.update_results`` scale-weights the per-step ``new_infections``
          tally (base starsim counts raw ``count_nonzero(ti_infected==ti)``,
          which would over-count any sub-unit-scale agent), so
          ``cum_infections`` is in people-space.
      (b) The CIN->cancer multiscale resolution no longer grows/removes
          placeholder agents (binomial-on-original); growing agents mid-run
          shifted starsim's slot-based CRN and systematically depressed
          transmission. With no population churn, transmission — and hence the
          cumulative infection total — stays within tolerance of single-scale.
    A small residual remains (the resolution reassigns a few CIN agents between
    the cancer/clear paths), so we allow the same 10% band as before."""
    cfg = dict(location='nigeria', genotypes=['hpv16'], start=1990, stop=2030,
               dt=0.25, total_pop=1e6, verbose=0)
    def cum_inf(ratio, seed):
        s = hpv.Sim(n_agents=5000, ms_agent_ratio=ratio, rand_seed=seed, **cfg)
        s.run()
        r = s.results.hpv16
        key = 'cum_infections' if 'cum_infections' in r else 'new_infections'
        v = (float(r[key][-1]) if key == 'cum_infections'
             else float(np.asarray(r[key]).sum()))
        return v * float(s.pars.pop_scale)
    base = np.mean([cum_inf(1, sd) for sd in range(4)])
    ms   = np.mean([cum_inf(10, sd) for sd in range(4)])
    assert abs(ms - base) / base < 0.10, f'infections inflated: {ms:.0f} vs {base:.0f}'


def test_multigenotype_split_keeps_modules_consistent():
    """With 2 genotypes + splitting, all module arrays match people length and
    no agent is cancerous in two genotypes at once."""
    sim = hpv.Sim(n_agents=4000, genotypes=['hpv16', 'hpv18'],
                  ms_agent_ratio=8, location='nigeria',
                  start=1990, stop=2030, dt=0.25, rand_seed=1, verbose=0)
    sim.run()
    n = len(sim.people.age.raw)
    g16, g18 = sim.diseases.hpv16, sim.diseases.hpv18
    assert len(g16.cancerous.raw) == len(g18.cancerous.raw) == n
    both = np.asarray(g16.cancerous.raw) & np.asarray(g18.cancerous.raw)
    assert both.sum() == 0, 'no agent may have invasive cancer in two genotypes'
