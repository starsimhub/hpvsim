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


def test_ratio_one_spawns_no_fine_agents():
    """ms_agent_ratio=1 grows no fine agents (true no-op split path)."""
    sim = hpv.Sim(location='nigeria', genotypes=['hpv16'], start=1990, stop=2010,
                  dt=0.25, n_agents=3000, ms_agent_ratio=1, verbose=0)
    sim.run()
    assert not bool(np.asarray(sim.diseases.hpv16.multiscale_fine.raw).any())


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


def test_split_conserves_scale_mass_and_marks_fine():
    """Splitting shrinks coarse cancer agents and adds fine agents at reduced
    scale; fine agents are tagged."""
    sim = hpv.Sim(n_agents=5000, ms_agent_ratio=10, **ANCHOR)
    sim.run()
    mod = sim.diseases.hpv16
    ppl = sim.people
    fine = mod.multiscale_fine.uids
    assert len(fine) > 0, 'expected some fine agents over a 40-year run'
    assert np.all(np.asarray(ppl.scale[fine]) < 0.9999)


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




def test_split_does_not_inflate_infections_via_network():
    """Fine agents must not transmit: total infections should be ~scale-
    invariant across ms_agent_ratio (within noise), not inflated by ratio.

    Three fixes together make this hold: (1) fine agents are excluded from the
    sexual network (no spurious transmission); (2) ``HPV.update_results`` drops
    fine agents from the ``new_infections`` tally (they are sub-resolutions of
    an already-counted source infection, not new transmission events — without
    this each 1/ratio fine agent was counted as a full infection); (3) the
    cancer original is left transmitting until cancer onset (not network-
    excluded) so the coarse transmission timeline matches single-scale."""
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


def test_treatment_tally_is_scale_weighted():
    """new_cin_treated sums people.scale for treated agents, not raw count.

    Fine agents have scale=1/ratio, so under multiscale ms_agent_ratio>1 the
    sum of scales for treated fine agents is less than their raw count. The
    old len(new) code would over-count; the fix uses scale[new].sum().

    Strategy: run matching ratio=1 and ratio=10 sims with same n_agents and
    aggressive treatment.  Both sims represent the same underlying population
    (pop_scale=1, same n_agents), so they should produce near-identical
    scale-weighted treatment tallies (within ~5% stochastic noise).

    Under the OLD len(new) code, fine agents (scale=0.1 in the ratio=10 sim)
    each count as 1 treatment instead of 0.1, inflating the ratio=10 tally by
    approximately N_fine_treated * 0.9 ~= 681 extra (a ~18% over-count above
    the ratio=1 baseline of ~3758).  The test uses a <10% tolerance so this
    inflation would cause a clear FAIL on old code.
    """
    cfg = dict(location='nigeria', genotypes=['hpv16'], start=1990, stop=2010,
               dt=0.25, verbose=0, rand_seed=42)

    def make_intvs():
        return [
            hpv.routine_screening(product='via', prob=0.9, start_year=1995,
                                  age_range=[25, 55], name='screen'),
            hpv.treat_num(name='cin_rx', product='excision', prob=1.0),
        ]

    sim1 = hpv.Sim(n_agents=3000, ms_agent_ratio=1,  interventions=make_intvs(), **cfg)
    sim1.run()

    sim10 = hpv.Sim(n_agents=3000, ms_agent_ratio=10, interventions=make_intvs(), **cfg)
    sim10.run()

    iv1  = sim1.interventions['cin_rx']
    iv10 = sim10.interventions['cin_rx']

    total1  = float(np.asarray(iv1.results['new_cin_treated']).sum())
    total10 = float(np.asarray(iv10.results['new_cin_treated']).sum())

    assert total1 > 0, 'no CIN treatments in ratio=1 sim — test is vacuous'
    assert total10 > 0, 'no CIN treatments in ratio=10 sim — test is vacuous'

    # Require fine agents to be present (test is vacuous without them)
    fine_uids = sim10.diseases.hpv16.multiscale_fine.uids
    assert len(fine_uids) > 0, 'no fine agents in ratio=10 sim — test is vacuous'

    # Scale-weighted totals should match within 10% (both sims: same pop, same seed).
    # Under OLD len(new) code: fine-agent treatments inflate total10 by ~18%,
    # causing this assertion to FAIL.  Under NEW scale[new].sum() code: PASSES.
    assert abs(total10 - total1) <= 0.10 * total1, (
        f'scale-weighting broken: ratio=10 tally={total10:.1f} diverges >10% from '
        f'ratio=1 tally={total1:.1f}; old len(new) code over-counts fine agents by ~18%'
    )


def test_screening_tally_is_scale_weighted():
    """n_screened/n_dx sum people.scale, not raw agent count.

    Fine agents have scale=1/ratio.  Under the OLD ss.BaseScreening.step
    (len(accept_uids) / len(outcomes['positive'])) a fine agent counts as
    1 screened/diagnosed instead of 1/ratio — inflating the ratio=10 tallies
    relative to ratio=1.

    Strategy: run matching ratio=1 and ratio=10 sims with same n_agents and
    aggressive screening coverage.  Both sims represent the same underlying
    population, so scale-weighted n_screened / n_dx totals should agree within
    ~10% stochastic noise.  Under OLD len() code the ratio=10 tallies are
    inflated by ~(ratio-1)*n_fine_screened — easily >10% of the ratio=1
    baseline — causing a clear FAIL.  Under the NEW scale-weighted override
    the test PASSES.
    """
    cfg = dict(location='nigeria', genotypes=['hpv16'], start=1990, stop=2010,
               dt=0.25, verbose=0, rand_seed=42)

    def make_intvs():
        return [
            hpv.routine_screening(product='via', prob=0.9, start_year=1995,
                                  age_range=[25, 55], name='screen'),
        ]

    sim1 = hpv.Sim(n_agents=3000, ms_agent_ratio=1,  interventions=make_intvs(), **cfg)
    sim1.run()

    sim10 = hpv.Sim(n_agents=3000, ms_agent_ratio=10, interventions=make_intvs(), **cfg)
    sim10.run()

    sc1  = sim1.interventions['screen']
    sc10 = sim10.interventions['screen']

    total_screened1  = float(np.asarray(sc1.results['n_screened']).sum())
    total_screened10 = float(np.asarray(sc10.results['n_screened']).sum())
    total_dx1        = float(np.asarray(sc1.results['n_dx']).sum())
    total_dx10       = float(np.asarray(sc10.results['n_dx']).sum())

    assert total_screened1  > 0, 'no screenings in ratio=1 sim — test is vacuous'
    assert total_screened10 > 0, 'no screenings in ratio=10 sim — test is vacuous'

    # Require fine agents to be present (test is vacuous without them)
    fine_uids = sim10.diseases.hpv16.multiscale_fine.uids
    assert len(fine_uids) > 0, 'no fine agents in ratio=10 sim — test is vacuous'

    # Scale-weighted n_screened should match ratio=1 within 10%.
    # Under OLD len(accept_uids) code: fine-agent screens inflate total_screened10.
    assert abs(total_screened10 - total_screened1) <= 0.10 * total_screened1, (
        f'n_screened scale-weighting broken: ratio=10 tally={total_screened10:.1f} '
        f'diverges >10% from ratio=1 tally={total_screened1:.1f}'
    )

    # Scale-weighted n_dx should match ratio=1 within 10%.
    if total_dx1 > 0:
        assert abs(total_dx10 - total_dx1) <= 0.10 * total_dx1, (
            f'n_dx scale-weighting broken: ratio=10 tally={total_dx10:.1f} '
            f'diverges >10% from ratio=1 tally={total_dx1:.1f}'
        )
