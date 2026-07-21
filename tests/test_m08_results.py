import numpy as np
import hpvsim as hpv


def test_hiv_stratified_results_present_and_shaped():
    sim = hpv.Sim(n_agents=500, start=2000, stop=2003, dt=0.25, location='nigeria',
                  genotypes=[16, 18], diseases=[hpv.HIV(beta_m2f=0.004)])
    sim.run()
    res = sim.results.hivstratifiedresults
    for key in ('cancers_with_hiv', 'cancers_no_hiv',
                'hpv_prevalence_with_hiv', 'hpv_prevalence_no_hiv'):
        assert key in res
        assert len(res[key]) == len(sim.results.timevec)
    # Prevalence is a fraction in [0, 1].
    assert np.all((res['hpv_prevalence_with_hiv'] >= 0) &
                  (res['hpv_prevalence_with_hiv'] <= 1))


def test_stratified_cancers_sum_matches_total():
    sim = hpv.Sim(n_agents=1500, start=1990, stop=2010, dt=0.25, location='nigeria',
                  genotypes=[16, 18, 'hi5', 'ohr'], diseases=[hpv.HIV(beta_m2f=0.004)])
    sim.run()
    res = sim.results.hivstratifiedresults
    strat_total = res['cancers_with_hiv'].sum() + res['cancers_no_hiv'].sum()
    # Per-genotype modules each record new_cancers; HPVTotal sums them.
    hpv_total = sim.results.hpvtotal['new_cancers'].sum()
    assert strat_total > 0
    # HIVStratifiedResults runs after step_die, so an agent who turns cancerous
    # and dies from background demographics in the same step is counted by
    # HPVTotal (recorded in step_state) but not here. That makes strat_total a
    # lower bound on hpv_total. See the comment in HIVStratifiedResults.step.
    assert strat_total <= hpv_total


def test_stratified_cancers_scale_weighted_at_ms_gt_1():
    """HIV-stratified cancer counts must be scale-weighted by people.scale.

    At ms_agent_ratio > 1, cancers concentrate on spawned fine agents (scale =
    1/ratio). Raw (unweighted) sums would over-count them ~ratio-fold and blow
    past the (scale-weighted) HPVTotal.new_cancers; scale-weighted counts keep
    the stratified total a lower bound on the total. Regression guard for the
    fix to HIVStratifiedResults.step (was raw int sums).
    """
    sim = hpv.Sim(n_agents=1500, start=1990, stop=2010, dt=0.25, location='nigeria',
                  genotypes=[16, 18, 'hi5', 'ohr'], ms_agent_ratio=5,
                  diseases=[hpv.HIV(beta_m2f=0.004)])
    sim.run()
    res = sim.results.hivstratifiedresults
    assert np.issubdtype(res['cancers_with_hiv'].dtype, np.floating)
    strat_total = res['cancers_with_hiv'].sum() + res['cancers_no_hiv'].sum()
    hpv_total = sim.results.hpvtotal['new_cancers'].sum()
    assert strat_total > 0
    assert strat_total <= hpv_total * 1.001  # float tolerance; raw counts would exceed
