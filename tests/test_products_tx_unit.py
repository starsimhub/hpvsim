"""Unit tests for hpv.tx — per-genotype state-flip treatment."""
import numpy as np
import pytest
import starsim as ss
import hpvsim as hpv
from hpvsim.products import tx as hpv_tx


def _four_genotype_sim():
    return hpv.Sim(
        n_agents=200, start=2020, stop=2021, location='nigeria',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
    )


def _attach_tx_and_init(sim, tx_instance):
    """Attach the tx product to a stub treat_num and init the sim.

    Returns the LIVE post-init copy of the product (Starsim deep-copies
    constructor inputs).
    """
    sim.pars['interventions'] = [hpv.treat_num(product=tx_instance, prob=0.0)]
    sim.init()
    return sim.interventions[0].product


def test_tx_unknown_name_raises():
    with pytest.raises(ValueError, match='Unknown tx product name'):
        hpv_tx(name='nope')


def test_tx_ablation_flips_cin_to_false():
    """Successful ablation flips cin[g]=False and schedules ti_clearance=ti+1."""
    sim = _four_genotype_sim()
    t_init = hpv_tx(name='ablation')
    t = _attach_tx_and_init(sim, t_init)
    uids = sim.people.alive.uids[:3]
    sim.diseases['hpv16'].cin[uids] = True
    out = t.administer(uids)
    assert 'successful' in out and 'unsuccessful' in out
    succ = out['successful']
    if len(succ):
        assert np.all(~sim.diseases['hpv16'].cin[succ])
        assert np.allclose(sim.diseases['hpv16'].ti_clearance[succ], sim.ti + 1)


def test_tx_zero_efficacy_row_means_zero_successful():
    """ablation row for cancerous has efficacy=0; cancerous-only agents must
    all be classified unsuccessful for that state contribution.

    All four genotype modules are cleared and only hpv16 set to cancerous —
    otherwise agents with a residual precin flag on hpv18 (from the Nigeria
    init prevalence) would hit the now-nonzero ablation.precin path.
    """
    sim = _four_genotype_sim()
    t = _attach_tx_and_init(sim, hpv_tx(name='ablation'))
    uids = sim.people.alive.uids[:3]
    for g in ('hpv16', 'hpv18', 'hi5', 'ohr'):
        sim.diseases[g].precin[uids] = False
        sim.diseases[g].cin[uids] = False
        sim.diseases[g].cancerous[uids] = False
    sim.diseases['hpv16'].cancerous[uids] = True
    out = t.administer(uids)
    # ablation efficacy on cancerous = 0, so no agent should be classified successful from this path
    assert len(out['successful']) == 0
    assert len(out['unsuccessful']) == len(uids)


def test_tx_disjoint_outcomes_keys():
    sim = _four_genotype_sim()
    t = _attach_tx_and_init(sim, hpv_tx(name='excision'))
    uids = sim.people.alive.uids[:5]
    sim.diseases['hpv16'].cin[uids] = True
    out = t.administer(uids)
    succ_set = set(int(u) for u in out['successful'])
    unsucc_set = set(int(u) for u in out['unsuccessful'])
    assert succ_set.isdisjoint(unsucc_set)
    assert succ_set | unsucc_set == set(int(u) for u in uids)


def test_tx_clears_ti_cin_and_ti_cancerous_on_success():
    """Successful treatment clears ti_cin and ti_cancerous to NaN.

    All four genotype modules are cleared and only hpv16 set to cin —
    otherwise agents with a residual precin flag on hpv18 (from Nigeria
    init prevalence) would land in the excision.precin path first, clear
    precin, and never touch hpv16.ti_cin.
    """
    sim = _four_genotype_sim()
    t = _attach_tx_and_init(sim, hpv_tx(name='excision'))
    uids = sim.people.alive.uids[:3]
    for g in ('hpv16', 'hpv18', 'hi5', 'ohr'):
        sim.diseases[g].precin[uids] = False
        sim.diseases[g].cin[uids] = False
        sim.diseases[g].cancerous[uids] = False
    sim.diseases['hpv16'].cin[uids] = True
    sim.diseases['hpv16'].ti_cin[uids] = 5.0
    sim.diseases['hpv16'].ti_cancerous[uids] = 10.0
    out = t.administer(uids)
    succ = out['successful']
    if len(succ):
        assert np.all(np.isnan(sim.diseases['hpv16'].ti_cin[succ]))
        assert np.all(np.isnan(sim.diseases['hpv16'].ti_cancerous[succ]))


def test_tx_treated_woman_clears_to_susceptible_not_latency():
    """Treatment-induced clearance must not strand a to_latent woman in latency.

    Invisible at the default hpv_control_prob=0, so latency is forced on and
    beta=0 rules out same-step reinfection.
    """
    sim = hpv.Sim(
        n_agents=200, start=2020, stop=2022, location='nigeria',
        diseases=[hpv.HPV('hpv16', hpv_control_prob=1.0, beta=0)],
    )
    t = _attach_tx_and_init(sim, hpv_tx(name='ablation'))
    hpv16 = sim.diseases['hpv16']
    uids = sim.people.female.uids[:10]
    hpv16.susceptible[uids] = False
    hpv16.infected[uids] = True
    hpv16.precin[uids] = False
    hpv16.cin[uids] = True
    hpv16.cancerous[uids] = False
    hpv16.to_latent[uids] = True
    succ = t.administer(uids)['successful']
    assert len(succ), 'ablation has efficacy 0.936 on cin; expected successes'
    # Two steps: ti increments at the end of a step, and treatment schedules
    # clearance for module.ti + 1.
    sim.run_one_step()
    sim.run_one_step()
    still_alive = succ[sim.people.alive[succ]]
    assert len(still_alive)
    assert not hpv16.latent[still_alive].any()
    assert hpv16.susceptible[still_alive].all()


def test_tx_empty_uids_returns_empty_outcomes():
    sim = _four_genotype_sim()
    t = _attach_tx_and_init(sim, hpv_tx(name='ablation'))
    out = t.administer(ss.uids())
    assert len(out['successful']) == 0
    assert len(out['unsuccessful']) == 0


def test_tx_neither_name_nor_df_raises():
    with pytest.raises(ValueError, match='at least one'):
        hpv_tx()


def test_tx_name_and_df_gives_custom_name_to_df_built_product():
    """name= + df= keeps df as the row table and uses name as the module
    name — enables multiple df-built tx products in a single sim."""
    import pandas as pd
    df = pd.DataFrame([
        {'name': 'x', 'state': 'cin', 'genotype': 'all', 'efficacy': 0.0},
    ])
    t = hpv_tx(name='my_tx', df=df)
    assert t.name == 'my_tx'
    assert list(t.df.state.unique()) == ['cin']
