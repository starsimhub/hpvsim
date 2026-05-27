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
    sim.pars['interventions'] = [ss.treat_num(product=tx_instance, prob=0.0)]
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
    """ablation row for precin has efficacy=0; precin agents must be all
    classified unsuccessful for that state contribution."""
    sim = _four_genotype_sim()
    t = _attach_tx_and_init(sim, hpv_tx(name='ablation'))
    uids = sim.people.alive.uids[:3]
    sim.diseases['hpv16'].precin[uids] = True
    out = t.administer(uids)
    # ablation efficacy on precin = 0, so no agent should be classified successful from this path
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
    """Successful treatment clears ti_cin and ti_cancerous to NaN."""
    sim = _four_genotype_sim()
    t = _attach_tx_and_init(sim, hpv_tx(name='excision'))
    uids = sim.people.alive.uids[:3]
    sim.diseases['hpv16'].cin[uids] = True
    sim.diseases['hpv16'].ti_cin[uids] = 5.0
    sim.diseases['hpv16'].ti_cancerous[uids] = 10.0
    out = t.administer(uids)
    succ = out['successful']
    if len(succ):
        assert np.all(np.isnan(sim.diseases['hpv16'].ti_cin[succ]))
        assert np.all(np.isnan(sim.diseases['hpv16'].ti_cancerous[succ]))


def test_tx_empty_uids_returns_empty_outcomes():
    sim = _four_genotype_sim()
    t = _attach_tx_and_init(sim, hpv_tx(name='ablation'))
    out = t.administer(ss.uids())
    assert len(out['successful']) == 0
    assert len(out['unsuccessful']) == 0
