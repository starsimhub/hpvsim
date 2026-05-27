"""Unit tests for hpv.radiation — cancer treatment product."""
import numpy as np
import starsim as ss
import hpvsim as hpv
from hpvsim.products import radiation as hpv_radiation


def _four_genotype_sim():
    return hpv.Sim(
        n_agents=200, start=2020, stop=2021, location='nigeria',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
    )


def _attach_and_init(sim, p_instance):
    """Attach product to stub treat_num and init; return live post-init copy."""
    sim.pars['interventions'] = [ss.treat_num(product=p_instance, prob=0.0)]
    sim.init()
    return sim.interventions[0].product


def test_radiation_extends_ti_dead_cancer_on_cancerous_agents():
    sim = _four_genotype_sim()
    r = _attach_and_init(sim, hpv_radiation())
    uids = sim.people.alive.uids[:3]
    sim.diseases['hpv16'].cancerous[uids] = True
    sim.diseases['hpv16'].ti_dead_cancer[uids] = 100.0
    r.administer(uids)
    # ti_dead_cancer must have been extended
    assert np.all(sim.diseases['hpv16'].ti_dead_cancer[uids] > 100.0)


def test_radiation_skips_non_cancer_agents():
    sim = _four_genotype_sim()
    r = _attach_and_init(sim, hpv_radiation())
    uids = sim.people.alive.uids[:3]
    sim.diseases['hpv16'].cancerous[uids] = False  # not cancer
    sim.diseases['hpv16'].ti_dead_cancer[uids] = np.nan
    r.administer(uids)
    assert np.all(np.isnan(sim.diseases['hpv16'].ti_dead_cancer[uids]))


def test_radiation_empty_uids_noop():
    sim = _four_genotype_sim()
    r = _attach_and_init(sim, hpv_radiation())
    out = r.administer(ss.uids())
    assert len(out) == 0


def test_radiation_default_duration_v2_match():
    """Default duration is normal(18 months, 2 months) converted to years."""
    r = hpv_radiation()
    assert r.pars.dur['par1'] == 18 / 12  # mean: 1.5 years
    assert r.pars.dur['par2'] == 2 / 12   # sd: ~0.167 years
