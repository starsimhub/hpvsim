"""Therapeutic vaccination: does it clear disease and confer immunity?

hpv.txvx is a treatment product, not a prophylactic. It clears infections and
lesions from a state x genotype efficacy table, then confers severity immunity
against future infection.

The shipped 'txvx1' product carries 1% placeholder efficacies, so these tests
build an explicit product with a plausible profile: a vaccine that clears most
infections and low-grade lesions, with weaker cross-protection beyond 16/18.
"""
import numpy as np
import pandas as pd
import starsim as ss
import hpvsim as hpv

GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')


def make_txvx(efficacy=1.0, **kwargs):
    df = pd.DataFrame([
        {'name': 'txv', 'state': state, 'genotype': g, 'efficacy': efficacy}
        for state in ('precin', 'cin') for g in GENOTYPES
    ])
    return hpv.txvx(df=df, rel_imm={'hpv16': 1.0, 'hpv18': 1.0, 'hi5': 0.5, 'ohr': 0.5},
                    imm_init=ss.beta_mean(mean=0.35, var=0.025), **kwargs)


def make_sim(interventions, n_agents=500, stop=2025):
    return hpv.Sim(
        n_agents=n_agents, start=2020, stop=stop, location='nigeria', rand_seed=1,
        diseases=[hpv.HPV(g) for g in GENOTYPES],
        interventions=interventions, verbose=0,
    )


def test_txvx_clears_lesions_and_confers_scaled_immunity():
    """A dose clears existing disease and leaves severity immunity behind.

    Immunity reaches everyone dosed — including those whose clearance draw
    failed — and is scaled per target genotype by rel_imm.
    """
    sim = make_sim([hpv.linked_txvx(name='txv', product=make_txvx(), prob=1.0,
                                    eligibility=lambda s: ss.uids())])
    sim.init()
    product = sim.interventions['txv'].product
    m16, mohr = sim.diseases['hpv16'], sim.diseases['ohr']

    uids = sim.people.female.uids[:40]
    m16.infected[uids] = True
    m16.cin[uids] = True
    product.administer(uids)

    assert not m16.cin[uids].any(), 'lesions not cleared'
    assert np.isfinite(m16.ti_clearance[uids]).all(), 'clearance not scheduled'
    # Severity immunity, not susceptibility: the therapeutic never touches rel_sus.
    assert (m16.txvx_sev_imm[uids] > 0).all()
    np.testing.assert_allclose(mohr.txvx_sev_imm[uids],
                               m16.txvx_sev_imm[uids] * 0.5, rtol=1e-6)


def test_txvx_immunity_reaches_those_it_fails_to_cure():
    """Immunity is conferred on dosing, independently of clearance success."""
    sim = make_sim([hpv.linked_txvx(name='txv', product=make_txvx(efficacy=0.0),
                                    prob=1.0, eligibility=lambda s: ss.uids())])
    sim.init()
    product = sim.interventions['txv'].product
    m16 = sim.diseases['hpv16']
    uids = sim.people.female.uids[:20]
    m16.infected[uids] = True
    m16.cin[uids] = True

    out = product.administer(uids)
    assert len(out['successful']) == 0, 'zero-efficacy product cured someone'
    assert m16.cin[uids].all(), 'lesions cleared despite zero efficacy'
    assert (m16.txvx_sev_imm[uids] > 0).all(), 'no immunity for the uncured'


def test_txvx_in_a_screening_program_reduces_cancer():
    """Delivered to screen-positive women, the therapeutic averts cancer."""
    def run(with_txvx):
        ivs = [hpv.routine_screening(name='scr', product='hpv', prob=1.0,
                                     age_range=[30, 50], start_year=2022)]
        if with_txvx:
            ivs.append(hpv.linked_txvx(
                name='txv', product=make_txvx(efficacy=0.9), prob=1.0,
                eligibility=lambda s: s.interventions['scr'].outcomes['positive']))
        sim = make_sim(ivs, n_agents=3000, stop=2060)
        sim.run()
        return sim

    base, txv = run(False), run(True)
    assert txv.interventions['txv'].tx_vaccinated.uids.size > 0, 'nobody dosed'
    base_cancers = float(base.results.all_hpv.cum_cancers[-1])
    txv_cancers = float(txv.results.all_hpv.cum_cancers[-1])
    assert txv_cancers < base_cancers, (
        f'therapeutic did not avert cancer: {txv_cancers:,.0f} vs {base_cancers:,.0f}')
