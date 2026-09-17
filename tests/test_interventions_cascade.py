"""Integration tests for the full screen -> triage -> treat cascade."""
import numpy as np
import starsim as ss
import hpvsim as hpv


def _cascade_sim(intvs):
    return hpv.Sim(
        n_agents=500, start=2020, stop=2025, location='nigeria',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        interventions=intvs,
    )


def test_full_cascade_composes():
    """screen -> triage -> treat composes by ordering and eligibility callbacks."""
    screen = hpv.routine_screening(
        name='primary', product='hpv', prob=0.7,
        age_range=[30, 50], sex='f',
        start_year=2021, end_year=2024,
    )
    triage = hpv.routine_triage(
        name='colpo', product='colposcopy', prob=0.9,
        eligibility=lambda s: s.interventions['primary'].outcomes['positive'],
        start_year=2021, end_year=2024,
    )
    treat = hpv.treat_num(
        name='excision_rx', product='excision', prob=0.8,
        eligibility=lambda s: s.interventions['colpo'].outcomes['hsil'],
    )
    sim = _cascade_sim([screen, triage, treat])
    sim.run()
    # Each stage must actually fire — guards against silent step()-vs-ti bugs.
    assert sim.interventions['primary'].screened.uids.size > 0, 'screen did not fire'
    assert sim.interventions['colpo'].screened.uids.size > 0, 'triage did not fire'
    assert sim.interventions['excision_rx'].cin_treated.uids.size > 0, 'treat did not fire'
    # Sanity: triage is a subset of screen
    assert sim.interventions['colpo'].screened.uids.size <= sim.interventions['primary'].screened.uids.size


def test_cascade_order_dependency_no_exception():
    """Registering treat BEFORE screen is legal but treat.outcomes won't have
    positives on the same step screen runs. Documented contract; must not raise."""
    screen = hpv.routine_screening(
        name='primary', product='hpv', prob=1.0,
        age_range=[30, 50], sex='f',
        start_year=2021, end_year=2023,
    )
    treat = hpv.treat_num(
        name='wrong_order_rx', product='excision', prob=1.0,
        eligibility=lambda s: s.interventions['primary'].outcomes['positive'],
    )
    # Wrong order: treat registered before screen
    sim = _cascade_sim([treat, screen])
    sim.run()
    # Either cin_treated count is acceptable; only the absence of an error matters.
    live_treat = sim.interventions['wrong_order_rx']
    assert isinstance(live_treat.cin_treated.uids.size, (int, np.integer))


def test_linked_txvx_in_cascade():
    """linked_txvx with eligibility=triage.outcomes['lsil'] gates txvx delivery."""
    screen = hpv.routine_screening(
        name='primary', product='hpv', prob=1.0,
        age_range=[25, 55], sex='f',
        start_year=2021, end_year=2024,
    )
    triage = hpv.routine_triage(
        name='colpo', product='colposcopy', prob=1.0,
        eligibility=lambda s: s.interventions['primary'].outcomes['positive'],
        start_year=2021, end_year=2024,
    )
    linked = hpv.linked_txvx(
        name='linked_v',
        product='txvx1',
        prob=1.0,
        eligibility=lambda s: s.interventions['colpo'].outcomes['lsil'],
    )
    sim = _cascade_sim([screen, triage, linked])
    sim.run()
    # linked.tx_vaccinated count <= triage.screened count (LSIL is a subset)
    assert sim.interventions['linked_v'].tx_vaccinated.uids.size <= sim.interventions['colpo'].screened.uids.size
