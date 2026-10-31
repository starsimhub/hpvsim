"""Integration tests for the full screen -> triage -> treat cascade."""
import numpy as np
import starsim as ss
import hpvsim as hpv


def _cascade_sim(intvs):
    return hpv.Sim(
        n_agents=500, start=2020, stop=2025, location='nigeria',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        interventions=intvs, verbose=0,
    )


def _cascade():
    women = ss.ppl.female & (ss.ppl.age >= 30) & (ss.ppl.age < 50)
    screen = ss.screen(hpv.dx('hpv'), where=women, coverage=0.7, interval=ss.years(1), start=2021, stop=2024, name='primary')
    triage = ss.screen(hpv.dx('colposcopy'), who=screen.positive, coverage=0.9, name='colpo')
    treat = ss.treat(hpv.tx('excision'), who=triage.hsil & hpv.no_cancer, coverage=0.8, name='excision_rx')
    return screen, triage, treat


def test_full_cascade_composes():
    """screen -> triage -> treat composes by events: each step is triggered by a result of the one before."""
    sim = _cascade_sim(list(_cascade()))
    sim.run()
    r = sim.results
    # Each stage must actually fire
    assert r.primary.delivered.sum() > 0, 'screen did not fire'
    assert r.colpo.delivered.sum() > 0, 'triage did not fire'
    assert r.excision_rx.delivered.sum() > 0, 'treat did not fire'
    # Each step reaches at most those referred, on the same step
    assert (np.asarray(r.colpo.delivered) <= np.asarray(r.primary.positive) + 1e-6).all()
    assert (np.asarray(r.excision_rx.delivered) <= np.asarray(r.colpo.hsil) + 1e-6).all()


def test_cascade_order_does_not_matter():
    """A follow-up listed before the step that triggers it still runs after it, on the same step."""
    screen, triage, treat = _cascade()
    right = _cascade_sim([screen, triage, treat]).run()
    screen, triage, treat = _cascade()
    wrong = _cascade_sim([treat, triage, screen]).run()
    for name in ('primary', 'colpo', 'excision_rx'):
        assert np.allclose(right.results[name].delivered, wrong.results[name].delivered)


def test_linked_txvx_in_cascade():
    """A therapeutic vaccine triggered by colposcopy's LSIL results reaches at most those with them."""
    women = ss.ppl.female & (ss.ppl.age >= 25) & (ss.ppl.age < 55)
    screen = ss.screen(hpv.dx('hpv'), where=women, coverage=1.0, interval=ss.years(1), start=2021, stop=2024, name='primary')
    triage = ss.screen(hpv.dx('colposcopy'), who=screen.positive, coverage=1.0, name='colpo')
    linked = ss.treat(hpv.txvx('txvx1'), who=triage.lsil & hpv.no_cancer, coverage=1.0, name='linked_v')
    sim = _cascade_sim([screen, triage, linked])
    sim.run()
    r = sim.results
    assert r.linked_v.delivered.sum() <= r.colpo.lsil.sum() + 1e-6
