"""Integration tests for screening and triage: ss.screen() delivering hpv.dx products."""
import numpy as np
import starsim as ss
import hpvsim as hpv

women = ss.ppl.female & (ss.ppl.age >= 30) & (ss.ppl.age < 50)


def _baseline_sim_with(interventions, analyzers=None):
    return hpv.Sim(
        n_agents=500, start=2020, stop=2025, location='nigeria',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        interventions=interventions, analyzers=analyzers, verbose=0,
    )


def _tracker(*names):
    """A function analyzer recording the agents each delivery reached, and the dict it records into"""
    reached = {name: [] for name in names}
    def track(sim):
        for name, uids in reached.items():
            uids.extend(sim.interventions[name].reached().tolist())
    return track, reached


def test_routine_screening_reaches_women_in_age_range():
    """Screening rounds reach some people, all of them women aged 30-50 when screened."""
    primary = ss.screen(hpv.dx('via'), where=women, coverage=0.5, interval=ss.years(1), start=2021, stop=2024, name='primary')
    ages = []
    def track(sim):
        uids = sim.interventions.primary.reached()
        assert sim.people.female[uids].all()
        ages.extend(sim.people.age[uids].tolist())
    sim = _baseline_sim_with([primary], analyzers=track)
    sim.run()
    assert sim.results.primary.delivered.sum() > 0
    assert len(ages) and min(ages) >= 30 and max(ages) < 50


def test_campaign_screening_runs():
    campaign = ss.screen(hpv.dx('via'), where=women, coverage=0.5, at=['2022'], name='campaign')
    sim = _baseline_sim_with([campaign])
    sim.run()
    delivered = np.asarray(sim.results.campaign.delivered)
    assert delivered.sum() > 0
    assert (delivered > 0).sum() == 1 # One round


def test_triage_consumes_screen_outcomes():
    """A triage triggered by screen-positives reaches only positives, on the same step."""
    primary = ss.screen(hpv.dx('via'), where=women, coverage=1.0, interval=ss.years(1), start=2021, stop=2022, name='primary')
    triage = ss.screen(hpv.dx('colposcopy'), who=primary.positive, coverage=1.0, name='triage')
    track, reached = _tracker('primary', 'triage')
    sim = _baseline_sim_with([primary, triage], analyzers=track)
    sim.run()
    r = sim.results
    assert np.allclose(np.asarray(r.triage.delivered), np.asarray(r.primary.positive))
    assert set(reached['triage']) <= set(reached['primary'])


def test_cytology_primary_screen_counts_results():
    """A product with several results (cytology's abnormal, ascus, normal) as a primary screen:
    each result is counted, and positive is any but the last"""
    primary = ss.screen(hpv.dx('lbc'), where=women, coverage=1.0, interval=ss.years(1), start=2021, stop=2024, name='primary')
    sim = _baseline_sim_with([primary])
    sim.run()
    r = sim.results.primary
    delivered = np.asarray(r.delivered)
    assert delivered.sum() > 0
    assert np.allclose(delivered, np.asarray(r.abnormal) + np.asarray(r.ascus) + np.asarray(r.normal))
    assert np.asarray(r.ascus).sum() > 0 # ASCUS is drawn for 10% of women with no lesion


def test_hpv_type_primary_screen_runs():
    """hpv_type (positive_1618/positive_ohr, no plain 'positive') as a primary screen."""
    primary = ss.screen(hpv.dx('hpv_type'), where=women, coverage=1.0, interval=ss.years(1), start=2021, stop=2024, name='primary')
    sim = _baseline_sim_with([primary])
    sim.run()
    r = sim.results.primary
    delivered = np.asarray(r.delivered)
    assert delivered.sum() > 0
    assert np.allclose(delivered, np.asarray(r.positive_1618) + np.asarray(r.positive_ohr) + np.asarray(r.negative))


def test_via_yields_by_state():
    """VIA's results follow products_dx.csv: positive for 15% of women with no lesion"""
    primary = ss.screen(hpv.dx('via'), where=women, coverage=1.0, interval=ss.years(1), start=2021, stop=2024, name='primary')
    sim = hpv.Sim(n_agents=3000, start=2020, stop=2025, location='nigeria', genotypes=[16], interventions=[primary], verbose=0)
    sim.run()
    r = sim.results.primary
    share = np.asarray(r.positive).sum()/np.asarray(r.delivered).sum()
    assert 0.12 < share < 0.2, share # Mostly susceptible women, at 0.15
