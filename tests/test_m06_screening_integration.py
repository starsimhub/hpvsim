"""Integration smoke tests for hpv.routine_screening / campaign_screening / triage."""
import numpy as np
import starsim as ss
import hpvsim as hpv


def _baseline_sim_with(interventions):
    return hpv.Sim(
        n_agents=500, start=2020, stop=2025, location='nigeria',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        interventions=interventions,
    )


def test_routine_screening_flips_screened():
    intv = hpv.routine_screening(
        name='primary',
        product='via',
        prob=0.5,
        age_range=[30, 50],
        sex='f',
        start_year=2021,
        end_year=2024,
    )
    sim = _baseline_sim_with([intv])
    sim.run()
    # Access the LIVE post-init copy via sim.interventions
    live_intv = sim.interventions['primary']
    assert live_intv.screened.uids.size > 0


def test_routine_screening_targets_female_only():
    intv = hpv.routine_screening(
        name='primary', product='via', prob=1.0,
        age_range=[30, 50], sex='f',
        start_year=2021, end_year=2022,
    )
    sim = _baseline_sim_with([intv])
    sim.run()
    live_intv = sim.interventions['primary']
    for u in live_intv.screened.uids[:20]:
        assert sim.people.female[u]


def test_campaign_screening_runs():
    intv = hpv.campaign_screening(
        name='campaign', product='via', prob=0.5,
        age_range=[30, 50], sex='f', years=[2022],
    )
    sim = _baseline_sim_with([intv])
    sim.run()
    live_intv = sim.interventions['campaign']
    assert live_intv.screened.uids.size > 0


def test_routine_triage_consumes_screen_outcomes():
    """A triage with eligibility=screen-positives only fires on positives."""
    screen = hpv.routine_screening(
        name='primary', product='via', prob=1.0,
        age_range=[30, 50], sex='f',
        start_year=2021, end_year=2022,
    )
    triage = hpv.routine_triage(
        name='triage',
        product='colposcopy',
        prob=1.0,
        eligibility=lambda s: s.interventions['primary'].outcomes['positive'],
        start_year=2021,
        end_year=2022,
    )
    sim = _baseline_sim_with([screen, triage])
    sim.run()
    # Triage screened set is <= screen screened set
    assert sim.interventions['triage'].screened.uids.size <= sim.interventions['primary'].screened.uids.size


def test_routine_screening_string_product_resolves_via_dx():
    """product='via' should resolve through hpv.dx(name='via')."""
    intv = hpv.routine_screening(
        name='primary', product='via', prob=0.1,
        age_range=[30, 50], sex='f', start_year=2021, end_year=2022,
    )
    # Access via sim.interventions after init
    sim = _baseline_sim_with([intv])
    sim.init()
    live = sim.interventions['primary']
    assert live.product.__class__.__name__ == 'dx'
    assert live.product.name == 'via'
