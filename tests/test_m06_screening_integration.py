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


def test_cytology_primary_screen_populates_n_dx():
    """Regression: a screening product with no plain 'positive' outcome
    (cytology's ascus/abnormal) works as a PRIMARY screen.

    Upstream ss.BaseScreening.step hardcodes outcomes['positive'] for n_dx,
    which KeyErrors for such products; hpv.BaseScreening overrides step to
    count any non-negative result instead.
    """
    intv = hpv.routine_screening(
        name='primary', product='lbc', prob=1.0,
        age_range=[30, 50], sex='f', start_year=2021, end_year=2024,
    )
    sim = _baseline_sim_with([intv])
    sim.run()  # must not raise KeyError: 'positive'
    live = sim.interventions['primary']
    n_screened = np.asarray(live.results['n_screened'])
    n_dx = np.asarray(live.results['n_dx'])
    assert n_screened.sum() > 0
    assert n_dx.sum() > 0                 # ascus/abnormal are counted as diagnosed
    assert (n_dx <= n_screened).all()     # diagnosed is a subset of screened


def test_hpv_type_primary_screen_runs():
    """hpv_type (positive_1618/positive_ohr, no plain 'positive') as a primary screen."""
    intv = hpv.routine_screening(
        name='primary', product='hpv_type', prob=1.0,
        age_range=[30, 50], sex='f', start_year=2021, end_year=2024,
    )
    sim = _baseline_sim_with([intv])
    sim.run()  # must not raise
    live = sim.interventions['primary']
    n_screened = np.asarray(live.results['n_screened'])
    n_dx = np.asarray(live.results['n_dx'])
    assert n_screened.sum() > 0
    assert (n_dx <= n_screened).all()


def test_via_primary_n_dx_unchanged():
    """A product WITH 'positive' (via) still records n_dx as the positive count."""
    intv = hpv.routine_screening(
        name='primary', product='via', prob=1.0,
        age_range=[30, 50], sex='f', start_year=2021, end_year=2024,
    )
    sim = _baseline_sim_with([intv])
    sim.run()
    live = sim.interventions['primary']
    n_screened = np.asarray(live.results['n_screened'])
    n_dx = np.asarray(live.results['n_dx'])
    assert n_dx.sum() > 0
    assert (n_dx <= n_screened).all()


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
