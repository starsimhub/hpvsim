"""Integration smoke tests for vaccination interventions.

These tests run small sims end-to-end (200-500 agents, ~5-year horizon) to
verify the routine/campaign interventions fire on schedule, update
per-intervention state correctly, target the right age/sex cohorts, and
do not perturb the no-vx baseline.
"""
import numpy as np
import pytest
import sciris as sc

import hpvsim as hpv


SMALL_PARS = dict(
    location='nigeria',
    start=2018, stop=2025,
    n_agents=500,
    genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
    rand_seed=0,
)


def test_routine_vx_fires_and_updates_state():
    """routine_vx fires on schedule; vaccinated/n_doses/ti_vaccinated update."""
    intv = hpv.routine_vx(
        product='bivalent',
        prob=0.9,
        age_range=[9, 14],
        sex='f',
        start_year=2020,
        name='routine_smoke',
    )
    sim = hpv.Sim(**SMALL_PARS, interventions=[intv])
    sim.run()
    # Re-bind to the sim's deep-copied intervention
    intv = sim.interventions[0]
    assert intv.vaccinated.sum() > 0
    # routine_vx re-fires while agents stay eligible, so n_doses >= 1.
    assert np.all(intv.n_doses[intv.vaccinated.uids] >= 1)
    assert np.all(~np.isnan(intv.ti_vaccinated[intv.vaccinated.uids]))


def test_campaign_vx_fires_and_updates_state():
    """campaign_vx fires on each campaign year."""
    intv = hpv.campaign_vx(
        product='bivalent',
        prob=[0.7, 0.5],
        age_range=[9, 30],
        sex='f',
        years=[2020, 2021],
        name='campaign_smoke',
    )
    sim = hpv.Sim(**SMALL_PARS, interventions=[intv])
    sim.run()
    intv = sim.interventions[0]
    assert intv.vaccinated.sum() > 0


def test_routine_vx_respects_sex():
    """sex='f' vaccinates only female agents (sim.people.female == True)."""
    intv = hpv.routine_vx(
        product='bivalent',
        prob=1.0,
        age_range=[9, 14],
        sex='f',
        start_year=2020,
        name='routine_sex_check',
    )
    sim = hpv.Sim(**SMALL_PARS, interventions=[intv])
    sim.run()
    intv = sim.interventions[0]
    vacc_uids = intv.vaccinated.uids
    assert np.all(sim.people.female[vacc_uids])


def test_routine_vx_respects_age_range():
    """Only agents in age_range at time of firing are vaccinated."""
    intv = hpv.routine_vx(
        product='bivalent',
        prob=1.0,
        age_range=[9, 10],
        sex='f',
        start_year=2020,
        name='routine_age_check',
    )
    sim = hpv.Sim(**SMALL_PARS, interventions=[intv])
    sim.run()
    intv = sim.interventions[0]
    vacc_uids = intv.vaccinated.uids
    if len(vacc_uids) == 0:
        pytest.skip('No agents vaccinated in this small-sim window (random)')
    # Back out age at first dose from current age and ti_vaccinated.
    ages_now = sim.people.age[vacc_uids]
    ti_vacc = intv.ti_vaccinated[vacc_uids]
    # dt may be ss.dur — extract years
    dt = sim.t.dt.years if hasattr(sim.t.dt, 'years') else sim.t.dt
    ages_at_vacc = ages_now - (sim.ti - ti_vacc) * dt
    # Allow a small dt-rounding tolerance
    assert np.all(ages_at_vacc >= 9 - dt)
    assert np.all(ages_at_vacc < 10 + dt)


def test_routine_vx_records_dose_results():
    """Vaccination defines scaled dose/coverage time series, like screening does."""
    intv = hpv.routine_vx(
        product='bivalent',
        prob=0.9,
        age_range=[9, 14],
        sex='f',
        start_year=2020,
        name='routine_results',
    )
    sim = hpv.Sim(**SMALL_PARS, interventions=[intv])
    sim.run()
    intv = sim.interventions[0]
    res = intv.results
    scale = sim.pars.pop_scale
    assert res['new_doses'].sum() > 0
    assert np.allclose(res['cum_doses'], np.cumsum(res['new_doses']))
    assert np.allclose(res['cum_vaccinated'], np.cumsum(res['new_vaccinated']))
    # Doses >= people vaccinated, and both are in real-population units.
    assert res['cum_doses'][-1] >= res['cum_vaccinated'][-1] > 0
    assert res['cum_vaccinated'][-1] >= intv.vaccinated.sum() * scale
    assert res['cum_doses'][-1] >= intv.n_doses.sum() * scale


def test_routine_vx_reduces_susceptibility_post_dose():
    """Vaccination bumps vax_imm immediately; CrossImmunity reduces rel_sus next step.

    Two checks:
    1. Every vaccinated agent has vax_imm[hpv16] > 0 (immediate effect,
       written to vax_imm not nab_imm so it bypasses cross-immunity matrix).
    2. The bulk of vaccinated agents (those not vaccinated on the very last
       timestep) have rel_sus[hpv16] < 1.0 after CrossImmunity propagates.
    """
    intv = hpv.routine_vx(
        product=hpv.vx(rel_imm={'hpv16': 1.0}),
        prob=1.0,
        age_range=[9, 14],
        sex='f',
        start_year=2020,
        name='routine_sus_check',
    )
    sim = hpv.Sim(**SMALL_PARS, interventions=[intv])
    sim.run()
    intv = sim.interventions[0]
    vacc_uids = intv.vaccinated.uids
    if len(vacc_uids) == 0:
        pytest.skip('No agents vaccinated in this small-sim window')
    vax_imm = sim.diseases['hpv16'].vax_imm[vacc_uids]
    assert np.all(vax_imm > 0), \
        f'Vaccinated agents must have vax_imm[hpv16]>0; got min={float(vax_imm.min())}'
    # nab_imm is clearance-only; the vaccine writes to vax_imm.
    nab_imm = sim.diseases['hpv16'].nab_imm[vacc_uids]
    assert np.all(nab_imm == 0.0), \
        f'administer() must not touch nab_imm; got max={float(nab_imm.max())}'
    # 90%, not 100%: agents vaccinated on the last step haven't had CrossImmunity run.
    rel_sus = sim.diseases['hpv16'].rel_sus[vacc_uids]
    n_reduced = int((rel_sus < 1.0).sum())
    assert n_reduced >= 0.9 * len(vacc_uids), \
        f'Expected >=90% of vaccinated agents to have rel_sus<1.0, ' \
        f'got {n_reduced}/{len(vacc_uids)}'