"""Integration smoke tests for M05 vaccination interventions.

These tests run small sims end-to-end (200-500 agents, ~5-year horizon) to
verify the routine/campaign interventions fire on schedule, update
per-intervention state correctly, target the right age/sex cohorts, and
do not perturb the M03 no-vx baseline (CRN-stream guard).
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
    # At least some agents were vaccinated
    assert intv.vaccinated.sum() > 0
    # Every vaccinated agent has received at least 1 dose
    # (routine_vx fires each year agents remain eligible, so n_doses >= 1)
    assert np.all(intv.n_doses[intv.vaccinated.uids] >= 1)
    # ti_vaccinated set for every vaccinated agent
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
    # All vaccinated agents must be female
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
    # At sim end, vaccinated agents must have been 9 <= age < 10 at the time
    # of their first dose. Compute their age at vaccination given current
    # age and how many timesteps ago they were vaccinated.
    ages_now = sim.people.age[vacc_uids]
    ti_vacc = intv.ti_vaccinated[vacc_uids]
    # dt may be ss.dur — extract years
    dt = sim.t.dt.years if hasattr(sim.t.dt, 'years') else sim.t.dt
    ages_at_vacc = ages_now - (sim.ti - ti_vacc) * dt
    # Allow a small dt-rounding tolerance
    assert np.all(ages_at_vacc >= 9 - dt)
    assert np.all(ages_at_vacc < 10 + dt)


# Pinned scalar from a no-vx sim under SMALL_PARS at rand_seed=0.
# If M05 (or any later change) perturbs the RNG streams used by HPV
# transmission / progression / clearance, this number will change and
# the assertion below will fail loudly. Regenerate by running the
# no-vx sim and printing float(sim.results['hpvtotal']['cum_infections'].sum()).
EXPECTED_NO_VX_TOTAL_INFECTIONS = 10707.0


def test_no_vx_baseline_unchanged():
    """A no-vx sim reproduces the pre-M05 baseline value (CRN stream guard).

    Pure-determinism would also produce identical values between two runs,
    but pinning against a pre-recorded scalar guards against M05 (or any
    later change) perturbing the RNG streams used by the M03 pipeline.
    """
    sim = hpv.Sim(**SMALL_PARS)
    sim.run()
    got = float(sim.results['hpvtotal']['cum_infections'].sum())
    assert got == pytest.approx(EXPECTED_NO_VX_TOTAL_INFECTIONS), \
        f'No-vx baseline drifted: got {got!r}, expected {EXPECTED_NO_VX_TOTAL_INFECTIONS!r}. ' \
        f'Either M05 perturbed RNG streams (investigate), or the underlying ' \
        f'model genuinely changed (regenerate the pinned value).'


def test_routine_vx_reduces_susceptibility_post_dose():
    """Vaccination bumps nab_imm immediately; CrossImmunity reduces rel_sus next step.

    Two checks:
    1. Every vaccinated agent has nab_imm[hpv16] > 0 (immediate effect).
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
    # 1. Immediate effect: nab_imm bumped for every vaccinated agent
    nab_imm = sim.diseases['hpv16'].nab_imm[vacc_uids]
    assert np.all(nab_imm > 0), \
        f'Vaccinated agents must have nab_imm[hpv16]>0; got min={float(nab_imm.min())}'
    # 2. Eventual effect: CrossImmunity reduces rel_sus for the bulk of
    #    vaccinated agents (allow up to ~5% latency for agents vaccinated
    #    on the very last few timesteps).
    rel_sus = sim.diseases['hpv16'].rel_sus[vacc_uids]
    n_reduced = int((rel_sus < 1.0).sum())
    assert n_reduced >= 0.9 * len(vacc_uids), \
        f'Expected >=90% of vaccinated agents to have rel_sus<1.0, ' \
        f'got {n_reduced}/{len(vacc_uids)}'