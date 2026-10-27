"""Integration smoke tests for vaccination: ss.vaccinate() delivering hpv.vx.

These tests run small sims end-to-end (500 agents, ~7-year horizon) to verify
routine and campaign vaccination fire on schedule, reach the right age/sex
cohorts, record their results, and reduce susceptibility.
"""
import numpy as np
import pytest
import starsim as ss
import hpvsim as hpv


SMALL_PARS = dict(
    location='nigeria',
    start=2018, stop=2025,
    n_agents=500,
    genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
    rand_seed=0,
    verbose=0,
)

girls = ss.ppl.female & (ss.ppl.age >= 9) & (ss.ppl.age < 14)


def run_vx(delivery, **kwargs):
    """Run a sim with a vaccination delivery named "vx"; return it, and the agents vaccinated with their ages then"""
    reached = dict(uids=[], ages=[])
    def track(sim):
        uids = sim.interventions.vx.reached()
        reached['uids'].extend(uids.tolist())
        reached['ages'].extend(sim.people.age[uids].tolist())
    sim = hpv.Sim(**(SMALL_PARS | kwargs), interventions=[delivery], analyzers=track)
    sim.run()
    return sim, ss.uids(reached['uids']), np.array(reached['ages'])


def test_routine_vx_fires_once_per_person():
    """Routine vaccination reaches girls from its start, each once (no re-dosing while eligible)."""
    sim, uids, ages = run_vx(ss.vaccinate(hpv.vx('bivalent'), where=girls, coverage=0.9, start=2020, name='vx'))
    assert len(uids) > 0
    assert len(np.unique(uids)) == len(uids)
    assert np.isclose(sim.results.vx.delivered.sum(), len(uids)*sim.pars.pop_scale)


def test_campaign_vx_fires_on_each_round():
    """A campaign fires on each of its rounds."""
    women = ss.ppl.female & (ss.ppl.age >= 9) & (ss.ppl.age < 30)
    sim, uids, ages = run_vx(ss.vaccinate(hpv.vx('bivalent'), where=women, coverage=0.5, at=['2020', '2021'], name='vx'))
    delivered = np.asarray(sim.results.vx.delivered)
    assert (delivered > 0).sum() == 2


def test_routine_vx_respects_sex_and_age():
    """Only girls aged 9-14 when it fires are vaccinated."""
    sim, uids, ages = run_vx(ss.vaccinate(hpv.vx('bivalent'), where=girls, coverage=1.0, start=2020, name='vx'))
    assert len(uids) > 0
    assert np.all(sim.people.female[uids])
    assert np.all((ages >= 9) & (ages < 14))


def test_vx_at_age_reaches_girls_as_they_turn_nine():
    """at_age=9 reaches girls as they reach 9, and not those already older at the start."""
    sim, uids, ages = run_vx(ss.vaccinate(hpv.vx('bivalent'), where=ss.ppl.female, at_age=9, coverage=1.0, start=2020, name='vx'))
    assert len(uids) > 0
    assert np.all((ages >= 9) & (ages < 9 + 2*sim.t.dt.years))


def test_routine_vx_reduces_susceptibility_post_dose():
    """Vaccination bumps vax_imm immediately; CrossImmunity reduces rel_sus next step.

    Two checks:
    1. Every vaccinated agent has vax_imm[hpv16] > 0 (immediate effect,
       written to vax_imm not nab_imm so it bypasses cross-immunity matrix).
    2. The bulk of vaccinated agents (those not vaccinated on the very last
       timestep) have rel_sus[hpv16] < 1.0 after CrossImmunity propagates.
    """
    sim, uids, ages = run_vx(ss.vaccinate(hpv.vx(rel_imm={'hpv16': 1.0}), where=girls, coverage=1.0, start=2020, name='vx'))
    uids = uids[sim.people.alive[uids]]
    assert len(uids) > 0
    hpv16 = sim.diseases['hpv16']
    vax_imm = hpv16.vax_imm[uids]
    assert np.all(vax_imm > 0), f'Vaccinated agents must have vax_imm[hpv16]>0; got min={float(vax_imm.min())}'
    # nab_imm is clearance-only; the vaccine writes to vax_imm.
    assert np.all(hpv16.nab_imm[uids] == 0.0), 'administer() must not touch nab_imm'
    n_reduced = int((hpv16.rel_sus[uids] < 1.0).sum())
    assert n_reduced >= 0.9 * len(uids), f'Expected >=90% of vaccinated agents to have rel_sus<1.0, got {n_reduced}/{len(uids)}'
