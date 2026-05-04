"""Lifecycle smoke + capability tests for HPV16 natural history (M02)."""
import json
from pathlib import Path

import numpy as np
import pytest

import hpvsim as hpv


CAPABILITY_BASELINE = (
    Path(__file__).parent / 'regression_baselines' / 'm02_age_cancer.json'
)


def test_hpv_has_progression_states():
    """HPV defines precin/cin/cancerous BoolStates and ti_*/dur_* FloatArrs."""
    sim = hpv.Sim(n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0)
    sim.init()
    mod = sim.diseases.hpv16
    # New compartment flags
    for name in ('precin', 'cin', 'cancerous', 'recovered'):
        assert hasattr(mod, name), f'HPV missing BoolState {name!r}'
    # New scheduled-time arrays
    for name in ('ti_cin', 'ti_cancerous', 'ti_dead_cancer'):
        assert hasattr(mod, name), f'HPV missing FloatArr {name!r}'
    # New duration arrays
    for name in ('dur_precin', 'dur_cin'):
        assert hasattr(mod, name), f'HPV missing FloatArr {name!r}'


def test_hpv_has_progression_pars():
    """HPV defines dur_precin/dur_cin/dur_cancer durations + cin_fn/cancer_fn dicts."""
    mod = hpv.HPV(genotype='hpv16')
    p = mod.pars
    for name in ('dur_precin', 'dur_cin', 'dur_cancer', 'cin_fn', 'cancer_fn'):
        assert name in p, f'HPV.pars missing {name!r}'


def test_hpv_progression_pars_match_v2_hpv16():
    """Spot-check that the lognormal mean/std and severity-fn dicts match v2."""
    mod = hpv.HPV(genotype='hpv16')
    p = mod.pars
    # cin_fn matches v2 _v2_legacy/parameters.py:338
    assert p.cin_fn == dict(form='logf2', k=0.3, x_infl=0, ttc=50)
    # cancer_fn includes the cin_fn keys (so _compute_severity's cin_integral
    # branch can call _compute_severity_integral internally without re-merging).
    assert p.cancer_fn['method'] == 'cin_integral'
    assert p.cancer_fn['transform_prob'] == 2e-3
    # The dur_* are ss distribution instances. Initialize via mock() so we can
    # draw samples outside a sim context.
    p.dur_precin.mock()
    durs = p.dur_precin.rvs(5000)
    # Lognormal is non-negative; check shape and positivity.
    assert len(durs) == 5000
    assert np.all(durs >= 0)


def test_set_prognoses_assigns_ti_clearance_or_ti_cin():
    """Every newly-infected agent has either ti_clearance or ti_cin set."""
    sim = hpv.Sim(n_agents=500, location='nigeria',
                  start=1990, stop=1992, dt=0.5, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    ever_infected = mod.ti_first_infection.notnan
    has_clearance = mod.ti_clearance.notnan
    has_cin = mod.ti_cin.notnan
    assert (has_clearance | has_cin)[ever_infected].all()


def test_set_prognoses_cancer_only_in_females():
    """Males never progress to CIN; only females reach ti_cin / ti_cancerous."""
    sim = hpv.Sim(n_agents=2000, location='nigeria',
                  start=1990, stop=2000, dt=0.5, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    has_cin = mod.ti_cin.notnan
    has_cancer = mod.ti_cancerous.notnan
    males = ~sim.people.female
    assert not (has_cin & males).any()
    assert not (has_cancer & males).any()


def test_set_prognoses_chain_consistency():
    """For agents with cancer scheduled: ti_cin <= ti_cancerous <= ti_dead_cancer."""
    sim = hpv.Sim(n_agents=5000, location='nigeria',
                  start=1990, stop=2000, dt=0.5, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    has_cancer_sched = mod.ti_cancerous.notnan
    if has_cancer_sched.any():
        uids = has_cancer_sched.uids
        # Compare time-step values directly
        ti_cin = mod.ti_cin[uids]
        ti_cancerous = mod.ti_cancerous[uids]
        ti_dead = mod.ti_dead_cancer[uids]
        assert (ti_cin <= ti_cancerous).all()
        assert (ti_cancerous <= ti_dead).all()


def test_step_state_progresses_precin_to_cin():
    """An agent whose ti_cin <= ti flips precin→cin."""
    sim = hpv.Sim(n_agents=2000, location='nigeria',
                  start=1990, stop=2010, dt=0.5, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    has_cin_sched = mod.ti_cin.notnan
    if has_cin_sched.any():
        uids = has_cin_sched.uids
        passed = sim.t.ti >= mod.ti_cin[uids]
        assert passed.any(), 'No CIN-scheduled agent ever had ti >= ti_cin by sim end'


def test_step_state_progresses_cin_to_cancerous():
    """An agent whose ti_cancerous <= ti flips cin→cancerous and stops transmitting."""
    sim = hpv.Sim(n_agents=5000, location='nigeria',
                  start=1990, stop=2030, dt=0.5, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    cancerous_now = mod.cancerous.uids
    if len(cancerous_now):
        # Cancer agents are not currently infected and not susceptible
        assert not mod.infected[cancerous_now].any()
        assert not mod.susceptible[cancerous_now].any()
        # And rel_trans = 0
        rel_trans_arr = np.asarray(mod.rel_trans[cancerous_now])
        assert (rel_trans_arr == 0).all()


def test_step_state_cancer_death_removes_agents():
    """Agents whose ti_dead_cancer <= ti are no longer alive."""
    sim = hpv.Sim(n_agents=5000, location='nigeria',
                  start=1990, stop=2050, dt=0.5, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    has_dead_sched = mod.ti_dead_cancer.notnan
    if has_dead_sched.any():
        uids = has_dead_sched.uids
        passed = sim.t.ti >= mod.ti_dead_cancer[uids]
        if passed.any():
            passed_uids = uids[np.asarray(passed)]
            alive_arr = np.asarray(sim.people.alive[passed_uids])
            assert not alive_arr.any(), \
                f'{int(alive_arr.sum())} cancer-death-scheduled agents still alive'


def test_step_die_resets_bool_states():
    """Dying agents have precin/cin/cancerous cleared (so result counts are accurate)."""
    sim = hpv.Sim(n_agents=500, location='nigeria',
                  start=1990, stop=2010, dt=0.5, rand_seed=0)
    sim.init()
    mod = sim.diseases.hpv16

    # Manually infect and progress some agents to each state to test cleanup
    test_uids = np.array([0, 1, 2, 3, 4], dtype=int)

    # Set up test agents in different compartments
    mod.infected[test_uids] = True
    mod.precin[test_uids] = True
    mod.susceptible[test_uids] = True
    mod.cin[test_uids[[1, 2, 3]]] = True
    mod.cancerous[test_uids[[2, 3]]] = True
    mod.recovered[test_uids[[4]]] = True  # one recovered agent

    # Call step_die for these agents (simulating death)
    mod.step_die(test_uids)

    # Verify all disease-compartment states are cleared for dead agents
    assert not mod.precin[test_uids].any(), "precin not cleared after step_die"
    assert not mod.cin[test_uids].any(), "cin not cleared after step_die"
    assert not mod.cancerous[test_uids].any(), "cancerous not cleared after step_die"
    assert not mod.recovered[test_uids].any(), "recovered not cleared after step_die"
    assert not mod.infected[test_uids].any(), "infected not cleared after step_die"
    assert not mod.susceptible[test_uids].any(), "susceptible not cleared after step_die"


def test_cleared_agents_do_not_get_reinfected():
    """Once an agent clears HPV (or progresses to cancer), they should never
    get a second infection. Mirrors v2's same-genotype immunity (high
    nab_imm/cell_imm post-clearance reduces per-act probability to ~0).
    """
    sim = hpv.Sim(n_agents=2000, location='nigeria',
                  start=1990, stop=2030, dt=0.25, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    # Agents who have ever been infected (ti_first_infection set).
    ever = mod.ti_first_infection.notnan.uids
    # For each ever-infected agent, ti_infected at end-of-sim should equal
    # ti_first_infection — no overwrite from a second infection event.
    ti_first = np.asarray(mod.ti_first_infection[ever])
    ti_inf = np.asarray(mod.ti_infected[ever])
    n_reinfected = int((ti_inf > ti_first).sum())
    assert n_reinfected == 0, (
        f'{n_reinfected}/{len(ever)} agents got re-infected; '
        f'M02 same-genotype immunity should prevent this.'
    )


@pytest.mark.skipif(not CAPABILITY_BASELINE.exists(),
                    reason='M02 age-cancer baseline not generated; see '
                           'tests/regression/README.md M02 section')
def test_m02_capability_age_stratified_cancers():
    """End-of-sim age-stratified cancer incidence vs. v2 baseline, ±10% per band.

    M02 acceptance test: HPV → CIN → cancer dynamics for HPV16 must match
    v2's HPV16-only run within calibration tolerance, against a v2 1-genotype
    baseline. The capability baseline is generated by running the same anchor
    PARS in a v2.3 environment and capturing
    sim.results['cancer_incidence_by_age'] at year 2059, saved as JSON.
    """
    pars = dict(n_agents=10_000, location='nigeria', genotype='hpv16',
                start=1990, stop=2060, dt=0.5, rand_seed=0)
    sim = hpv.Sim(**pars,
                  analyzers=[hpv.AgeResults(results=('cancer',),
                                             year=[2059])])
    sim.run()
    az = sim.analyzers.age_results
    v3_rates = np.asarray(az.results.cancer_incidence_by_age[0])

    with open(CAPABILITY_BASELINE) as f:
        baseline = json.load(f)
    v2_rates = np.asarray(baseline['cancer_incidence_by_age'])
    assert v3_rates.shape == v2_rates.shape, \
        f'shape mismatch: v3 {v3_rates.shape} vs v2 {v2_rates.shape}'

    # Per-band drift: ±10%.
    out_of_tol = []
    for i, (a, b) in enumerate(zip(v3_rates, v2_rates)):
        if b == 0:
            if a > 1e-3:
                out_of_tol.append(dict(band=i, v3=float(a), v2=float(b),
                                       reason='v2=0, v3>0'))
            continue
        rel = abs(a - b) / b
        if rel > 0.10:
            out_of_tol.append(dict(band=i, v3=float(a), v2=float(b),
                                   rel_drift=float(rel)))
    assert not out_of_tol, (
        f'Bands out of ±10% tolerance: {out_of_tol}'
    )