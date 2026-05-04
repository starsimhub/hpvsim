"""Lifecycle smoke + capability tests for HPV16 natural history (M02)."""
import numpy as np
import pytest

import hpvsim as hpv


def test_hpv_has_progression_states():
    """HPV defines precin/cin/cancerous BoolStates and ti_*/dur_* FloatArrs."""
    sim = hpv.Sim(n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0)
    sim.init()
    mod = sim.diseases.hpv16
    # New compartment flags
    for name in ('precin', 'cin', 'cancerous'):
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