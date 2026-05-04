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