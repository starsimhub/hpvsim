"""Unit tests for hpv.Calibration, compute_gof, and the default eval_fn."""
import numpy as np
import pandas as pd
import pytest
import sciris as sc
import starsim as ss

import hpvsim as hpv


# ---------------------------------------------------------------------------
# Calibration class / build_sim
# ---------------------------------------------------------------------------

def test_calibration_importable():
    """hpv.Calibration exists at top level and is an ss.Calibration."""
    sim = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0)
    calib_pars = dict(beta=dict(low=0.10, high=0.30, guess=0.20))
    calib = hpv.Calibration(sim, calib_pars, total_trials=2, debug=True)
    assert isinstance(calib, ss.Calibration)


def test_build_sim_routes_top_level_pars():
    """A bare-name calib_pars key writes into sim.pars."""
    sim = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0)
    sim.init()
    out = hpv.calibration.build_sim(sim, calib_pars={'beta': 0.25})
    assert out.pars.beta == 0.25


def test_build_sim_routes_per_genotype_pars():
    """A '<genotype>.<...>' calib_pars key writes into that disease's pars."""
    sim = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0,
                  genotypes=[16, 18, 'hi5', 'ohr'])
    sim.init()
    out = hpv.calibration.build_sim(sim, calib_pars={'hpv16.cin_fn.k': 0.77})
    assert out.diseases.hpv16.pars.cin_fn['k'] == 0.77


def test_build_sim_routes_cross_immunity():
    """A 'cross_immunity.<matrix>.<tgt>.<src>' key writes into the connector matrix."""
    sim = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0,
                  genotypes=[16, 18, 'hi5', 'ohr'])
    sim.init()
    out = hpv.calibration.build_sim(
        sim, calib_pars={'cross_immunity.cross_imm_sus.hpv16.hpv18': 0.42})
    conn = [c for c in out.connectors.values()
            if isinstance(c, hpv.CrossImmunity)][0]
    idx = {m.name: i for i, m in enumerate(conn.hpv_modules)}
    assert conn.cross_imm_sus[idx['hpv16'], idx['hpv18']] == 0.42


def test_build_sim_raises_on_unknown_key():
    """An unrecognized calib_pars key raises ValueError."""
    sim = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0)
    sim.init()
    with pytest.raises(ValueError, match='unrecognized'):
        hpv.calibration.build_sim(sim, calib_pars={'notapar.foo': 1.0})


def test_build_sim_does_not_mutate_base():
    """build_sim mutates the passed sim — ss.Calibration's dcp is the
    no-mutate guarantee. This test confirms we mutate the *passed* sim, not
    something else, and that the test verifies the contract by passing dcp'd
    copies."""
    sim_base = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0)
    sim_base.init()
    original_seed = sim_base.pars.rand_seed
    sim_copy = sc.dcp(sim_base)
    hpv.calibration.build_sim(sim_copy, calib_pars={'rand_seed': 999})
    assert sim_base.pars.rand_seed == original_seed
    assert sim_copy.pars.rand_seed == 999


# ---------------------------------------------------------------------------
# compute_gof
# ---------------------------------------------------------------------------

def test_compute_gof_normalized_abs_error_default():
    """Default is per-point absolute error divided by max(|actual|)."""
    actual = np.array([0.0, 10.0, 20.0])
    predicted = np.array([0.0, 12.0, 18.0])
    gof = hpv.calibration.compute_gof(actual, predicted)
    # Errors |0-0|, |10-12|, |20-18| = [0, 2, 2]; max(|actual|) = 20.
    np.testing.assert_allclose(gof, [0.0, 0.1, 0.1])


def test_compute_gof_as_scalar_sum_returns_float():
    """as_scalar='sum' collapses the per-point array to a scalar."""
    gof = hpv.calibration.compute_gof([0, 10, 20], [0, 12, 18],
                                      as_scalar='sum')
    assert isinstance(gof, float)
    assert gof == pytest.approx(0.2)


def test_compute_gof_mse_via_use_squared_mean():
    """MSE = compute_gof(normalize=False, use_squared=True, as_scalar='mean')."""
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.5, 2.5, 3.5])
    gof = hpv.calibration.compute_gof(actual, predicted, normalize=False,
                                      use_squared=True, as_scalar='mean')
    assert gof == pytest.approx(0.25)


def test_compute_gof_zero_actual_max_does_not_divide():
    """If actual is all zero, the normalize step is a no-op (no divide-by-zero)."""
    gof = hpv.calibration.compute_gof([0, 0, 0], [1, 2, 3])
    np.testing.assert_array_equal(gof, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# default_eval_fn
# ---------------------------------------------------------------------------

def _sim_with_by_age(*, n_agents=400, years=(2020,), keys=('cancers',),
                         genotypes=None):
    edges = np.array([0., 30., 60., 100.])
    kw = dict(n_agents=n_agents, start=min(years) - 1,
              stop=max(years) + 1, dt=1.0, rand_seed=0,
              analyzers=[hpv.by_age(list(keys), years=list(years), edges=edges)])
    if genotypes is not None:
        kw['genotypes'] = genotypes
    return hpv.Sim(**kw)


def test_default_eval_fn_zero_when_data_matches_sim():
    """If the data dataframe IS the sim's by_age output, mismatch is 0."""
    sim = _sim_with_by_age(keys=('cancers',))
    sim.run()
    ar = sim.analyzers['by_age']
    actual = ar.to_dataframe(key='cancers')

    data = {'cancers': actual.copy()}
    fit = hpv.calibration.default_eval_fn(sim, data=data)
    assert fit == pytest.approx(0.0, abs=1e-9)


def test_default_eval_fn_weighted_sum_across_targets():
    """Total fit is sum of per-key compute_gof * weights[key]."""
    sim = _sim_with_by_age(keys=('cancers', 'hpv_prevalence'))
    sim.run()
    ar = sim.analyzers['by_age']
    cancers = ar.to_dataframe(key='cancers')
    prev = ar.to_dataframe(key='hpv_prevalence')

    # Construct off-target data so each key contributes a known mismatch.
    cancers_off = cancers + 1.0
    prev_off = prev + 0.05
    data = {'cancers': cancers_off, 'hpv_prevalence': prev_off}

    # Per-key gofs we expect default_eval_fn to compute (sum-scalar by default).
    gof_cancers = hpv.calibration.compute_gof(
        cancers_off.values.ravel(), cancers.values.ravel(), as_scalar='sum')
    gof_prev = hpv.calibration.compute_gof(
        prev_off.values.ravel(), prev.values.ravel(), as_scalar='sum')

    # Unweighted = sum of per-key gofs.
    fit_unweighted = hpv.calibration.default_eval_fn(sim, data=data)
    assert fit_unweighted == pytest.approx(gof_cancers + gof_prev)

    # Weighted scales each per-key contribution.
    fit_weighted = hpv.calibration.default_eval_fn(
        sim, data=data, weights={'cancers': 2.0, 'hpv_prevalence': 0.5})
    assert fit_weighted == pytest.approx(2.0 * gof_cancers + 0.5 * gof_prev)


# test_default_eval_fn_type_distribution_uses_raw_counts removed:
# by_age no longer supports type-distribution keys
# (cancerous_genotype_dist, cin_genotype_dist). Callers wanting per-genotype
# distribution should use `hpv.results_by_genotype(sim, key='cum_cancers')`.


def test_default_eval_fn_aligns_on_expected_subset():
    """data only needs to cover a subset of the sim's snapshot years/bins."""
    sim = _sim_with_by_age(keys=('cancers',),
                                years=(2018, 2019, 2020))
    sim.run()
    ar = sim.analyzers['by_age']
    actual = ar.to_dataframe(key='cancers')

    # Pick a single year + a single age bin.
    one_year = actual.index[1:2]
    one_col = actual.columns[:1]
    sub = actual.loc[one_year, one_col]
    data = {'cancers': sub.copy()}
    fit = hpv.calibration.default_eval_fn(sim, data=data)
    assert fit == pytest.approx(0.0, abs=1e-9)


def test_default_eval_fn_missing_row_raises():
    """An expected timepoint that by_age didn't record surfaces as KeyError."""
    sim = _sim_with_by_age(keys=('cancers',), years=(2020,))
    sim.run()
    ar = sim.analyzers['by_age']
    actual = ar.to_dataframe(key='cancers')

    bad_data = actual.copy()
    bad_data.index = pd.Index([1850.0], name='t')   # not in sim's years
    with pytest.raises(KeyError, match='1850'):
        hpv.calibration.default_eval_fn(sim, data={'cancers': bad_data})


def test_calibration_validates_data_keys_and_index():
    """data= must use by_age result keys and 't'-named indexes."""
    sim = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0)
    calib_pars = dict(beta=dict(low=0.10, high=0.30, guess=0.20))

    good_idx = pd.DataFrame([[1.0]], index=pd.Index([2020.0], name='t'),
                            columns=['0-100'])

    with pytest.raises(ValueError, match='unknown result key'):
        hpv.Calibration(sim, calib_pars, data={'not_a_result': good_idx},
                        total_trials=1, debug=True)

    bad_idx = pd.DataFrame([[1.0]], index=pd.Index([2020.0], name='year'),
                           columns=['0-100'])
    with pytest.raises(ValueError, match="must be 't'"):
        hpv.Calibration(sim, calib_pars, data={'cancers': bad_idx},
                        total_trials=1, debug=True)


def test_calibration_rejects_both_data_and_eval_fn():
    """Can't pass both data= and eval_fn= — they overwrite each other."""
    sim = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0)
    calib_pars = dict(beta=dict(low=0.10, high=0.30, guess=0.20))
    good = pd.DataFrame([[1.0]], index=pd.Index([2020.0], name='t'),
                        columns=['0-100'])
    with pytest.raises(ValueError, match='data='):
        hpv.Calibration(sim, calib_pars, data={'cancers': good},
                        eval_fn=lambda s: 0.0, total_trials=1, debug=True)


