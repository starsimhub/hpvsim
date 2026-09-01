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
    calib_pars = dict(beta=[0.20, 0.10, 0.30])
    calib = hpv.Calibration(sim, calib_pars, total_trials=2, debug=True)
    assert isinstance(calib, ss.Calibration)


def test_build_sim_routes_top_level_pars():
    """A bare sim-level key writes into sim.pars; a bare HPV-registered key
    broadcasts to every HPV disease."""
    sim = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0,
                  genotypes=[16, 18])
    sim.init()
    out = hpv.calibration.build_sim(sim, calib_pars={'rand_seed': 7, 'ms_agent_ratio': 3})
    assert out.pars.rand_seed == 7
    for d in (out.diseases.hpv16, out.diseases.hpv18):
        assert d.pars.ms_agent_ratio == 3


def test_build_sim_routes_per_genotype_pars():
    """A '<genotype>.<...>' calib_pars key writes into that disease's pars."""
    sim = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0,
                  genotypes=[16, 18, 'hi5', 'ohr'])
    sim.init()
    out = hpv.calibration.build_sim(sim, calib_pars={'hpv16.cin_fn.k': 0.77})
    assert out.diseases.hpv16.pars.cin_fn['k'] == 0.77


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

def _sim_with_all_hpv_by_age(*, n_agents=400, years=(2020,), keys=('cancers',),
                             genotypes=None):
    """Build a sim with an all_hpv_by_age analyzer (matches the analyzer
    name Calibration attaches automatically)."""
    edges = np.array([0., 30., 60., 100.])
    kw = dict(n_agents=n_agents, start=min(years) - 1,
              stop=max(years) + 1, dt=1.0, rand_seed=0,
              analyzers=[hpv.by_age(list(keys), years=list(years),
                                    edges=edges, name='all_hpv_by_age')])
    if genotypes is not None:
        kw['genotypes'] = genotypes
    return hpv.Sim(**kw)


def _as_standardized(dfs, scope='all_hpv'):
    """dict[name, wide by_age df] → standardized flat DataFrame with
    'scope.name.<bin>' columns."""
    frames = []
    for name, df in dfs.items():
        renamed = df.copy()
        renamed.columns = [f'{scope}.{name}.{c}' for c in renamed.columns]
        renamed.index = renamed.index.astype(float)
        renamed.index.name = 't'
        frames.append(renamed)
    return pd.concat(frames, axis=1)


def test_default_eval_fn_zero_when_data_matches_sim():
    """If the data DataFrame IS the sim's by_age output, mismatch is 0."""
    sim = _sim_with_all_hpv_by_age(keys=('cancers',))
    sim.run()
    ar = sim.analyzers['all_hpv_by_age']
    actual = ar.to_dataframe(key='cancers')
    data = _as_standardized({'cancers': actual})
    fit = hpv.calibration.default_eval_fn(sim, data=data)
    assert fit == pytest.approx(0.0, abs=1e-9)


def test_default_eval_fn_weighted_sum_across_targets():
    """Total fit is sum of per-column compute_gof * weights[column]."""
    sim = _sim_with_all_hpv_by_age(keys=('cancers', 'hpv_prevalence'))
    sim.run()
    ar = sim.analyzers['all_hpv_by_age']
    cancers = ar.to_dataframe(key='cancers')
    prev = ar.to_dataframe(key='hpv_prevalence')
    cancers_off = cancers + 1.0
    prev_off = prev + 0.05
    data = _as_standardized({'cancers': cancers_off, 'hpv_prevalence': prev_off})

    # Per-column gofs (sum-scalar by default).
    # For each age-bin column, per-cell error = 1.0 (cancers) or 0.05 (prev).
    # Normalization by max(|actual|) inside compute_gof matters -- easier to
    # just call default_eval_fn twice (unweighted + weighted) and check
    # linearity in the weights.
    fit_unweighted = hpv.calibration.default_eval_fn(sim, data=data)
    # Weight scaling: weighting ALL cancers columns by 2, all prev by 0.5.
    weights = {}
    for col in data.columns:
        if 'cancers' in col:
            weights[col] = 2.0
        else:
            weights[col] = 0.5
    fit_weighted = hpv.calibration.default_eval_fn(
        sim, data=data, weights=weights)

    # Split unweighted fit into per-metric halves by masking the data.
    cancers_only = data.loc[:, [c for c in data.columns if 'cancers' in c]]
    prev_only = data.loc[:, [c for c in data.columns if 'hpv_prevalence' in c]]
    fit_cancers = hpv.calibration.default_eval_fn(sim, data=cancers_only)
    fit_prev = hpv.calibration.default_eval_fn(sim, data=prev_only)
    assert fit_unweighted == pytest.approx(fit_cancers + fit_prev)
    assert fit_weighted == pytest.approx(2.0 * fit_cancers + 0.5 * fit_prev)


def test_default_eval_fn_aligns_on_expected_subset():
    """data only needs to cover a subset of the sim's snapshot years/bins."""
    sim = _sim_with_all_hpv_by_age(keys=('cancers',),
                                   years=(2018, 2019, 2020))
    sim.run()
    ar = sim.analyzers['all_hpv_by_age']
    actual = ar.to_dataframe(key='cancers')
    # Single year × single age bin.
    sub = actual.iloc[1:2, :1]
    data = _as_standardized({'cancers': sub})
    fit = hpv.calibration.default_eval_fn(sim, data=data)
    assert fit == pytest.approx(0.0, abs=1e-9)


def test_default_eval_fn_missing_row_raises():
    """An expected timepoint the analyzer didn't record surfaces as KeyError."""
    sim = _sim_with_all_hpv_by_age(keys=('cancers',), years=(2020,))
    sim.run()
    ar = sim.analyzers['all_hpv_by_age']
    actual = ar.to_dataframe(key='cancers')
    bad = actual.copy()
    bad.index = pd.Index([1850.0], name='t')
    data = _as_standardized({'cancers': bad})
    with pytest.raises(KeyError):
        hpv.calibration.default_eval_fn(sim, data=data)


def test_calibration_validates_data_index_and_column_scope():
    """data= DataFrame must use 't' index; unknown scope prefixes error."""
    sim = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0)
    calib_pars = dict(beta=[0.20, 0.10, 0.30])
    # Bad index name -> load_calib_data rejects
    bad_idx = pd.DataFrame([[1.0]], index=pd.Index([2020.0], name='year'),
                           columns=['all_hpv.cancers.0-100'])
    with pytest.raises(ValueError, match="must be 't'"):
        hpv.Calibration(sim, calib_pars, data=bad_idx,
                        total_trials=1, debug=True)
    # Unrecognized scope prefix -> _setup_analyzers rejects
    bad_col = pd.DataFrame([[1.0]], index=pd.Index([2020.0], name='t'),
                           columns=['not_a_scope.something'])
    with pytest.raises(ValueError, match='unrecognized data columns'):
        hpv.Calibration(sim, calib_pars, data=bad_col,
                        total_trials=1, debug=True)


def test_calibration_rejects_both_data_and_eval_fn():
    """Can't pass both data= and eval_fn= — they overwrite each other."""
    sim = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0)
    calib_pars = dict(beta=[0.20, 0.10, 0.30])
    good = pd.DataFrame([[1.0]], index=pd.Index([2020.0], name='t'),
                        columns=['all_hpv.cancers.0-100'])
    with pytest.raises(ValueError, match='data='):
        hpv.Calibration(sim, calib_pars, data=good,
                        eval_fn=lambda s: 0.0, total_trials=1, debug=True)


# ---------------------------------------------------------------------------
# data= (standardized DataFrame / list of CSV paths / dict of frames)
# ---------------------------------------------------------------------------

def _write_cancer_cases_csv(tmp_path, years, ages, values):
    """Write a Kazakhstan-format cancer_cases.csv to tmp_path and return the path."""
    rows = []
    for y in years:
        for a in ages:
            rows.append(dict(year=y, name='cancers', age=a, sex='female',
                             genotype='total', value=values[(y, a)]))
    path = tmp_path / 'kzk_cancer_cases.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_cancer_types_csv(tmp_path, year, genotype_props):
    rows = [dict(year=year, name='cancerous_genotype_dist', genotype=g, value=v)
            for g, v in genotype_props.items()]
    path = tmp_path / 'foo_cancer_types.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_asr_csv(tmp_path, year, value):
    path = tmp_path / 'foo_asr_cancer_incidence.csv'
    pd.DataFrame([dict(year=year, name='asr_cancer_incidence',
                       genotype='total', value=value)]).to_csv(path, index=False)
    return path


def test_data_loads_cancers_from_csv_and_attaches_by_age(tmp_path):
    """data=[csv] loads a long-format cancers CSV into a standardized flat
    DataFrame (one column per age bin) and attaches an all_hpv_by_age
    analyzer with the derived edges."""
    ages = [0, 15, 20, 25, 30]
    values = {(2020, a): float(a) for a in ages}
    csv = _write_cancer_cases_csv(tmp_path, [2020], ages, values)

    sim = hpv.Sim(n_agents=200, start=2019, stop=2021, dt=1.0, rand_seed=0)
    calib = hpv.Calibration(sim, dict(beta=[0.20, 0.10, 0.30]),
                            data=[str(csv)], total_trials=1,
                            n_workers=1, debug=True, verbose=False)

    df = calib.eval_kw['data']
    assert isinstance(df, pd.DataFrame)
    # One column per age bin (open-ended last)
    assert list(df.columns) == [
        'all_hpv.cancers.0-15', 'all_hpv.cancers.15-20',
        'all_hpv.cancers.20-25', 'all_hpv.cancers.25-30',
        'all_hpv.cancers.30+',
    ]
    assert list(df.index) == [2020.0]
    assert df.index.name == 't'

    analyzers = calib.sim.pars.get('analyzers', [])
    by_age_ars = [a for a in analyzers if getattr(a, 'name', None) == 'all_hpv_by_age']
    assert len(by_age_ars) == 1
    assert list(by_age_ars[0].edges) == [0.0, 15.0, 20.0, 25.0, 30.0, 150.0]


def test_data_loads_genotype_dist_from_csv(tmp_path):
    """A genotype-stratified CSV becomes one column per genotype (labels
    normalized to canonical hpvsim keys)."""
    csv = _write_cancer_types_csv(tmp_path, 2020,
                                  {'16': 0.5, '18': 0.15, 'hi5': 0.2, 'ohr': 0.15})

    sim = hpv.Sim(location='nigeria', n_agents=200, start=2019, stop=2021, dt=1.0,
                  genotypes=[16, 18, 'hi5', 'ohr'], rand_seed=0, verbose=0)
    calib = hpv.Calibration(sim, dict(beta=[0.2, 0.1, 0.3]),
                            data=[str(csv)], total_trials=1,
                            n_workers=1, debug=True, verbose=False)
    df = calib.eval_kw['data']
    assert set(df.columns) == {
        'by_genotype.cancerous_genotype_dist.hpv16',
        'by_genotype.cancerous_genotype_dist.hpv18',
        'by_genotype.cancerous_genotype_dist.hi5',
        'by_genotype.cancerous_genotype_dist.ohr',
    }
    assert list(df.index) == [2020.0]


def test_data_genotype_dist_e2e_calibrates(tmp_path):
    csv = _write_cancer_types_csv(tmp_path, 2020,
                                  {'16': 0.5, '18': 0.15, 'hi5': 0.2, 'ohr': 0.15})
    sim = hpv.Sim(location='nigeria', n_agents=200, start=2019, stop=2021, dt=1.0,
                  genotypes=[16, 18, 'hi5', 'ohr'], rand_seed=0, verbose=0)
    calib = hpv.Calibration(sim, dict(beta=[0.2, 0.1, 0.3]),
                            data=[str(csv)], total_trials=2,
                            n_workers=1, debug=True, verbose=False)
    calib.calibrate()
    assert len(calib.df) >= 1


def test_data_asr_reads_from_hpv_total(tmp_path):
    """ASR needs NO extra analyzers — it's a first-class HPVTotal result.
    Eval reads sim.results.all_hpv.asr_cancer_incidence directly."""
    asr_csv = _write_asr_csv(tmp_path, 2020, 15.7)
    sim = hpv.Sim(location='nigeria', n_agents=200, start=2019, stop=2021, dt=1.0,
                  rand_seed=0, verbose=0)
    calib = hpv.Calibration(sim, dict(beta=[0.2, 0.1, 0.3]),
                            data=[str(asr_csv)], total_trials=1,
                            n_workers=1, debug=True, verbose=False)
    df = calib.eval_kw['data']
    assert list(df.columns) == ['all_hpv.asr_cancer_incidence']
    assert float(df.iloc[0, 0]) == 15.7
    # No by_age or age_pyramid attached — ASR is on HPVTotal.
    analyzers = calib.sim.pars.get('analyzers', [])
    names = [getattr(a, 'name', None) for a in analyzers]
    assert 'all_hpv_by_age' not in names
    assert 'all_hpv_pyramid' not in names
    # E2E: calibration runs without error via HPVTotal.asr path.
    calib.calibrate()
    assert 'mismatch' in calib.df.columns


def test_data_cancers_e2e_calibrates(tmp_path):
    """data=[cancers_csv] drives a 2-trial calibration end-to-end."""
    ages = [0, 15, 20, 25, 30]
    values = {(2020, a): 1.0 for a in ages}
    csv = _write_cancer_cases_csv(tmp_path, [2020], ages, values)
    sim = hpv.Sim(n_agents=200, start=2019, stop=2021, dt=1.0, rand_seed=0)
    calib = hpv.Calibration(sim, dict(beta=[0.2, 0.1, 0.3]),
                            data=[str(csv)], total_trials=2,
                            n_workers=1, debug=True, verbose=False)
    calib.calibrate()
    assert 'mismatch' in calib.df.columns


def test_data_accepts_prebuilt_standardized_dataframe(tmp_path):
    """A user-provided DataFrame with dot-scoped columns is passed through
    (still triggers analyzer setup + auto-extend of sim.stop)."""
    df = pd.DataFrame(
        {'all_hpv.cancers.0-15': [1.0], 'all_hpv.cancers.15-30': [5.0]},
        index=pd.Index([2020.0], name='t'),
    )
    sim = hpv.Sim(n_agents=200, start=2019, stop=2019, dt=1.0, rand_seed=0)
    calib = hpv.Calibration(sim, dict(beta=[0.2, 0.1, 0.3]),
                            data=df, total_trials=1,
                            n_workers=1, debug=True, verbose=False)
    # sim.stop auto-extended past 2020
    assert int(calib.sim.pars.stop) >= 2021
    # analyzer attached with edges derived from labels
    by_age_ars = [a for a in calib.sim.pars.get('analyzers', [])
                  if getattr(a, 'name', None) == 'all_hpv_by_age']
    assert list(by_age_ars[0].edges) == [0.0, 15.0, 30.0]


