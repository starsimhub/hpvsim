"""Unit tests for hpv.by_age analyzer."""
import numpy as np
import pytest

import hpvsim as hpv

# hpv.Sim deep-copies its inputs, so the live analyzer must be retrieved
# post-run via sim.analyzers['by_age'] rather than the passed-in reference.


def test_by_age_string_key():
    ar = hpv.by_age('cancers')
    assert ar.keys == ['cancers']
    assert list(ar.edges) == list(np.arange(0, 101, 5))


def test_by_age_list_of_keys():
    ar = hpv.by_age(['cancers', 'hpv_prevalence'], edges=[0, 20, 40, 60, 100])
    assert ar.keys == ['cancers', 'hpv_prevalence']
    assert list(ar.edges) == [0, 20, 40, 60, 100]
    assert ar.bin_labels == ['0-20', '20-40', '40-60', '60+']


def test_by_age_years_filter():
    ar = hpv.by_age('cancers', years=2020)
    assert ar.years == [2020]
    ar = hpv.by_age('cancers', years=[2015, 2020])
    assert ar.years == [2015, 2020]


def test_by_age_unknown_key_raises():
    with pytest.raises(ValueError, match='unknown'):
        hpv.by_age('bogus')


def test_by_age_missing_keys_raises():
    with pytest.raises(ValueError, match='at least one'):
        hpv.by_age()


def test_by_age_per_bin_results_scale_flag():
    """COUNT + FLOW keys register with scale=True; PREV with scale=False."""
    edges = [0., 20., 40., 60., 100.]
    sim = hpv.Sim(location='nigeria', n_agents=200, start=2000, stop=2001, dt=1.0,
                  analyzers=[hpv.by_age(['cancers', 'n_cancerous', 'hpv_prevalence'],
                                        edges=edges)])
    sim.init()
    ar = sim.analyzers['by_age']
    assert ar.results[ar._result_name('cancers', 0., 20.)].scale is True
    assert ar.results[ar._result_name('n_cancerous', 0., 20.)].scale is True
    assert ar.results[ar._result_name('hpv_prevalence', 0., 20.)].scale is False


def test_by_age_convenience_arrays_populated_after_run():
    """After run, self.<key> is a 2D array of shape (npts, n_bins)."""
    edges = [0., 25., 50., 75., 100.]
    n_bins = len(edges) - 1
    sim = hpv.Sim(location='nigeria', n_agents=200, start=2000, stop=2003, dt=1.0,
                  analyzers=[hpv.by_age(['cancers', 'hpv_prevalence'], edges=edges)])
    sim.run()
    ar = sim.analyzers['by_age']
    npts = len(sim.results.timevec)
    assert ar.cancers.shape == (npts, n_bins)
    assert ar.hpv_prevalence.shape == (npts, n_bins)
    assert (ar.hpv_prevalence >= 0).all()
    assert (ar.hpv_prevalence <= 1 + 1e-9).all()


def test_by_age_to_dataframe_returns_year_x_binlabels():
    edges = [0., 30., 60., 100.]
    sim = hpv.Sim(location='nigeria', n_agents=200, start=2000, stop=2003, dt=1.0,
                  analyzers=[hpv.by_age('cancers', edges=edges)])
    sim.run()
    ar = sim.analyzers['by_age']
    df = ar.to_dataframe('cancers')
    assert list(df.columns) == ['0-30', '30-60', '60+']
    assert df.index.name == 't'
    assert len(df) >= 3


def test_by_age_years_reporting_filter_applied():
    """years= at construction filters to_dataframe() output."""
    edges = [0., 50., 100.]
    sim = hpv.Sim(location='nigeria', n_agents=200, start=2000, stop=2004, dt=1.0,
                  analyzers=[hpv.by_age('cancers', years=[2001, 2003], edges=edges)])
    sim.run()
    ar = sim.analyzers['by_age']
    df = ar.to_dataframe('cancers')
    assert sorted(df.index.tolist()) == [2001.0, 2003.0]


def test_by_age_cancers_pop_scaled():
    """Per-bin cancers Result is scale=True; finalize multiplies by pop_scale."""
    sim = hpv.Sim(location='nigeria', total_pop=200_000, n_agents=2000,
                  start=1990, stop=2000, dt=1.0,
                  analyzers=[hpv.by_age('cancers', edges=[0., 40., 100.])])
    sim.run()
    ar = sim.analyzers['by_age']
    assert sim.pars.pop_scale == pytest.approx(100.0)
    r = sim.results.by_age.cancers_0_40
    assert r.scale is True
    assert (ar.cancers >= 0).all()


def test_by_age_cancers_matches_all_hpv_new_cancers():
    """Sum across bins of annualized ``cancers`` matches
    ``sim.results.all_hpv.new_cancers`` summed over each year's ticks."""
    edges = [0., 40., 100.]
    sim = hpv.Sim(location='nigeria', total_pop=200_000, n_agents=2000,
                  start=1990, stop=2001, dt=0.25, rand_seed=0,
                  analyzers=[hpv.by_age('cancers', edges=edges)])
    sim.run()
    ar = sim.analyzers['by_age']
    df = ar.to_dataframe('cancers')
    all_hpv = np.asarray(sim.results.all_hpv.new_cancers)
    tvy = sim.timevec.years
    for year in df.index:
        ticks = np.where((tvy >= year) & (tvy < year + 1))[0]
        expected = float(all_hpv[ticks].sum())
        actual = float(df.loc[year].sum())
        assert np.isclose(actual, expected, rtol=1e-6, atol=1e-6), (
            f'year {year}: by_age sum={actual}, all_hpv.new_cancers sum={expected}')


def test_by_age_demographic_denominators():
    """n_alive/n_females/n_males register scale=True, are internally
    consistent per bin, and sum across bins to the sim population."""
    edges = [0., 30., 60., 150.]
    sim = hpv.Sim(location='nigeria', total_pop=200_000, n_agents=2000,
                  start=2000, stop=2003, dt=1.0, rand_seed=0,
                  analyzers=[hpv.by_age(['n_alive', 'n_females', 'n_males'],
                                        edges=edges)])
    sim.run()
    ar = sim.analyzers['by_age']
    assert sim.results.by_age.n_alive_0_30.scale is True
    assert (ar.n_alive > 0).all()
    np.testing.assert_allclose(ar.n_females + ar.n_males, ar.n_alive)
    np.testing.assert_allclose(ar.n_alive.sum(axis=1),
                               np.asarray(sim.results.n_alive, dtype=float))
    female_frac = ar.n_females.sum() / ar.n_alive.sum()
    assert 0.4 < female_frac < 0.6


def test_by_age_hpv_prevalence_in_zero_one_range():
    """Whole-population prevalence bin stays in [0, 1] at every timestep."""
    edges = [0., 100.]  # single bin => whole population
    sim = hpv.Sim(location='nigeria', n_agents=500, start=2000, stop=2003, dt=1.0,
                  analyzers=[hpv.by_age('hpv_prevalence', edges=edges)])
    sim.run()
    ar = sim.analyzers['by_age']
    whole_pop_prev = ar.hpv_prevalence[:, 0]
    assert (whole_pop_prev >= 0).all() and (whole_pop_prev <= 1 + 1e-9).all()
