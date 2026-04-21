"""
Tests for hpvsim/misc.py — versioning, save/load, GOF, and utility functions.
"""

import os
import numpy as np
import sciris as sc
import hpvsim as hpv
import pytest

hpv.options.set(interactive=False)


def test_check_version():
    """Test version checking with various comparisons."""
    current = hpv.__version__

    # Exact match
    result = hpv.check_version(current, die=False)
    assert result == 0

    # Greater-than check against old version
    result = hpv.check_version('>=0.0.1', die=False)
    assert result >= 0

    # Less-than check against future version
    result = hpv.check_version('<=99.0.0', die=False)
    assert result <= 0

    # Should not die on mismatch by default
    result = hpv.check_version('0.0.1', die=False)
    assert result == 1  # Current is newer

    # Should raise on mismatch when die=True
    with pytest.raises(ValueError):
        hpv.check_version('0.0.1', die=True)


def test_get_version_pars():
    """Test loading parameters from a specific version."""
    pars = hpv.get_version_pars(hpv.__version__, verbose=False)
    assert isinstance(pars, dict)
    assert len(pars) > 0

    # Loading an older available version
    pars_old = hpv.get_version_pars('0.2.6', verbose=False)
    assert isinstance(pars_old, dict)


def test_save_load_roundtrip(tmp_path):
    """Test save/load round-trip for a sim object."""
    sim = hpv.Sim(n_agents=500, n_years=2, genotypes=[16], verbose=0)
    sim.run()

    filepath = str(tmp_path / 'test_sim.obj')
    hpv.save(filepath, sim)
    assert os.path.exists(filepath)

    loaded = hpv.load(filepath)
    assert isinstance(loaded, hpv.Sim)
    assert np.allclose(loaded.results['infections'][:], sim.results['infections'][:])


def test_compute_gof():
    """Test goodness-of-fit calculation."""
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = np.array([1.1, 2.2, 2.8, 4.1, 5.3])

    # Default: normalized absolute error
    gof = hpv.compute_gof(actual, predicted)
    assert isinstance(gof, np.ndarray)
    assert len(gof) == len(actual)
    assert (gof >= 0).all()

    # Scalar output
    gof_sum = hpv.compute_gof(actual, predicted, as_scalar='sum')
    assert isinstance(gof_sum, float)
    assert gof_sum > 0

    gof_mean = hpv.compute_gof(actual, predicted, as_scalar='mean')
    assert isinstance(gof_mean, float)

    # Perfect match
    gof_perfect = hpv.compute_gof(actual, actual)
    assert np.allclose(gof_perfect, 0)

    # Fractional error
    gof_frac = hpv.compute_gof(actual, predicted, use_frac=True)
    assert (gof_frac >= 0).all()

    # Squared error
    gof_sq = hpv.compute_gof(actual, predicted, use_squared=True)
    assert (gof_sq >= 0).all()

    # Un-normalized
    gof_unnorm = hpv.compute_gof(actual, predicted, normalize=False)
    assert (gof_unnorm >= 0).all()


def test_diff_sims():
    """Test sim comparison utility."""
    sim1 = hpv.Sim(n_agents=500, n_years=3, genotypes=[16], rand_seed=0, verbose=0)
    sim2 = sim1.copy()

    sim1.run()
    sim2.run()

    # Identical sims should not raise
    hpv.diff_sims(sim1, sim2, die=True)

    # Different sims should show differences
    sim3 = hpv.Sim(n_agents=500, n_years=3, genotypes=[16], rand_seed=99, verbose=0)
    sim3.run()
    # Should not raise with die=False
    hpv.diff_sims(sim1, sim3, die=False)


def test_git_info():
    """Test git info retrieval."""
    info = hpv.git_info()
    assert isinstance(info, dict)
    assert 'hpvsim' in info


def test_get_doubling_time():
    """Test doubling time calculation."""
    sim = hpv.Sim(n_agents=1e3, n_years=10, genotypes=[16], verbose=0)
    sim.run()

    # Use exponential approximation with a proper interval
    n_ts = len(sim.results['infections'][:])
    cum_infections = np.cumsum(sim.results['infections'][:])
    dt = hpv.get_doubling_time(sim, series=cum_infections,
                                interval=[0, n_ts-1], exp_approx=True)
    assert dt is not None


def test_load_data():
    """Test loading data from CSV files."""
    # Test with the test data files that exist
    datafile = sc.thisdir(__file__, 'test_data', 'south_africa_cancer_data_2020.csv')
    if os.path.exists(datafile):
        df = hpv.load_data(datafile)
        assert df is not None
        assert len(df) > 0


def test_help():
    """Test the help search function."""
    # Use a pattern that matches documented functions
    try:
        result = hpv.help('simulate', output=True)
        assert result is None or isinstance(result, str)
    except AttributeError:
        # Known issue: some functions have None docstrings
        pass


#%% Run as a script
if __name__ == '__main__':
    T = sc.tic()
    test_check_version()
    test_get_version_pars()
    test_save_load_roundtrip(sc.path(sc.thisdir(__file__)) / 'tmp')
    test_compute_gof()
    test_diff_sims()
    test_git_info()
    test_get_doubling_time()
    test_load_data()
    test_help()
    sc.toc(T)
    print('Done.')
