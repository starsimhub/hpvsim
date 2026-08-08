"""Rendering gates for hpv.plot_calibration.

Both need a real ``Calibration.calibrate()`` run to produce something to plot,
which is why they live here rather than in ``tests/test_plotting.py``.
"""
import matplotlib
matplotlib.use('Agg')  # headless, before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import sciris as sc

import hpvsim as hpv


def teardown_function():
    plt.close('all')


@pytest.mark.slow
def test_plot_calibration_data_vs_fit():
    edges = np.array([0., 30., 50., 100.])
    def make_sim(seed=1):
        ar = hpv.AgeResults(result_args=sc.objdict(
            cancers=sc.objdict(years=[2020], edges=edges)))
        return hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1990,
                       stop=2020, n_agents=600, rand_seed=seed, analyzers=[ar])
    target = pd.DataFrame({'0-30': [1.0], '30-50': [5.0], '50+': [3.0]},
                          index=pd.Index([2020.0], name='t'))
    calib = hpv.Calibration(
        make_sim(), calib_pars=dict(beta=dict(low=0.1, high=0.3)),
        data={'cancers': target}, total_trials=2, n_workers=1)
    calib.calibrate()
    assert 'beta' in calib.best_pars   # beta is a free (sampled) parameter
    fig = hpv.plot_calibration(calib)
    assert len(fig.axes) == 1            # one panel per target
    ax = fig.axes[0]
    # Single-year, age-stratified target -> age on the x-axis: exactly two
    # series (data + fit), one tick per age bin (3 bins for edges 0/30/50/100).
    assert len(ax.lines) == 2
    assert len(ax.get_xticks()) == 3
    assert ax.get_xlabel() == 'Age group'


@pytest.mark.slow
def test_plot_calibration_multiyear_uses_year_axis():
    edges = np.array([0., 50., 100.])
    def make_sim(seed=1):
        ar = hpv.AgeResults(result_args=sc.objdict(
            cancers=sc.objdict(years=[2010, 2015, 2020], edges=edges)))
        return hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1990,
                       stop=2020, n_agents=600, rand_seed=seed, analyzers=[ar])
    # Multi-year target (3 years x 2 age-bin columns).
    target = pd.DataFrame({'0-50': [2.0, 3.0, 4.0], '50+': [1.0, 2.0, 3.0]},
                          index=pd.Index([2010.0, 2015.0, 2020.0], name='t'))
    calib = hpv.Calibration(
        make_sim(), calib_pars=dict(beta=dict(low=0.1, high=0.3)),
        data={'cancers': target}, total_trials=2, n_workers=1)
    calib.calibrate()
    fig = hpv.plot_calibration(calib)
    ax = fig.axes[0]
    # Year on x-axis: one data + one fit series per age-bin column (2 cols -> 4 lines).
    assert len(ax.lines) == 4
    assert ax.get_xlabel() == 'Year'
