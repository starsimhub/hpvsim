import matplotlib
matplotlib.use('Agg')  # headless, before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import hpvsim as hpv


def _sim_with_age_results(genotypes=('hpv16',), stop=2030):
    edges = np.array([0., 15., 30., 50., 100.])
    ar = hpv.AgeResults(result_args=sc.objdict(
        cancers=sc.objdict(years=[2015, 2025], edges=edges),
        hpv_prevalence=sc.objdict(years=[2015, 2025], edges=edges),
        cancerous_genotype_dist=sc.objdict(years=[2025], edges=edges),
    ))
    sim = hpv.Sim(genotypes=list(genotypes), location='nigeria', start=1990,
                  stop=stop, n_agents=1000, rand_seed=1, analyzers=[ar])
    sim.run()
    return sim, sim.analyzers['ageresults']


def teardown_function():
    plt.close('all')


def test_plot_by_age_one_series_per_year():
    _, ar = _sim_with_age_results()
    fig = hpv.plot_by_age(ar, 'cancers')
    ax = fig.axes[0]
    # One line per recorded year (2015, 2025).
    assert len(ax.lines) == 2
    # X axis has one tick per age bin (4 bins for the 5 edges above).
    assert len(ax.get_xticks()) == 4
    # Subsetting years works.
    fig2 = hpv.plot_by_age(ar, 'cancers', years=[2025])
    assert len(fig2.axes[0].lines) == 1


def test_plot_by_genotype_line_per_genotype():
    sim = hpv.Sim(genotypes=['hpv16', 'hpv18'], location='nigeria',
                  start=1990, stop=2030, n_agents=1000, rand_seed=1)
    sim.run()
    fig = hpv.plot_by_genotype(sim, key='cum_cancers')
    ax = fig.axes[0]
    assert len(ax.lines) == 2  # one line per genotype
    # Normalized variant uses stacked areas (one collection per genotype).
    fig2 = hpv.plot_by_genotype(sim, key='cum_cancers', normalize=True)
    assert len(fig2.axes[0].collections) == 2


def test_plot_type_distribution_bars_sum_to_one():
    sim = hpv.Sim(genotypes=['hpv16', 'hpv18'], location='nigeria',
                  start=1990, stop=2030, n_agents=1000, rand_seed=1)
    sim.run()
    fig = hpv.plot_type_distribution(sim)            # Sim source, last year
    ax = fig.axes[0]
    heights = [p.get_height() for p in ax.patches]
    assert len(heights) == 2                          # one bar per genotype
    assert np.isclose(sum(heights), 1.0)              # normalized shares


def test_plot_type_distribution_age_results_source():
    _, ar = _sim_with_age_results(genotypes=('hpv16', 'hpv18'))
    fig = hpv.plot_type_distribution(ar, key='cancerous_genotype_dist')
    ax = fig.axes[0]
    heights = [p.get_height() for p in ax.patches]
    assert len(heights) == 2                  # one bar per genotype
    assert np.isclose(sum(heights), 1.0)      # normalized shares


def test_plot_sim_default_four_panels_and_requires_age_results():
    sim, _ = _sim_with_age_results(genotypes=('hpv16', 'hpv18'))
    fig = hpv.plot_sim(sim, which='default')
    assert len(fig.axes) == 4
    # 'all' delegates to ss.Sim.plot and still returns a Figure.
    assert hpv.plot_sim(sim, which='all') is not None
    # Without an AgeResults analyzer, 'default' raises a clear error.
    bare = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1990,
                   stop=2010, n_agents=500, rand_seed=1)
    bare.run()
    with pytest.raises(ValueError):
        hpv.plot_sim(bare, which='default')


def test_plot_sim_default_raises_when_age_results_lacks_keys():
    # AgeResults present but recording neither 'cancers' nor a prevalence key.
    edges = np.array([0., 50., 100.])
    ar = hpv.AgeResults(result_args=sc.objdict(
        cin_incidence=sc.objdict(years=[2010], edges=edges)))
    sim = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1990, stop=2010,
                  n_agents=400, rand_seed=1, analyzers=[ar])
    sim.run()
    with pytest.raises(ValueError):
        hpv.plot_sim(sim, which='default')


def test_plot_intervention_impact_averted_identity():
    base = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1990,
                   stop=2030, n_agents=1000, rand_seed=1)
    base.run()
    # Scenario with routine vaccination from 2000.
    vx = hpv.routine_vx(product='bivalent', prob=0.9, age_range=[9, 10],
                        start_year=2000)
    scen = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1990,
                   stop=2030, n_agents=1000, rand_seed=1, interventions=[vx])
    scen.run()
    fig = hpv.plot_intervention_impact(base, scen, key='cum_cancers')
    assert len(fig.axes) == 2
    # Bottom-panel line is exactly baseline_total - scenario_total.
    from hpvsim.analyzers import results_by_genotype
    b = results_by_genotype(base, 'cum_cancers').sum(axis=1).values
    s = results_by_genotype(scen, 'cum_cancers').sum(axis=1).values
    averted = fig.axes[1].lines[0].get_ydata()
    assert np.allclose(averted, b - s)
    # Mismatched timevecs raise.
    other = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1990,
                    stop=2025, n_agents=300, rand_seed=1)
    other.run()
    with pytest.raises(ValueError):
        hpv.plot_intervention_impact(base, other, key='cum_cancers')


def test_plot_intervention_impact_multisim_smoke():
    def make_msim(seeds, interventions=None):
        sims = [hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1990,
                        stop=2025, n_agents=600, rand_seed=s,
                        interventions=interventions or [])
                for s in seeds]
        msim = ss.MultiSim(sims)
        msim.run()
        return msim
    base = make_msim([1, 2])
    vx = hpv.routine_vx(product='bivalent', prob=0.9, age_range=[9, 10],
                        start_year=2000)
    scen = make_msim([1, 2], interventions=[vx])
    fig = hpv.plot_intervention_impact(base, scen, key='cum_cancers')
    assert len(fig.axes) == 2
    # Each arm renders a 10/90 band (one fill_between collection per arm).
    assert len(fig.axes[0].collections) == 2


def test_plot_calibration_data_vs_fit():
    import pandas as pd
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


def test_plot_calibration_multiyear_uses_year_axis():
    import pandas as pd
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


def test_analyzer_plot_methods_return_figures():
    ap = hpv.age_pyramid(timepoints=[2010])
    aci = hpv.age_causal_infection(start=2000)
    dal = hpv.dalys(start=2000)
    sim = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1990, stop=2030,
                  n_agents=1500, rand_seed=1, analyzers=[ap, aci, dal])
    sim.run()
    f1 = sim.analyzers['age_pyramid'].plot()
    f2 = sim.analyzers['age_causal_infection'].plot()
    f3 = sim.analyzers['dalys'].plot()
    for f in (f1, f2, f3):
        assert f is not None and len(f.axes) >= 1
    # age_causal_infection.plot() shows 3 age + 3 dwell-time panels.
    assert len(f2.axes) == 6
    # Pyramid sign convention: male bars drawn on the left (negative width).
    assert min(p.get_width() for p in f1.axes[0].patches) < 0
    plt.close('all')


def test_plotting_exports():
    for name in ['plot_sim', 'plot_by_age', 'plot_by_genotype',
                 'plot_type_distribution', 'plot_intervention_impact',
                 'plot_calibration']:
        assert hasattr(hpv, name), f'hpv.{name} not exported'
