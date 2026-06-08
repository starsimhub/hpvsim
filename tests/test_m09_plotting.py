import matplotlib
matplotlib.use('Agg')  # headless, before pyplot import
import matplotlib.pyplot as plt
import numpy as np
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
