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
