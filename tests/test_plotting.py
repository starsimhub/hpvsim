import matplotlib
matplotlib.use('Agg')  # headless, before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import hpvsim as hpv

# Age-bin edges shared by the by_age recorders and the by-age assertions
# (5 edges -> 4 bins -> 4 age ticks).
EDGES = np.array([0., 15., 30., 50., 100.])


@pytest.fixture(scope='module')
def rich_sim():
    """One 2-genotype, single-scale sim with a rich analyzer stack, run once and
    shared (read-only) across the structure-only plotting tests.

    Recording ``cancers`` / ``hpv_prevalence`` / ``cancerous_genotype_dist`` in
    by_age plus ``age_pyramid`` / ``age_causal_infection`` / ``dalys`` covers
    every plot_* entry point except the intervention-impact pair (needs a
    scenario arm) and the two ``plot_sim`` negative cases (need a sim that lacks
    the required by_age keys). 6000 agents at dt=0.5 keeps enough cancer
    signal for cancerous_genotype_dist and the dalys/age_causal panels — a wide,
    coarse run is cheaper than a narrow, fine one for the same signal, since
    cost is dominated by a fixed per-timestep overhead.
    """
    ar = hpv.by_age(['cancers', 'hpv_prevalence'],
                    years=[2015, 2025], edges=EDGES)
    sim = hpv.Sim(genotypes=['hpv16', 'hpv18'], location='nigeria', start=2000,
                  stop=2030, dt=0.5, n_agents=6000, rand_seed=1,
                  analyzers=[ar,
                             hpv.age_pyramid(timepoints=[2010]),
                             hpv.age_causal_infection(start=2000),
                             hpv.dalys(start=2000)])
    sim.run()
    return sim


def teardown_function():
    plt.close('all')


def test_plot_by_age_one_series_per_year(rich_sim):
    ar = rich_sim.analyzers['by_age']
    fig = hpv.plot_by_age(ar, 'cancers')
    ax = fig.axes[0]
    # One line per recorded year (2015, 2025).
    assert len(ax.lines) == 2
    # X axis has one tick per age bin (4 bins for the 5 edges above).
    assert len(ax.get_xticks()) == 4
    # Subsetting years works.
    fig2 = hpv.plot_by_age(ar, 'cancers', years=[2025])
    assert len(fig2.axes[0].lines) == 1


def test_plot_by_genotype_line_per_genotype(rich_sim):
    fig = hpv.plot_by_genotype(rich_sim, key='cum_cancers')
    ax = fig.axes[0]
    assert len(ax.lines) == 2  # one line per genotype
    # Normalized variant uses stacked areas (one collection per genotype).
    fig2 = hpv.plot_by_genotype(rich_sim, key='cum_cancers', normalize=True)
    assert len(fig2.axes[0].collections) == 2


def test_plot_type_distribution_bars_sum_to_one(rich_sim):
    fig = hpv.plot_type_distribution(rich_sim)        # Sim source, last year
    ax = fig.axes[0]
    heights = [p.get_height() for p in ax.patches]
    assert len(heights) == 2                          # one bar per genotype
    assert np.isclose(sum(heights), 1.0)              # normalized shares


# test_plot_type_distribution_age_results_source removed:
# by_age no longer supports type-distribution keys; plot_type_distribution
# takes only an hpv.Sim source now (delegates to results_by_genotype).


def test_plot_sim_default_four_panels_and_requires_age_results(rich_sim):
    fig = hpv.plot_sim(rich_sim, which='default')
    assert len(fig.axes) == 4
    # 'all' delegates to ss.Sim.plot and still returns a Figure.
    assert hpv.plot_sim(rich_sim, which='all') is not None
    # Without an by_age analyzer, 'default' raises a clear error.
    bare = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1995,
                   stop=2000, dt=1.0, n_agents=200, rand_seed=1)
    bare.run()
    with pytest.raises(ValueError):
        hpv.plot_sim(bare, which='default')


def test_plot_sim_default_raises_when_age_results_lacks_keys():
    # by_age present but recording neither 'cancers' nor a prevalence key.
    edges = np.array([0., 50., 100.])
    ar = hpv.by_age('n_cancerous', edges=edges)
    sim = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=2000, stop=2006,
                  dt=1.0, n_agents=200, rand_seed=1, analyzers=[ar])
    sim.run()
    with pytest.raises(ValueError):
        hpv.plot_sim(sim, which='default')


def test_plot_intervention_impact_averted_identity():
    # Asserts an algebraic identity (averted == baseline - scenario) + panel
    # count, so it holds for any values — a small/short annual-step sim suffices.
    base = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1995,
                   stop=2015, dt=1.0, n_agents=300, rand_seed=1)
    base.run()
    # Scenario with routine vaccination from 2000.
    vx = hpv.routine_vx(product='bivalent', prob=0.9, age_range=[9, 10],
                        start_year=2000)
    scen = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1995,
                   stop=2015, dt=1.0, n_agents=300, rand_seed=1, interventions=[vx])
    scen.run()
    fig = hpv.plot_intervention_impact(base, scen, key='cum_cancers')
    assert len(fig.axes) == 2
    # Bottom-panel line is exactly baseline_total - scenario_total.
    from hpvsim.analyzers import results_by_genotype
    b = results_by_genotype(base, 'cum_cancers').sum(axis=1).values
    s = results_by_genotype(scen, 'cum_cancers').sum(axis=1).values
    averted = fig.axes[1].lines[0].get_ydata()
    assert np.allclose(averted, b - s)
    # Mismatched timevecs raise (different stop -> different timevec than base).
    other = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1995,
                    stop=2020, dt=1.0, n_agents=200, rand_seed=1)
    other.run()
    with pytest.raises(ValueError):
        hpv.plot_intervention_impact(base, other, key='cum_cancers')


def test_plot_intervention_impact_multisim_smoke():
    def make_msim(seeds, interventions=None):
        # Smoke test of band rendering (collection counts) — small/short is fine.
        sims = [hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1995,
                        stop=2015, dt=1.0, n_agents=300, rand_seed=s,
                        interventions=interventions or [])
                for s in seeds]
        msim = ss.MultiSim(sims)
        msim.run(parallel=False)  # 4 tiny sims: process startup costs more than the runs
        return msim
    base = make_msim([1, 2])
    vx = hpv.routine_vx(product='bivalent', prob=0.9, age_range=[9, 10],
                        start_year=2000)
    scen = make_msim([1, 2], interventions=[vx])
    fig = hpv.plot_intervention_impact(base, scen, key='cum_cancers')
    assert len(fig.axes) == 2
    # Each arm renders a 10/90 band (one fill_between collection per arm).
    assert len(fig.axes[0].collections) == 2


def test_analyzer_plot_methods_return_figures(rich_sim):
    f1 = rich_sim.analyzers['age_pyramid'].plot()
    f2 = rich_sim.analyzers['age_causal_infection'].plot()
    f3 = rich_sim.analyzers['dalys'].plot()
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
