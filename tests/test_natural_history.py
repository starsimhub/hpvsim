"""
Test natural history validation against v2 baselines (GitHub issue #40).

Runs v3 with the same parameters used to generate v2 baselines and compares
rate-based metrics (HPV prevalence, cancer incidence) using overlapping
confidence intervals across seeds.

Builds on the regression test patterns from test_v2_regression.py, adapted
for the v3 Starsim-based API.
"""

import sciris as sc
import numpy as np
import hpvsim as hpv
import matplotlib.pyplot as plt
import pytest

from compare_baselines import (
    load_v2_baselines,
    extract_v3_results,
    compare_metric,
)

# Module-level constants
n_agents = 10_000
n_seeds = 5
do_plot = False
sc.options(interactive=False)

# Match v2 baseline generation parameters
locations = ['nigeria', 'india', 'south africa']
genotypes = [16, 18, 'hi5', 'ohr']

# Comparison window: post-burnin (v2 used burnin=20 from start=1990)
year_start = 2010
year_end = 2060

# CI overlap threshold — fraction of post-burnin time points that must overlap
overlap_threshold = 0.5


def make_sim(location='nigeria', seed=0, n_agents=n_agents):
    """
    Create a v3 sim with parameters matching the v2 baselines.

    V2 used: n_agents=10e3, start=1990, end=2060, dt=0.5, burnin=20.
    V3 equivalent: start=1990, stop=2060, dt=0.5 (no burnin concept).
    """
    sim = hpv.Sim(
        n_agents=n_agents,
        start=1990,
        stop=2060,
        dt=0.5,
        location=location,
        genotypes=genotypes,
        rand_seed=seed,
        verbose=0,
    )
    return sim


def run_seeds(location='nigeria', n_seeds=n_seeds, n_agents=n_agents):
    """Run v3 across multiple seeds for a location, returning extracted results."""
    results = []
    for seed in range(n_seeds):
        sim = make_sim(location=location, seed=seed, n_agents=n_agents)
        sim.run()
        results.append(extract_v3_results(sim))
    return results


@sc.timer()
def test_smoke(do_plot=do_plot):
    """Quick check that v3 produces non-trivial results for a natural history run."""
    sc.heading('Testing natural history smoke...')
    sim = make_sim(location='nigeria', seed=0)
    sim.run()
    res = extract_v3_results(sim)

    assert len(res['year']) > 0, 'Expected non-empty year array'
    assert np.any(res['hpv_prevalence'] > 0), 'Expected nonzero HPV prevalence'
    assert np.any(res['n_infectious'] > 0), 'Expected some infections'

    # Prevalence should be in a plausible range (0-50%)
    max_prev = np.max(res['hpv_prevalence'])
    assert 0 < max_prev < 0.5, f'Expected max prevalence between 0 and 0.5, got {max_prev:.4f}'

    if do_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('V3 Natural History Smoke Test — Nigeria')
        axes[0, 0].plot(res['year'], res['hpv_prevalence'])
        axes[0, 0].set_title('HPV Prevalence')
        axes[0, 0].set_ylabel('Proportion')
        axes[0, 1].plot(res['year'], res['cancer_incidence'])
        axes[0, 1].set_title('Cancer Incidence (per 100k)')
        axes[0, 1].set_ylabel('Per 100k women 15+')
        axes[1, 0].plot(res['year'], res['n_infectious'])
        axes[1, 0].set_title('N Infected')
        axes[1, 1].plot(res['year'], res['n_cancerous'])
        axes[1, 1].set_title('N Cancerous')
        for ax in axes.flat:
            ax.set_xlabel('Year')
        plt.tight_layout()

    return sim


@sc.timer()
def test_cancer_emerges(do_plot=do_plot):
    """Check that cancers emerge at all during the simulation."""
    sc.heading('Testing cancer emergence...')
    sim = make_sim(location='nigeria', seed=0)
    sim.run()
    res = extract_v3_results(sim)
    years = np.array(res['year'])

    # Cancers should appear somewhere in the full 70-year run
    total_cancers = np.sum(res['new_cancers'])
    assert total_cancers > 0, (
        f'Expected some cancers over 70 years with 10k agents, got total={total_cancers}'
    )

    # Peak cancer incidence should be in a plausible range (0-200 per 100k)
    max_ci = np.max(res['cancer_incidence'])
    assert 0 < max_ci < 200, (
        f'Expected peak cancer incidence in (0, 200) per 100k, got {max_ci:.2f}'
    )

    if do_plot:
        plt.figure()
        plt.plot(years, res['cancer_incidence'])
        plt.axvline(2010, color='gray', linestyle='--', label='Burnin end')
        plt.xlabel('Year')
        plt.ylabel('Cancer incidence (per 100k)')
        plt.title('Cancer Emergence Over Time')
        plt.legend()

    return sim


@sc.timer()
def test_genotype_distribution(do_plot=do_plot):
    """Check that genotype-specific prevalence is plausible (HPV16 dominant)."""
    sc.heading('Testing genotype distribution...')
    sim = make_sim(location='nigeria', seed=0)
    sim.run()
    res = extract_v3_results(sim)
    years = np.array(res['year'])
    late_mask = years >= 2030

    # HPV16 should have the highest prevalence due to highest rel_beta
    gnames = ['hpv16', 'hpv18', 'hi5', 'ohr']
    mean_prevs = {}
    for gname in gnames:
        key = f'prevalence_{gname}'
        if key in res:
            mean_prevs[gname] = np.mean(res[key][late_mask])

    assert len(mean_prevs) > 0, 'Expected genotype-level prevalence data'

    if 'hpv16' in mean_prevs and 'hpv18' in mean_prevs:
        assert mean_prevs['hpv16'] >= mean_prevs['hpv18'], (
            f'Expected HPV16 prevalence >= HPV18, got {mean_prevs["hpv16"]:.4f} vs {mean_prevs["hpv18"]:.4f}'
        )

    if do_plot:
        plt.figure()
        for gname in gnames:
            key = f'prevalence_{gname}'
            if key in res:
                plt.plot(years, res[key], label=gname)
        plt.xlabel('Year')
        plt.ylabel('Prevalence')
        plt.title('Genotype-Specific Prevalence')
        plt.legend()

    return sim


@pytest.mark.slow
@sc.timer()
def test_prevalence_validation(do_plot=do_plot):
    """
    Validate HPV prevalence against v2 baselines for multiple locations.

    Acceptance: overlapping 95% CIs for 2+ locations, measured as
    >50% of post-burnin time points overlapping.
    """
    sc.heading('Validating HPV prevalence against v2...')
    v2_baselines = load_v2_baselines('natural_history')
    locations_passing = []

    for location in locations:
        v3_results = run_seeds(location=location, n_seeds=n_seeds)
        result = compare_metric(
            v2_baselines, v3_results,
            v2_key='hpv_prevalence', v3_key='hpv_prevalence',
            location=location,
            year_start=year_start, year_end=year_end,
        )
        frac = result['overlap']['fraction']
        print(f'  {location}: HPV prevalence CI overlap = {frac:.1%}')
        if frac >= overlap_threshold:
            locations_passing.append(location)

        if do_plot:
            _plot_comparison(result, location, 'HPV Prevalence')

    n_pass = len(locations_passing)
    assert n_pass >= 2, (
        f'Expected overlapping CIs for 2+ locations, got {n_pass}: {locations_passing}'
    )
    return locations_passing


@pytest.mark.slow
@sc.timer()
def test_cancer_incidence_validation(do_plot=do_plot):
    """
    Validate cancer incidence against v2 baselines for multiple locations.

    Acceptance: overlapping 95% CIs for 2+ locations.
    """
    sc.heading('Validating cancer incidence against v2...')
    v2_baselines = load_v2_baselines('natural_history')
    locations_passing = []

    for location in locations:
        v3_results = run_seeds(location=location, n_seeds=n_seeds)
        result = compare_metric(
            v2_baselines, v3_results,
            v2_key='cancer_incidence', v3_key='cancer_incidence',
            location=location,
            year_start=year_start, year_end=year_end,
        )
        frac = result['overlap']['fraction']
        print(f'  {location}: cancer incidence CI overlap = {frac:.1%}')
        if frac >= overlap_threshold:
            locations_passing.append(location)

        if do_plot:
            _plot_comparison(result, location, 'Cancer Incidence (per 100k)')

    n_pass = len(locations_passing)
    assert n_pass >= 2, (
        f'Expected overlapping CIs for 2+ locations, got {n_pass}: {locations_passing}'
    )
    return locations_passing


def _plot_comparison(result, location, metric_name):
    """Plot v2 vs v3 comparison with confidence intervals."""
    years = result['years']
    ci_v2 = result['ci_v2']
    ci_v3 = result['ci_v3']

    plt.figure(figsize=(10, 5))
    plt.fill_between(years, ci_v2['lower'], ci_v2['upper'], alpha=0.3, color='blue', label='v2 95% CI')
    plt.plot(years, ci_v2['mean'], 'b-', linewidth=1.5, label='v2 mean')
    plt.fill_between(years, ci_v3['lower'], ci_v3['upper'], alpha=0.3, color='red', label='v3 95% CI')
    plt.plot(years, ci_v3['mean'], 'r-', linewidth=1.5, label='v3 mean')

    frac = result['overlap']['fraction']
    plt.title(f'{metric_name} — {location} (CI overlap: {frac:.0%})')
    plt.xlabel('Year')
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    test_smoke(do_plot=do_plot)
    test_cancer_emerges(do_plot=do_plot)
    test_genotype_distribution(do_plot=do_plot)
    test_prevalence_validation(do_plot=do_plot)
    test_cancer_incidence_validation(do_plot=do_plot)

    T.toc()
    if do_plot:
        plt.show()
