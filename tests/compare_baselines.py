"""
V2-V3 baseline comparison utility (GitHub issue #36).

Provides functions for loading v2 baselines, extracting v3 results in a
comparable format, and checking overlapping confidence intervals.

Builds on the v2 regression test patterns from test_v2_regression.py.
"""

import os
import numpy as np
import sciris as sc
from scipy import stats

baseline_dir = sc.thisdir(__file__, 'regression_baselines')


def load_v2_baselines(scenario='natural_history'):
    """Load v2 baseline results from JSON."""
    filepath = os.path.join(baseline_dir, f'{scenario}.json')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Baseline file not found: {filepath}. Run generate_v2_baselines.py first.')
    return sc.loadjson(filepath)


def resample_to_annual(timevec, values, target_years=None):
    """
    Resample sub-annual disease module results to annual resolution.

    V3 disease modules store results at ~monthly resolution; v2 baselines
    are annual. This finds the nearest time point to each target year.
    """
    tv = np.asarray(timevec, dtype=float)
    vals = np.asarray(values, dtype=float)
    if target_years is None:
        target_years = np.arange(int(np.ceil(tv[0])), int(np.floor(tv[-1])) + 1)
    indices = [np.argmin(np.abs(tv - yr)) for yr in target_years]
    return target_years, vals[indices]


def extract_v3_results(sim):
    """
    Extract v3 results in a dict comparable to v2 baselines.

    Focuses on rate-based metrics that are scale-independent since v3
    doesn't yet implement population scaling.

    Returns dict with keys matching v2 time_series where possible.
    """
    hpv_r = sim.results['hpv']
    tv = np.array([float(t) for t in hpv_r['timevec']])
    target_years = np.arange(int(np.ceil(tv[0])), int(np.floor(tv[-1])) + 1)

    res = dict()
    res['year'] = target_years.tolist()

    # Map v2 key names → v3 HPV connector key names
    key_map = dict(
        hpv_prevalence  = 'prevalence',
        cancer_incidence = 'cancer_incidence',
        new_cancers     = 'new_cancers',
        new_cins        = 'new_cins',
        n_cancerous     = 'n_cancerous',
        n_infectious    = 'n_infected',
    )

    for v2_name, v3_name in key_map.items():
        if v3_name in hpv_r:
            _, annual = resample_to_annual(tv, np.array(hpv_r[v3_name]), target_years)
            res[v2_name] = annual

    # Per-genotype metrics
    genotype_names = list(sim.diseases.keys())
    for gname in genotype_names:
        gr = sim.results[gname]
        gtv = np.array([float(t) for t in gr['timevec']])
        for key in ['prevalence', 'cancer_incidence', 'new_infections']:
            if key in gr:
                _, annual = resample_to_annual(gtv, np.array(gr[key]), target_years)
                res[f'{key}_{gname}'] = annual

    # Demographics from sim-level results
    sim_tv = np.array([float(t) for t in sim.results['timevec']])
    if 'n_alive' in sim.results:
        _, annual = resample_to_annual(sim_tv, np.array(sim.results['n_alive']), target_years)
        res['n_alive'] = annual

    return res


def compute_ci(values_per_seed, alpha=0.05):
    """
    Compute mean and confidence interval across seeds using t-distribution.
    """
    stacked = np.array(values_per_seed)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0, ddof=1)
    n = stacked.shape[0]
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_crit * std / np.sqrt(n)
    return dict(mean=mean, lower=mean - margin, upper=mean + margin)


def check_ci_overlap(ci_v2, ci_v3):
    """Check whether confidence intervals overlap at each time point."""
    overlap = (ci_v2['lower'] <= ci_v3['upper']) & (ci_v3['lower'] <= ci_v2['upper'])
    return dict(overlap=overlap, fraction=np.mean(overlap))


def compare_metric(v2_baselines, v3_results_per_seed, v2_key, v3_key,
                   location, n_seeds=5, year_start=2010, year_end=2060):
    """
    Compare a single metric between v2 baselines and v3 results.

    Args:
        v2_baselines: loaded v2 baseline dict (from load_v2_baselines)
        v3_results_per_seed: list of v3 result dicts (from extract_v3_results)
        v2_key: key in v2 time_series dict
        v3_key: key in v3 extracted results
        location: location name for filtering v2 baselines
        n_seeds: number of seeds in v2 baselines
        year_start: first year for comparison (after burnin)
        year_end: last year for comparison

    Returns dict with ci_v2, ci_v3, overlap info, and years.
    """
    # Extract v2 values
    v2_values = []
    for seed in range(n_seeds):
        label = f'natural_history_{location}_seed{seed}'
        entry = v2_baselines[label]
        years_v2 = np.array(entry['year'])
        mask = (years_v2 >= year_start) & (years_v2 <= year_end)
        ts = np.array(entry['time_series'][v2_key])
        v2_values.append(ts[mask])

    # Extract v3 values
    v3_values = []
    for v3_res in v3_results_per_seed:
        years_v3 = np.array(v3_res['year'])
        mask = (years_v3 >= year_start) & (years_v3 <= year_end)
        v3_values.append(v3_res[v3_key][mask])

    ci_v2 = compute_ci(v2_values)
    ci_v3 = compute_ci(v3_values)
    overlap = check_ci_overlap(ci_v2, ci_v3)

    comparison_years = years_v2[mask] if len(years_v2) else np.array([])
    return dict(ci_v2=ci_v2, ci_v3=ci_v3, overlap=overlap, years=comparison_years)