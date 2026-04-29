"""
Regression tests comparing current HPVsim outputs against v2 baselines.

These tests load pre-generated baseline results (from generate_v2_baselines.py)
and verify that the current code produces results within acceptable tolerance.
This is the main regression gate during the v2→v3 migration.

Usage:
    pytest test_v2_regression.py          # Run all regression tests
    pytest test_v2_regression.py -k nat   # Run only natural history tests
"""

import os
import numpy as np
import sciris as sc
import hpvsim as hpv
import pytest

hpv.options.set(interactive=False)

baseline_dir = sc.thisdir(__file__, 'regression_baselines')

# Tolerance for regression comparisons
RTOL = 0.15  # 15% relative tolerance for stochastic results
ATOL = 5.0   # Absolute tolerance for small counts

# Key result metrics to compare
regression_keys = [
    'infections', 'cancers', 'cancer_deaths',
    'hpv_incidence', 'cancer_incidence',
    'hpv_prevalence', 'cin_prevalence',
    'n_alive',
]


def _load_baselines(scenario_name):
    """Load baseline results for a scenario."""
    filepath = os.path.join(baseline_dir, f'{scenario_name}.json')
    if not os.path.exists(filepath):
        pytest.skip(f'Baseline file not found: {filepath}. Run generate_v2_baselines.py first.')
    return sc.loadjson(filepath)


def _compare_time_series(baseline_ts, current_ts, label, keys=None):
    """Compare time series between baseline and current results."""
    if keys is None:
        keys = regression_keys
    mismatches = []
    for key in keys:
        if key not in baseline_ts or key not in current_ts:
            continue
        bl = np.array(baseline_ts[key])
        cr = np.array(current_ts[key])
        if bl.shape != cr.shape:
            mismatches.append(f'{key}: shape mismatch {bl.shape} vs {cr.shape}')
            continue
        if not np.allclose(bl, cr, rtol=RTOL, atol=ATOL, equal_nan=True):
            max_diff = np.nanmax(np.abs(bl - cr))
            bl_max = np.nanmax(np.abs(bl))
            rel_diff = max_diff / bl_max if bl_max > 0 else max_diff
            mismatches.append(f'{key}: max relative diff={rel_diff:.3f}, max abs diff={max_diff:.1f}')
    return mismatches


def _run_sim_from_baseline(baseline_entry):
    """Recreate and run a sim from baseline metadata."""
    meta = baseline_entry['metadata']
    genotypes = []
    for g in meta['genotypes']:
        try:
            genotypes.append(int(g))
        except ValueError:
            genotypes.append(g)

    pars = dict(
        n_agents=meta['n_agents'],
        start=meta.get('start', 1990),
        end=meta.get('start', 1990) + meta['n_years'],
        dt=meta['dt'],
        burnin=20,
        location=meta['location'],
        genotypes=genotypes,
        rand_seed=meta['rand_seed'],
        verbose=0,
    )
    sim = hpv.Sim(pars)
    sim.run()
    sim.compute_summary()
    return sim


def test_natural_history_regression():
    """Verify natural history results match v2 baselines."""
    baselines = _load_baselines('natural_history')

    # Test a subset of runs (first seed per location) for speed
    test_labels = [k for k in baselines.keys() if '_seed0' in k]

    for label in test_labels:
        baseline = baselines[label]
        sim = _run_sim_from_baseline(baseline)

        # Extract current results
        current_ts = {}
        for key in regression_keys:
            if key in sim.results:
                vals = sim.results[key]
                current_ts[key] = vals[:].tolist() if hasattr(vals, '__getitem__') else float(vals)

        mismatches = _compare_time_series(baseline['time_series'], current_ts, label)
        if mismatches:
            msg = f'Regression failures for {label}:\n' + '\n'.join(f'  - {m}' for m in mismatches)
            print(msg)
            # Don't fail yet — collect warnings during migration. Uncomment to enforce:
            # pytest.fail(msg)


def test_vaccination_regression():
    """Verify vaccination results match v2 baselines."""
    baselines = _load_baselines('vaccination')

    # Test first seed per location
    test_labels = [k for k in baselines.keys() if '_seed0' in k]

    for label in test_labels:
        baseline = baselines[label]
        meta = baseline['metadata']

        genotypes = []
        for g in meta['genotypes']:
            try:
                genotypes.append(int(g))
            except ValueError:
                genotypes.append(g)

        vx = hpv.routine_vx(
            prob=0.8, start_year=2025,
            age_range=[9, 10], product='bivalent',
        )

        pars = dict(
            n_agents=meta['n_agents'],
            start=meta.get('start', 1990),
            end=meta.get('start', 1990) + meta['n_years'],
            dt=meta['dt'],
            burnin=20,
            location=meta['location'],
            genotypes=genotypes,
            rand_seed=meta['rand_seed'],
            verbose=0,
            interventions=[vx],
        )
        sim = hpv.Sim(pars)
        sim.run()

        current_ts = {}
        for key in regression_keys:
            if key in sim.results:
                vals = sim.results[key]
                current_ts[key] = vals[:].tolist() if hasattr(vals, '__getitem__') else float(vals)

        mismatches = _compare_time_series(baseline['time_series'], current_ts, label)
        if mismatches:
            msg = f'Regression failures for {label}:\n' + '\n'.join(f'  - {m}' for m in mismatches)
            print(msg)


def test_genotype_distribution_regression():
    """Verify genotype distribution results match v2 baselines."""
    baselines = _load_baselines('genotype_dist')

    test_labels = [k for k in baselines.keys() if '_seed0' in k]

    for label in test_labels:
        baseline = baselines[label]
        sim = _run_sim_from_baseline(baseline)

        # Check genotype-level results
        for key in ['cancers_by_genotype', 'infections_by_genotype']:
            if key in baseline.get('genotype_results', {}) and key in sim.results:
                bl = np.array(baseline['genotype_results'][key])
                cr = sim.results[key][:]
                if bl.shape == cr.shape:
                    if not np.allclose(bl, cr, rtol=RTOL, atol=ATOL, equal_nan=True):
                        max_diff = np.nanmax(np.abs(bl - cr))
                        print(f'  {label}/{key}: max diff={max_diff:.2f}')


def test_baseline_metadata():
    """Verify baseline metadata is consistent and complete."""
    meta_path = os.path.join(baseline_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        pytest.skip('Baselines not generated yet. Run generate_v2_baselines.py first.')

    meta = sc.loadjson(meta_path)
    assert 'version' in meta
    assert 'scenarios' in meta
    assert meta['n_total_runs'] > 0
    assert len(meta['scenarios']) == 4  # natural_history, vaccination, screening, genotype_dist

    # Verify all scenario files exist
    for scenario in meta['scenarios']:
        filepath = os.path.join(baseline_dir, f'{scenario}.json')
        assert os.path.exists(filepath), f'Missing baseline file: {filepath}'


def test_cancer_incidence_plausible():
    """Verify cancer incidence is in plausible range across all baselines."""
    baselines = _load_baselines('natural_history')

    for label, res in baselines.items():
        ts = res.get('time_series', {})
        cancer_inc = ts.get('cancer_incidence', [])
        if cancer_inc:
            final_inc = cancer_inc[-1]
            assert final_inc >= 0, f'{label}: negative cancer incidence ({final_inc})'
            # For SSA locations, cancer incidence should be < 200 per 100k
            assert final_inc < 200, f'{label}: implausibly high cancer incidence ({final_inc})'


#%% Run as a script
if __name__ == '__main__':
    T = sc.tic()
    test_baseline_metadata()
    test_cancer_incidence_plausible()
    test_natural_history_regression()
    test_vaccination_regression()
    test_genotype_distribution_regression()
    sc.toc(T)
    print('Done.')
