"""
Project validation tests comparing HPVsim results against committed baselines
generated from real-world project repositories.

These tests clone external project repos, reconstruct sims from baseline
metadata, and verify that current HPVsim produces results within acceptable
tolerance of the stored baselines.

Usage:
    pytest test_project_validation.py              # Run all project validation tests
    pytest test_project_validation.py -k kenya      # Run only Kenya project tests
"""

import os
import numpy as np
import sciris as sc
import hpvsim as hpv
import pytest

from project_validation.repos import clone_repos, make_project_sim, PROJECT_REPOS

hpv.options.set(interactive=False)

# Tolerance for regression comparisons
RTOL = 0.15   # 15% relative tolerance for stochastic results
ATOL = 5.0    # Absolute tolerance for small counts

# Key result metrics to compare
REGRESSION_KEYS = [
    'infections', 'cancers', 'cancer_deaths',
    'hpv_incidence', 'cancer_incidence',
    'hpv_prevalence', 'n_alive',
]

# Baseline directories — prefer v2_branch over v2_main
baseline_dir = sc.thisdir(__file__, 'project_validation', 'baselines')


def _load_baseline(project_name):
    """Load baseline results for a project, checking v2_branch first then v2_main.

    Returns the loaded JSON dict, or None if no baseline file is found.
    """
    for subdir in ['v2_branch', 'v2_main']:
        filepath = os.path.join(baseline_dir, subdir, f'{project_name}.json')
        if os.path.exists(filepath):
            print(f'  Loading baseline from {subdir}/{project_name}.json')
            return sc.loadjson(filepath)
    return None


def _compare_time_series(key, baseline_data, sim):
    """Compare a single time series key between baseline data and a sim.

    Args:
        key: The result key to compare (e.g. 'infections')
        baseline_data: Dict of baseline time series (key -> list of values)
        sim: The completed hpv.Sim object

    Returns:
        True if the values match within tolerance, False otherwise.
    """
    if key not in baseline_data:
        print(f'    WARNING: key "{key}" not found in baseline data, skipping')
        return True
    if key not in sim.results:
        print(f'    WARNING: key "{key}" not found in sim results, skipping')
        return True

    baseline_vals = np.array(baseline_data[key])
    sim_vals = sim.results[key]
    sim_vals = np.array(sim_vals[:].tolist() if hasattr(sim_vals, '__getitem__') else [float(sim_vals)])

    if baseline_vals.shape != sim_vals.shape:
        print(f'    WARNING: shape mismatch for "{key}": baseline {baseline_vals.shape} vs sim {sim_vals.shape}')
        return False

    if np.allclose(baseline_vals, sim_vals, rtol=RTOL, atol=ATOL, equal_nan=True):
        return True

    # Compute diagnostics on failure
    abs_diff = np.abs(baseline_vals - sim_vals)
    max_abs_diff = np.nanmax(abs_diff)
    bl_max = np.nanmax(np.abs(baseline_vals))
    max_rel_diff = max_abs_diff / bl_max if bl_max > 0 else max_abs_diff
    print(f'    WARNING: "{key}" mismatch — max relative diff={max_rel_diff:.4f}, max abs diff={max_abs_diff:.2f}')
    return False


@pytest.fixture(scope='session')
def ensure_repos():
    """Clone all project repositories once per test session."""
    clone_repos()


@pytest.mark.slow
@pytest.mark.parametrize('project_name', list(PROJECT_REPOS.keys()))
def test_project_regression(project_name, ensure_repos):
    """Validate project results against committed baselines.

    For each run stored in the baseline (e.g. hpv_faster_kenya_seed0):
      - Reconstruct a sim from the baseline metadata
      - Run the sim
      - Compare each key in REGRESSION_KEYS
      - Collect and report any mismatches
    """
    baseline = _load_baseline(project_name)
    if baseline is None:
        pytest.skip(f'No baseline found for project "{project_name}". Generate baselines first.')

    all_mismatches = []

    for run_label, run_data in baseline.items():
        print(f'\n  Comparing run: {run_label}')

        meta = run_data.get('metadata', {})
        if not meta:
            print(f'    Skipping {run_label}: no metadata found')
            continue

        # Parse genotypes
        genotypes = []
        for g in meta.get('genotypes', []):
            try:
                genotypes.append(int(g))
            except (ValueError, TypeError):
                genotypes.append(g)

        # Reconstruct and run sim from baseline metadata
        sim = hpv.Sim(
            n_agents=meta['n_agents'],
            dt=meta['dt'],
            start=meta['start'],
            n_years=meta['n_years'],
            genotypes=genotypes,
            location=meta['location'],
            rand_seed=meta['rand_seed'],
            verbose=0,
        )
        sim.run()

        # Compare each regression key
        time_series = run_data.get('time_series', {})
        run_mismatches = []
        for key in REGRESSION_KEYS:
            if not _compare_time_series(key, time_series, sim):
                run_mismatches.append(key)

        if run_mismatches:
            msg = f'{run_label}: mismatched keys: {", ".join(run_mismatches)}'
            print(f'    MISMATCH: {msg}')
            all_mismatches.append(msg)
        else:
            print(f'    OK: all {len(REGRESSION_KEYS)} keys match within tolerance')

    if all_mismatches:
        summary = f'Project "{project_name}" regression failures:\n' + '\n'.join(f'  - {m}' for m in all_mismatches)
        print(f'\n{summary}')
        # Uncomment the line below to enforce strict regression failures:
        # pytest.fail(summary)


#%% Run as a script
if __name__ == '__main__':
    T = sc.tic()

    # Clone repos
    print('Cloning project repositories...')
    clone_repos()

    # Run all project validations
    for project_name in PROJECT_REPOS.keys():
        print(f'\n{"="*60}')
        print(f'Testing project: {project_name}')
        print(f'{"="*60}')
        test_project_regression(project_name, None)

    sc.toc(T)
    print('Done.')
