"""
Generate baseline results for project validation.

Runs each project configuration across multiple seeds and saves results as JSON
for later comparison. Uses project definitions from the local repos module.

Usage:
    python generate_baselines.py
    python generate_baselines.py --label v2_branch
"""

import argparse
import numpy as np
import sciris as sc
import hpvsim as hpv

from repos import clone_repos, make_project_sim, load_project_pars, PROJECT_REPOS

# Configuration
n_seeds = 3

# Result keys to save for time series
ts_keys = [
    'infections', 'dysplasias', 'precins', 'cins',
    'cancers', 'cancer_deaths', 'detected_cancers',
    'hpv_incidence', 'cancer_incidence', 'asr_cancer_incidence',
    'hpv_prevalence', 'cin_prevalence',
    'n_alive', 'n_infectious', 'n_cancerous',
    'births', 'other_deaths',
]
genotype_keys = [
    'infections_by_genotype', 'cancers_by_genotype',
    'cancer_deaths_by_genotype', 'hpv_incidence_by_genotype',
    'cancer_incidence_by_genotype', 'n_infectious_by_genotype',
    'n_cancerous_by_genotype',
]
age_keys = [
    'infections_by_age', 'cancers_by_age', 'cancer_deaths_by_age',
    'hpv_incidence_by_age', 'cancer_incidence_by_age',
]


def extract_results(sim, label=''):
    """Extract a standardized dict of results from a completed sim."""
    res = dict()

    # Metadata
    res['metadata'] = dict(
        version=hpv.__version__,
        label=label,
        location=sim['location'],
        n_agents=sim['n_agents'],
        dt=sim['dt'],
        start=sim['start'],
        n_years=sim['n_years'],
        genotypes=[str(g) for g in sim['genotypes']],
        rand_seed=sim['rand_seed'],
    )

    # Time vector
    res['year'] = sim.results['year'].tolist()

    # Time series
    res['time_series'] = {}
    for key in ts_keys:
        if key in sim.results:
            vals = sim.results[key]
            res['time_series'][key] = vals[:].tolist() if hasattr(vals, '__getitem__') else float(vals)

    # Genotype-level results
    res['genotype_results'] = {}
    for key in genotype_keys:
        if key in sim.results:
            res['genotype_results'][key] = sim.results[key][:].tolist()

    # Age-stratified results
    res['age_results'] = {}
    for key in age_keys:
        if key in sim.results:
            res['age_results'][key] = sim.results[key][:].tolist()

    # Summary statistics
    res['summary'] = {}
    if hasattr(sim, 'summary') and sim.summary is not None:
        for key, val in sim.summary.items():
            try:
                res['summary'][key] = float(val)
            except (TypeError, ValueError):
                res['summary'][key] = str(val)

    return res


def generate_baselines(label='v2_main'):
    """Run all project sims across seeds and save baselines."""
    sc.tic()

    basedir = sc.thisdir(__file__)
    output_dir = sc.path(basedir) / 'baselines' / label

    hpv.options.set(interactive=False)

    print(f'Generating project baselines with HPVsim {hpv.__version__}')
    print(f'Label: {label}')
    print(f'Output directory: {output_dir}')
    print(f'Seeds: {n_seeds}')
    print(f'Projects: {list(PROJECT_REPOS.keys())}')

    # Clone repos if needed
    print('\nCloning/updating project repos...')
    clone_repos()

    # Run each project across seeds
    for project_name in PROJECT_REPOS:
        print(f'\n{"="*60}')
        print(f'Project: {project_name}')
        print(f'{"="*60}')

        project_results = {}

        for seed in range(n_seeds):
            run_label = f'{project_name}_seed{seed}'
            print(f'  Running {run_label}...')

            try:
                sim = make_project_sim(project_name, seed=seed)
                sim.run()
                if hasattr(sim, 'compute_summary'):
                    sim.compute_summary()
                project_results[run_label] = extract_results(sim, label=run_label)
                print(f'    Done.')
            except Exception as e:
                print(f'    ERROR: {e}')
                project_results[run_label] = dict(error=str(e), label=run_label)

        # Save per-project JSON
        sc.path(output_dir).mkdir(parents=True, exist_ok=True)
        filename = sc.path(output_dir) / f'{project_name}.json'
        sc.savejson(filename=str(filename), obj=project_results, indent=2)
        print(f'\nSaved {filename} ({len(project_results)} runs)')

    sc.toc()
    print(f'\nDone! Generated baselines for {len(PROJECT_REPOS)} projects x {n_seeds} seeds.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate project validation baselines')
    parser.add_argument('--label', type=str, default='v2_main',
                        help='Label for this baseline set (e.g., v2_main, v2_branch)')
    args = parser.parse_args()
    generate_baselines(label=args.label)
