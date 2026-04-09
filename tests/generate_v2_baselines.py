"""
Generate v2 baseline outputs for regression testing (GitHub issue #35).

Runs HPVsim v2 across multiple scenarios and seeds, saving results as JSON
for later comparison against v3. This script should be run once on the v2
codebase before starting the v3 migration.

Usage:
    python generate_v2_baselines.py
"""

import numpy as np
import sciris as sc
import hpvsim as hpv

# Configuration
output_dir = sc.thisdir(__file__, 'regression_baselines')
n_seeds = 5
locations = ['nigeria', 'india', 'south africa']
genotypes = [16, 18, 'hi5', 'ohr']
snapshot_years = [2020, 2040, 2060]

# Result keys to save for time series
ts_keys = [
    'infections', 'dysplasias', 'precins', 'cins',
    'cancers', 'cancer_deaths', 'detected_cancers',
    'hpv_incidence', 'cancer_incidence', 'asr_cancer_incidence',
    'hpv_prevalence', 'cin_prevalence',
    'n_alive', 'n_infectious', 'n_cancerous',
    'births', 'other_deaths',
]

# Genotype-level keys
genotype_keys = [
    'infections_by_genotype', 'cancers_by_genotype',
    'cancer_deaths_by_genotype', 'hpv_incidence_by_genotype',
    'cancer_incidence_by_genotype', 'n_infectious_by_genotype',
    'n_cancerous_by_genotype',
]

# Age-stratified keys
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


def scenario_natural_history():
    """Scenario 1: Natural history with no interventions."""
    print('\n' + '='*60)
    print('Scenario 1: Natural history (no interventions)')
    print('='*60)

    all_results = {}
    for location in locations:
        for seed in range(n_seeds):
            label = f'natural_history_{location}_seed{seed}'
            print(f'  Running {label}...')
            pars = dict(
                n_agents=10e3,
                start=1990,
                end=2060,
                dt=0.5,
                burnin=20,
                location=location,
                genotypes=genotypes,
                rand_seed=seed,
                verbose=0,
            )
            sim = hpv.Sim(pars)
            sim.run()
            sim.compute_summary()
            all_results[label] = extract_results(sim, label=label)

    return all_results


def scenario_vaccination():
    """Scenario 2: Routine vaccination with bivalent vaccine."""
    print('\n' + '='*60)
    print('Scenario 2: Vaccination')
    print('='*60)

    all_results = {}
    for location in locations:
        for seed in range(n_seeds):
            label = f'vaccination_{location}_seed{seed}'
            print(f'  Running {label}...')

            vx = hpv.routine_vx(
                prob=0.8,
                start_year=2025,
                age_range=[9, 10],
                product='bivalent',
            )

            pars = dict(
                n_agents=10e3,
                start=1990,
                end=2060,
                dt=0.5,
                burnin=20,
                location=location,
                genotypes=genotypes,
                rand_seed=seed,
                verbose=0,
                interventions=[vx],
            )
            sim = hpv.Sim(pars)
            sim.run()
            sim.compute_summary()
            all_results[label] = extract_results(sim, label=label)

    return all_results


def make_who_algorithm(algo_num, primary_screen_prob=0.2, triage_screen_prob=0.9,
                       ablate_prob=0.9, start_year=2025):
    """Create WHO screening algorithm interventions. Based on examples/t05_screen_algorithms.py.

    Each algorithm uses a unique prefix (a1_, a2_, etc.) for intervention labels
    to avoid collisions when multiple algorithms are compared.
    """
    p = f'a{algo_num}_'  # Unique prefix per algorithm

    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | (sim.t > (sim.people.date_screened + 5 / sim['dt']))

    if algo_num == 1:
        via_primary = hpv.routine_screening(
            product='via', prob=primary_screen_prob, eligibility=screen_eligible,
            start_year=start_year, label=f'{p}via primary')
        via_positive = lambda sim: sim.get_intervention(f'{p}via primary').outcomes['positive']
        ablation = hpv.treat_num(prob=ablate_prob, product='ablation',
                                 eligibility=via_positive, label=f'{p}ablation')
        return [via_primary, ablation]

    elif algo_num == 2:
        hpv_primary = hpv.routine_screening(
            product='hpv', prob=primary_screen_prob, eligibility=screen_eligible,
            start_year=start_year, label=f'{p}hpv primary')
        hpv_positive = lambda sim: sim.get_intervention(f'{p}hpv primary').outcomes['positive']
        ablation = hpv.treat_num(prob=ablate_prob, product='ablation',
                                 eligibility=hpv_positive, label=f'{p}ablation')
        return [hpv_primary, ablation]

    elif algo_num == 3:
        cytology = hpv.routine_screening(
            product='lbc', prob=primary_screen_prob, eligibility=screen_eligible,
            start_year=start_year, label=f'{p}cytology')
        ascus = lambda sim: sim.get_intervention(f'{p}cytology').outcomes['ascus']
        hpv_triage = hpv.routine_triage(product='hpv', prob=triage_screen_prob,
                                         annual_prob=False, eligibility=ascus, label=f'{p}hpv triage')
        to_colpo = lambda sim: list(set(
            sim.get_intervention(f'{p}cytology').outcomes['abnormal'].tolist() +
            sim.get_intervention(f'{p}hpv triage').outcomes['positive'].tolist()))
        colpo = hpv.routine_triage(product='colposcopy', prob=triage_screen_prob,
                                    annual_prob=False, eligibility=to_colpo, label=f'{p}colposcopy')
        hsils = lambda sim: sim.get_intervention(f'{p}colposcopy').outcomes['hsil']
        ablation = hpv.treat_num(prob=ablate_prob, product='ablation',
                                 eligibility=hsils, label=f'{p}ablation')
        return [cytology, hpv_triage, colpo, ablation]

    elif algo_num == 4:
        hpv_primary = hpv.routine_screening(
            product='hpv_type', prob=primary_screen_prob, eligibility=screen_eligible,
            start_year=start_year, label=f'{p}hpv primary')
        pos_ohr = lambda sim: sim.get_intervention(f'{p}hpv primary').outcomes['positive_ohr']
        via_triage = hpv.routine_triage(product='via', prob=triage_screen_prob,
                                         annual_prob=False, eligibility=pos_ohr, label=f'{p}via triage')
        to_assign = lambda sim: list(set(
            sim.get_intervention(f'{p}hpv primary').outcomes['positive_1618'].tolist() +
            sim.get_intervention(f'{p}via triage').outcomes['positive'].tolist()))
        tx_assigner = hpv.routine_triage(product='tx_assigner', prob=triage_screen_prob,
                                          annual_prob=False, eligibility=to_assign, label=f'{p}tx assigner')
        to_ablate = lambda sim: sim.get_intervention(f'{p}tx assigner').outcomes['ablation']
        ablation = hpv.treat_num(prob=ablate_prob, product='ablation',
                                 eligibility=to_ablate, label=f'{p}ablation')
        return [hpv_primary, via_triage, tx_assigner, ablation]

    elif algo_num == 5:
        hpv_primary = hpv.routine_screening(
            product='hpv', prob=primary_screen_prob, eligibility=screen_eligible,
            start_year=start_year, label=f'{p}hpv primary')
        screen_pos = lambda sim: sim.get_intervention(f'{p}hpv primary').outcomes['positive']
        via_triage = hpv.routine_triage(product='via', prob=triage_screen_prob,
                                         annual_prob=False, eligibility=screen_pos, label=f'{p}via triage')
        to_assign = lambda sim: sim.get_intervention(f'{p}via triage').outcomes['positive']
        tx_assigner = hpv.routine_triage(product='tx_assigner', prob=triage_screen_prob,
                                          annual_prob=False, eligibility=to_assign, label=f'{p}tx assigner')
        to_ablate = lambda sim: sim.get_intervention(f'{p}tx assigner').outcomes['ablation']
        ablation = hpv.treat_num(prob=ablate_prob, product='ablation',
                                 eligibility=to_ablate, label=f'{p}ablation')
        return [hpv_primary, via_triage, tx_assigner, ablation]

    elif algo_num == 6:
        hpv_primary = hpv.routine_screening(
            product='hpv', prob=primary_screen_prob, eligibility=screen_eligible,
            start_year=start_year, label=f'{p}hpv primary')
        to_colpo = lambda sim: sim.get_intervention(f'{p}hpv primary').outcomes['positive']
        colpo = hpv.routine_triage(product='colposcopy', prob=triage_screen_prob,
                                    annual_prob=False, eligibility=to_colpo, label=f'{p}colposcopy')
        hsils = lambda sim: sim.get_intervention(f'{p}colposcopy').outcomes['hsil']
        ablation = hpv.treat_num(prob=ablate_prob, product='ablation',
                                 eligibility=hsils, label=f'{p}ablation')
        return [hpv_primary, colpo, ablation]

    elif algo_num == 7:
        hpv_primary = hpv.routine_screening(
            product='hpv', prob=primary_screen_prob, eligibility=screen_eligible,
            start_year=start_year, label=f'{p}hpv primary')
        to_cytology = lambda sim: sim.get_intervention(f'{p}hpv primary').outcomes['positive']
        cytology = hpv.routine_triage(product='lbc', annual_prob=False, prob=triage_screen_prob,
                                       eligibility=to_cytology, label=f'{p}cytology')
        to_colpo = lambda sim: list(set(
            sim.get_intervention(f'{p}cytology').outcomes['abnormal'].tolist() +
            sim.get_intervention(f'{p}cytology').outcomes['ascus'].tolist()))
        colpo = hpv.routine_triage(product='colposcopy', annual_prob=False, prob=triage_screen_prob,
                                    eligibility=to_colpo, label=f'{p}colpo')
        hsils = lambda sim: sim.get_intervention(f'{p}colpo').outcomes['hsil']
        ablation = hpv.treat_num(prob=ablate_prob, product='ablation',
                                 eligibility=hsils, label=f'{p}ablation')
        return [hpv_primary, cytology, colpo, ablation]

    else:
        raise ValueError(f'Unknown algorithm number: {algo_num}')


def scenario_screening():
    """Scenario 3: WHO screening/treatment algorithms."""
    print('\n' + '='*60)
    print('Scenario 3: WHO screening algorithms')
    print('='*60)

    all_results = {}
    location = 'nigeria'  # Use one location for screening algorithms

    for algo_num in range(1, 8):
        for seed in range(n_seeds):
            label = f'screening_algo{algo_num}_seed{seed}'
            print(f'  Running {label}...')

            interventions = make_who_algorithm(algo_num)
            pars = dict(
                n_agents=10e3,
                start=2000,
                end=2060,
                dt=0.5,
                burnin=0,
                location=location,
                genotypes=[16, 18],
                rand_seed=seed,
                verbose=0,
                interventions=interventions,
            )
            sim = hpv.Sim(pars)
            sim.run()
            sim.compute_summary()

            res = extract_results(sim, label=label)

            # Also save intervention-specific results
            res['intervention_results'] = {}
            for intv in sim['interventions']:
                intv_label = intv.label
                if hasattr(intv, 'outcomes'):
                    res['intervention_results'][intv_label] = {
                        k: len(v) if hasattr(v, '__len__') else int(v)
                        for k, v in intv.outcomes.items()
                    }

            all_results[label] = res

    return all_results


def scenario_genotype_dist():
    """Scenario 4: Genotype distribution of cancers over time."""
    print('\n' + '='*60)
    print('Scenario 4: Genotype distribution')
    print('='*60)

    all_results = {}
    for location in locations:
        for seed in range(n_seeds):
            label = f'genotype_dist_{location}_seed{seed}'
            print(f'  Running {label}...')

            pars = dict(
                n_agents=10e3,
                start=1990,
                end=2060,
                dt=0.5,
                burnin=20,
                location=location,
                genotypes=genotypes,
                rand_seed=seed,
                verbose=0,
            )
            sim = hpv.Sim(pars)
            sim.run()
            sim.compute_summary()

            res = extract_results(sim, label=label)

            # Add genotype distribution keys
            for key in ['precin_genotype_dist', 'cin_genotype_dist', 'cancerous_genotype_dist']:
                if key in sim.results:
                    res['genotype_results'][key] = sim.results[key][:].tolist()

            all_results[label] = res

    return all_results


def validate_results(all_results):
    """Run basic sanity checks on the generated baselines."""
    print('\n' + '='*60)
    print('Validating baselines...')
    print('='*60)

    n_warnings = 0
    for label, res in all_results.items():
        ts = res.get('time_series', {})

        # Check cancer incidence is in plausible range (5-50 per 100k for SSA)
        cancer_inc = ts.get('cancer_incidence', [])
        if cancer_inc:
            final_inc = cancer_inc[-1]
            if final_inc < 0:
                print(f'  WARNING: {label}: negative cancer incidence ({final_inc:.2f})')
                n_warnings += 1
            elif final_inc > 200:
                print(f'  WARNING: {label}: very high cancer incidence ({final_inc:.2f})')
                n_warnings += 1

        # Check that infections are non-negative
        infections = ts.get('infections', [])
        if infections and any(v < 0 for v in infections):
            print(f'  WARNING: {label}: negative infection counts')
            n_warnings += 1

    if n_warnings == 0:
        print('  All validations passed!')
    else:
        print(f'  {n_warnings} warnings found')

    return n_warnings


def generate_all():
    """Run all scenarios and save baselines."""
    sc.tic()

    hpv.options.set(interactive=False)

    print(f'Generating v2 baselines with HPVsim {hpv.__version__}')
    print(f'Output directory: {output_dir}')

    # Run all scenarios
    results = {}
    results['natural_history'] = scenario_natural_history()
    results['vaccination'] = scenario_vaccination()
    results['screening'] = scenario_screening()
    results['genotype_dist'] = scenario_genotype_dist()

    # Validate
    all_flat = {}
    for scenario_results in results.values():
        all_flat.update(scenario_results)
    validate_results(all_flat)

    # Save each scenario separately
    for scenario_name, scenario_results in results.items():
        filename = sc.path(output_dir) / f'{scenario_name}.json'
        sc.savejson(filename=str(filename), obj=scenario_results, indent=2)
        print(f'Saved {filename} ({len(scenario_results)} runs)')

    # Save a combined metadata file
    meta = dict(
        version=hpv.__version__,
        n_seeds=n_seeds,
        locations=locations,
        genotypes=[str(g) for g in genotypes],
        snapshot_years=snapshot_years,
        scenarios=list(results.keys()),
        n_total_runs=sum(len(v) for v in results.values()),
    )
    sc.savejson(filename=str(sc.path(output_dir) / 'metadata.json'), obj=meta, indent=2)

    sc.toc()
    print(f'\nDone! Generated baselines for {meta["n_total_runs"]} runs across {len(results)} scenarios.')


if __name__ == '__main__':
    generate_all()
