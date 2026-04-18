"""
Test that the current version of HPVsim exactly matches
the baseline results.
"""

import numpy as np
import sciris as sc
import hpvsim as hpv

do_plot = 1
do_save = 0
baseline_filename  = sc.thisdir(__file__, 'baseline.json')
benchmark_filename = sc.thisdir(__file__, 'benchmark.json')
parameters_filename = sc.thisdir(hpv.__file__, 'regression', f'pars_v{hpv.__version__}.json')
hpv.options.set(interactive=False) # Assume not running interactively


def make_sim(use_defaults=False, do_plot=False, **kwargs):
    '''
    Define a default simulation for testing the baseline, including
    interventions to increase coverage. If run directly (not via pytest), also
    plot the sim by default.
    '''

    # Define some interventions
    prob = 0.02
    screen      = hpv.routine_screening(start_year=2040, prob=prob, product='hpv', label='screen')

    to_triage   = lambda sim: sim.get_intervention('screen').outcomes['positive']
    triage      = hpv.routine_triage(eligibility=to_triage, prob=prob, product='via', label='triage')

    to_treat    = lambda sim: sim.get_intervention('triage').outcomes['positive']
    assign_tx   = hpv.routine_triage(eligibility=to_treat, prob=prob, product='tx_assigner', label='assign_tx')

    to_ablate   = lambda sim: sim.get_intervention('assign_tx').outcomes['ablation']
    ablation    = hpv.treat_num(eligibility=to_ablate, prob=prob, product='ablation')

    to_excise   = lambda sim: sim.get_intervention('assign_tx').outcomes['excision']
    excision    = hpv.treat_delay(eligibility=to_excise, prob=prob, product='excision')



    vx          = hpv.routine_vx(prob=prob, start_year=2020, age_range=[9,10], product='bivalent')
    txvx        = hpv.routine_txvx(prob=prob, start_year=2040, product='txvx1')


    # Define the parameters
    pars = dict(
        n_agents      = 10e3,       # Population size
        start         = 2000,       # Starting year
        n_years       = 40,         # Number of years to simulate
        verbose       = 0,          # Don't print details of the run
        rand_seed     = 2,          # Set a non-default seed
        genotypes     = [16, 18],   # Include the two genotypes of greatest general interest
        interventions = [screen, triage, assign_tx, ablation, excision, vx, txvx],  # Include the most common interventions
    )

    # Create the sim
    sim = hpv.Sim(pars)

    # Optionally plot
    if do_plot:
        sim.run()
        sim.plot()

    return sim


def save_baseline():
    '''
    Refresh the baseline results. This function is not called during standard testing,
    but instead is called by the update_baseline script.
    '''

    print('Updating baseline values...')

    # Export default parameters
    s1 = make_sim(use_defaults=True)
    s1.export_pars(filename=parameters_filename) # If not different from previous version, can safely delete

    # Export results
    s2 = make_sim(use_defaults=False)
    s2.run()
    s2.to_json(filename=baseline_filename, keys='summary')

    print('Done.')

    return


def test_baseline():
    ''' Compare the current default sim against the saved baseline '''

    # Load existing baseline
    baseline = sc.loadjson(baseline_filename)
    old = baseline['summary']

    # Calculate new baseline
    new = make_sim()
    new.run()

    # Compute the comparison
    hpv.diff_sims(old, new, full=True, die=True)

    return new


def test_benchmark(do_save=do_save, repeats=1, verbose=True):
    ''' Compare benchmark performance '''

    if verbose: print('Running benchmark...')
    previous = sc.loadjson(benchmark_filename)

    t_inits = []
    t_runs  = []

    def normalize_performance():
        ''' Normalize performance across CPUs '''
        t_bls = []
        bl_repeats = 3
        n_outer = 10
        n_inner = 1e6
        for r in range(bl_repeats):
            t0 = sc.tic()
            for i in range(n_outer):
                a = np.random.random(int(n_inner))
                b = np.random.random(int(n_inner))
                a*b
            t_bl = sc.toc(t0, output=True)
            t_bls.append(t_bl)
        t_bl = min(t_bls)
        reference = 0.07 # Benchmarked on an Intel i7-12700H CPU @ 2.90GHz
        ratio = reference/t_bl
        return ratio


    # Test CPU performance before the run
    r1 = normalize_performance()

    # Do the actual benchmarking
    for r in range(repeats):

        # Create the sim
        sim = make_sim(verbose=0)

        # Time initialization
        t0 = sc.tic()
        sim.initialize()
        t_init = sc.toc(t0, output=True)

        # Time running
        t0 = sc.tic()
        sim.run()
        t_run = sc.toc(t0, output=True)

        # Store results
        t_inits.append(t_init)
        t_runs.append(t_run)

    # Test CPU performance after the run
    r2 = normalize_performance()
    ratio = (r1+r2)/2
    t_init = min(t_inits)*ratio
    t_run  = min(t_runs)*ratio

    # Construct json
    n_decimals = 3
    json = {'time': {
                'initialize': round(t_init, n_decimals),
                'run':        round(t_run,  n_decimals),
                },
            'parameters': {
                'n_agents': sim['n_agents'],
                'n_genotypes': sim['n_genotypes'],
                'n_years':   sim['n_years'],
                'n_interventions': len(sim['interventions']),
                'n_analyzers': len(sim['analyzers']),
                },
            'cpu_performance': ratio,
            }

    if verbose:
        print('Previous benchmark:')
        sc.pp(previous)

        print('\nNew benchmark:')
        sc.pp(json)
    else:
        brief = sc.dcp(json['time'])
        brief['cpu_performance'] = json['cpu_performance']
        sc.pp(brief)

    if do_save:
        sc.savejson(filename=benchmark_filename, obj=json, indent=2)

    if verbose:
        print('Done.')

    return json



if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    hpv.options.set(interactive=do_plot)
    T = sc.tic()

    json = test_benchmark(do_save=do_save, repeats=5) # Run this first so benchmarking is available even if results are different
    new  = test_baseline()
    make_sim(do_plot=do_plot)

    print('\n'*2)
    sc.toc(T)
    print('Done.')
