'''
Tests for single simulations
'''

#%% Imports and settings
import numpy as np
import sciris as sc
import seaborn as sns
import hpvsim as hpv

do_plot = 1
do_save = 0


#%% Define the tests
def test_latency(do_plot=False, do_save=False, fig_path=None):
    sc.heading('Test latency')

    verbose = .1
    debug = 0
    n_agents = 1e3

    pars = {
        'n_agents': n_agents,
        'n_years': 60,
        'burnin': 30,
        'start': 1970,
        'genotypes': [16, 18],
        'location': 'tanzania',
        'dt': 1.0,
    }


    az = hpv.age_results(
        sc.objdict(total_hpv_prevalence=sc.objdict(
            timepoints=['2025'],
            edges=np.array([0.,20.,25.,30.,40.,45.,50.,55.,65.,100.]),
        )),
    )


    sim = hpv.Sim(pars=pars, analyzers=[az])
    n_runs = 3

    # Define the scenarios
    scenarios = {
        'no_latency': {
            'name': 'No latency',
            'pars': {
            }
        },
        '50%_latency': {
            'name': '50% of cleared infection are controlled by body',
            'pars': {
                'hpv_control_prob': 0.5,
            }
        },
        '100%_latency': {
            'name': '100% of cleared infection are controlled by body',
            'pars': {
                'hpv_control_prob': 1,
            }
        },
    }

    metapars = {'n_runs': n_runs}

    scens = hpv.Scenarios(sim=sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)
    scens.compare()

    if do_plot:
        to_plot = {
            'HPV prevalence': [
                'total_hpv_prevalence',
            ],
            'Cancers per 100,000 women': [
                'cancer_incidence',
            ],
        }
        # scens.plot(to_plot=to_plot)
        # scens.plot_age_results(plot_type=sns.boxplot)

    return scens


#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    scens3 = test_latency(do_plot=True)

    sc.toc(T)
    print('Done.')
