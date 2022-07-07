'''
Tests for single simulations
'''

#%% Imports and settings
import os
import pytest
import sys
import sciris as sc
import numpy as np
import seaborn as sns
import hpvsim as hpv
import hpvsim.parameters as hpvpar
import hpvsim.utils as hpu

do_plot = 1
do_save = 0
hpv16 = hpv.genotype('HPV16')
hpv18 = hpv.genotype('HPV18')

base_pars = {
    'pop_size': 50e3,
    'start': 1985,
    'burnin': 30,
    'end': 2050,
    'genotypes': [hpv16, hpv18],
    'location': 'tanzania',
    'dt': .2,
}


#%% Define the tests



def test_latency(do_plot=False, do_save=False, fig_path=None):
    sc.heading('Test latency')

    hpv16 = hpv.genotype('HPV16')
    hpv18 = hpv.genotype('HPV18')
    verbose = .1
    debug = 0
    n_agents = 50e3

    pars = {
        'pop_size': n_agents,
        'n_years': 60,
        'burnin': 30,
        'start': 1970,
        'genotypes': [hpv16, hpv18],
        'pop_scale' : 25.2e6 / n_agents,
        'location': 'tanzania',
        'dt': .2,
    }


    az = hpv.age_results(
        timepoints=['2030'],
        result_keys=['total_cin_prevalence']
    )


    sim = hpv.Sim(pars=pars, analyzers=[az])
    n_runs = 3

    # Define the scenarios
    scenarios = {
        # 'no_latency': {
        #     'name': 'No latency',
        #     'pars': {
        #     }
        # },
        '50%_latency': {
            'name': f'50% of cleared infection are controlled by body',
            'pars': {
                'hpv_control_prob': 0.5,
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
            'CIN prevalence': [
                'total_cin_prevalence',
            ],
            'Cancers per 100,000 women': [
                'total_cancer_incidence',
            ],
        }
        scens.plot(to_plot=to_plot)
        scens.plot_age_results(plot_type=sns.boxplot)

    return scens


#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    scens3 = test_latency(do_plot=True)

    sc.toc(T)
    print('Done.')
