'''
Tests for single simulations
'''

#%% Imports and settings
import sciris as sc
import numpy as np
import seaborn as sns
import hpvsim as hpv
from hpvsim.data import get_data as data

do_plot = 0
do_save = 0

n_agents = [2e3,50e3][1] # Swap between sizes


#%% Define the tests

def test_hiv(model_hiv=True):
    sc.heading('Testing hiv')

    pars = {
        'n_agents': n_agents,
        'start': 1990,
        'burnin': 30,
        'end': 2050,
        'genotypes': [16, 18],
        'location': 'south africa',
        'dt': .5,
        'model_hiv': model_hiv
    }

    sim = hpv.Sim(pars=pars)
    sim.run()
    sim.plot(to_plot=['hiv_prevalence'])
    return sim

def test_impact_on_cancer():
    sc.heading('Testing hiv')

    pars = {
        'n_agents': n_agents,
        'start': 1970,
        'burnin': 30,
        'end': 2050,
        'genotypes': [16, 18],
        'location': 'south africa',
        'dt': .5,
    }

    base_sim = hpv.Sim(pars=pars)

    scenarios = {
        'no_hiv': {
            'name': 'No HIV',
            'pars': {
                'model_hiv': False
            }
        },
        'hiv_baseline': {
            'name': 'HIV, baseline',
            'pars': {
                'model_hiv': True
            }
        },
        # 'hiv_elevated_risk': {
        #     'name': 'HIV, elevated risk',
        #     'pars': {
        #         'model_hiv': True,
        #         'hiv_pars': {
        #             'rel_sus': 3,
        #             'dysp_rate': 5,
        #             'prog_rate': 5,
        #             'prog_time': 1/5,
        #         }
        #     }
        # }
    }

    metapars = {'n_runs': 3}
    scens = hpv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run()
    to_plot = {
        'HIV prevalence': [
            'hiv_prevalence',
        ],
        'HPV prevalence': [
            'total_hpv_prevalence',
        ],
        'Age standardized cancer incidence (per 100,000 women)': [
            'asr_cancer',
        ],
        'Cancer deaths per 100,000 women': [
            'cancer_mortality',
        ],
    }
    scens.plot(to_plot=to_plot)
    return scens

#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()
    # sim0 = test_hiv(model_hiv=True)
    # sim1 = test_hiv(model_hiv=False)
    sim2 = test_impact_on_cancer()
    sc.toc(T)
    print('Done.')
