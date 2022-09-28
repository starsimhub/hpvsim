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

#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()
    sim0 = test_hiv(model_hiv=True)
    sc.toc(T)
    T = sc.tic()
    sim1 = test_hiv(model_hiv=False)
    sc.toc(T)
    print('Done.')
