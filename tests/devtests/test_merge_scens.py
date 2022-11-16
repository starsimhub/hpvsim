'''
Scenario merging
''' 
import sciris as sc
import numpy as np
import hpvsim as hpv
import matplotlib.pyplot as plt


do_plot = 1
do_save = 0


# %% Define the tests
def test_merge_scens(do_plot=True):
    sc.heading('Test merging scenarios')

    n_agents = 1000

    basepars = {'n_agents': n_agents}

    # First scenario set
    scenarios1 = {
        'baseline_scen': {
            'name': 'Baseline scenario',
            'pars': {}
        }
    }
    scens1 = hpv.Scenarios(basepars=basepars, scenarios=scenarios1)
    scens1.run()

    # Second scenario set
    scenarios2 = {
        'second_scen': {
            'name': 'Second scenario',
            'pars': {
                'genotypes': [16,18]
            }
        }
    }
    scens2 = hpv.Scenarios(basepars=basepars, scenarios=scenarios2)
    scens2.run()

    # Merge scenarios and plot
    merged_scens = hpv.Scenarios.merge(scens1, scens2)
    merged_scens.plot()

    return merged_scens




# %% Run as a script
if __name__ == '__main__':
    T = sc.tic()

    scens = test_merge_scens()

    sc.toc(T)
    print('Done.')
