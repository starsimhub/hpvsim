'''
Tests for single simulations
'''

#%% Imports and settings
import os
import pytest
import sys
import sciris as sc
import numpy as np

# Add module to paths and import hpvsim
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import hpvsim.sim as hps

do_plot = 1
do_save = 0


#%% Define the tests

def test_dynamic_pars():
    sc.heading('Test dynamics pars intervention')

    import hpvsim.interventions as hpint
    import hpvsim.sim as hps

    pars = {
        'pop_size': 10e3,
        'n_years': 10,
    }

    # Model an intervention to increase condom use
    condom_int = hpint.dynamic_pars(
        condoms=dict(timepoints=10, vals={'c': 0.9}))  # Increase condom use among casual partners to 90%

    # Model an intervention to increase the age of sexual debut
    debut_int = hpint.dynamic_pars(
        {'debut': {
            'timepoints': '2020',
            'vals': dict(f=dict(dist='normal', par1=20, par2=2.1), # Increase mean age of sexual debut
                         m=dict(dist='normal', par1=19.6,par2=1.8))
        }
        }
    )

    sim = hps.Sim(pars=pars, interventions=[condom_int, debut_int])
    sim.run()
    return sim




#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    sim0 = test_dynamic_pars()

    sc.toc(T)
    print('Done.')
