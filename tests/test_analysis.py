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
def test_snapshot():

    sc.heading('Testing snapshot analyzer')
    import hpvsim.analysis as hpa
    import hpvsim.sim as hps

    pars = dict(n_years=10, dt=0.5)

    sim = hps.Sim(pars, analyzers=hpa.snapshot('2016', '2019'))
    sim.run()
    snapshot = sim.get_analyzer()
    people1 = snapshot.snapshots[0]         # Option 1
    people2 = snapshot.snapshots['2016.0']  # Option 2
    people3 = snapshot.get('2019.0')        # Option 3
    people4 = snapshot.get()                # Option 5

    assert people1 == people2, 'Snapshot options should match but do not'
    assert people3 != people4, 'Snapshot options should not match but do'
    return people4


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    people = test_snapshot()

    sc.toc(T)
    print('Done.')
