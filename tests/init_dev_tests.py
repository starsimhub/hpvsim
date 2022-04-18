''' Dev tests '''

import numpy as np
import sciris as sc
import pylab as pl
import sys
import os

# Add module to paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def test_random():
    ''' Make the simplest possible sim with one kind of partnership '''
    from hpvsim.sim import Sim
    from hpvsim.analysis import snapshot
    pars = {'pop_size':20_000, 'rand_seed':100, 'location':'zimbabwe'}
    sim = Sim(pars=pars, analyzers=snapshot('2015', '2020'))
    sim.run()
    return sim

def test_basic():
    ''' Make a sim with two kinds of partnership, regular and casual '''
    from hpvsim.sim import Sim
    pars = {'network':'basic'}
    sim = Sim(pars=pars)
    sim.run()
    return sim


if __name__ == '__main__':

    sim0 = test_random()
    # sim1 = test_basic()

    snapshot = sim0['analyzers'][0]
    people_2015 = snapshot.snapshots[0]
    people_2020 = snapshot.snapshots[1]

    people_2015.plot()
    people_2020.plot()








