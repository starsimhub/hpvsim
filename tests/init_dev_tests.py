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


def test_genotypes():
    ''' Make a sim with two kinds of partnership, regular and casual and 2 HPV genotypes'''
    from hpvsim.sim import Sim
    from hpvsim.immunity import genotype

    hpv16 = genotype('HPV16')
    hpv18 = genotype('HPV18')
    pars = {
        'network': 'basic',
        'genotypes': [hpv16, hpv18]
    }
    sim = Sim(pars=pars)
    sim.run()

    fig, ax = pl.subplots(2, 1, figsize=(8, 12))
    timevec = sim.results['year']
    ax[0].plot(timevec, sim.results['n_alive'].values)
    for i, genotype in sim['genotype_map'].items():
        ax[1].plot(timevec, sim.results['genotype']['new_infections_by_genotype'].values[i,:], label=genotype)
    ax[1].legend()

    fig.show()
    return sim


if __name__ == '__main__':

    # sim0 = test_random()
    # sim1 = test_basic()
    sim2 = test_genotypes()

    # sim0.people.story(40)

    # # Check that population growth is about right
    # pop_growth = (sim0.results['n_alive'][1:]/sim0.results['n_alive'][:-1]-1)/sim0['dt']

    # # Plot people
    # snapshot = sim0['analyzers'][0]
    # people_2015 = snapshot.snapshots[0]
    # people_2020 = snapshot.snapshots[1]
    # people_2015.plot()
    # people_2020.plot()










