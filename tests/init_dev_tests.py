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
    hpvhi5 = genotype('hpvhi5')
    pars = {
        'pop_size': 50e3,
        'network': 'basic',
        'genotypes': [hpv16, hpv18, hpvhi5],
        'dt': .1,
        'end': 2035
    }
    sim = Sim(pars=pars)
    sim.run()

    fig, ax = pl.subplots(2, 2, figsize=(10, 10))
    timevec = sim.results['year']

    for i, genotype in sim['genotype_map'].items():
        ax[0,0].plot(timevec, sim.results['hpv_incidence'].values[i,:], label=genotype)
        ax[0,1].plot(timevec, sim.results['hpv_prevalence'].values[i, :])
        ax[1,0].plot(timevec, sim.results['new_precancers'].values[i,:])
        ax[1,1].plot(timevec, sim.results['cin_prevalence'].values[i, :])

    ax[0,0].legend()
    ax[0,0].set_title('HPV incidence by genotype')
    ax[0,1].set_title('HPV prevalence by genotype')
    ax[1,0].set_title('New CIN by genotype')
    ax[1,1].set_title('CIN prevalence by genotype')
    fig.show()
    return sim



if __name__ == '__main__':

    # sim0 = test_random() # NOT WORKING
    # sim1 = test_basic() # NOT WORKING
    sim2 = test_genotypes()





